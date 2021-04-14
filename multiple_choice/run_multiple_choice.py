import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_metric

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

import transformers
from transformers import (
	AutoConfig,
	AutoModelForMultipleChoice,
	AutoTokenizer,
	EvalPrediction,
	HfArgumentParser,
	Trainer,
	TrainingArguments,
	set_seed,
)
from transformers.trainer_utils import is_main_process
from utils_multiple_choice import MultipleChoiceDataset, Split, processors


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""

	task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
	data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
	max_seq_length: int = field(
		default=128,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
	)


def main():
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	# Add custom arguments for computing pre-train loss
	parser.add_argument("--ptl", type=bool, default=False)
	model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

	if (
		os.path.exists(training_args.output_dir)
		and os.listdir(training_args.output_dir)
		and training_args.do_train
		and not training_args.overwrite_output_dir
	):
		raise ValueError(
			f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
		)

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
	)
	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		training_args.local_rank,
		training_args.device,
		training_args.n_gpu,
		bool(training_args.local_rank != -1),
		training_args.fp16,
	)
	# Set the verbosity to info of the Transformers logger (on main process only):
	if is_main_process(training_args.local_rank):
		transformers.utils.logging.set_verbosity_info()
		transformers.utils.logging.enable_default_handler()
		transformers.utils.logging.enable_explicit_format()
	logger.info("Training/evaluation parameters %s", training_args)

	# Set seed
	set_seed(training_args.seed)

	try:
		processor = processors[data_args.task_name]()
		label_list = processor.get_labels()
		num_labels = len(label_list)
	except KeyError:
		raise ValueError("Task not found: %s" % (data_args.task_name))

	# Load pretrained model and tokenizer
	config = AutoConfig.from_pretrained(
		model_args.config_name if model_args.config_name else model_args.model_name_or_path,
		num_labels=num_labels,
		finetuning_task=data_args.task_name,
		cache_dir=model_args.cache_dir,
	)
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		# Default fast tokenizer is buggy on CaseHOLD task, switch to legacy tokenizer
		use_fast=False,
	)
	model = AutoModelForMultipleChoice.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
	)

	train_dataset = None
	eval_dataset = None

	# Get datasets
	# See utils_multiple_choice.py for more details on dataset processing and to change behavior
	# If ptl=True passed, default to computing pre-train loss, otherwise do training/evaluation/prediction
	if not custom_args.ptl:
		# If do_train passed, train_dataset by default loads train split from file named train.csv in data directory
		if training_args.do_train:
			train_dataset = \
				MultipleChoiceDataset(
					data_dir=data_args.data_dir,
					tokenizer=tokenizer,
					task=data_args.task_name,
					max_seq_length=data_args.max_seq_length,
					overwrite_cache=data_args.overwrite_cache,
					mode=Split.train,
				)

		# If do_eval or do_predict passed, eval_dataset by default loads dev split from file named dev.csv in data directory
		if training_args.do_eval or training_args.do_predict:
			eval_dataset = \
				MultipleChoiceDataset(
					data_dir=data_args.data_dir,
					tokenizer=tokenizer,
					task=data_args.task_name,
					max_seq_length=data_args.max_seq_length,
					overwrite_cache=data_args.overwrite_cache,
					# Pass mode=Split.test to load test split from file named test.csv in data directory
					mode=Split.dev,
				)
	# If ptl=True passed, eval_dataset by default loads full dataset from file named all.csv in data directory
	else:
		eval_dataset = \
			MultipleChoiceDataset(
				data_dir=data_args.data_dir,
				tokenizer=tokenizer,
				task=data_args.task_name,
				max_seq_length=data_args.max_seq_length,
				overwrite_cache=data_args.overwrite_cache,
				mode=Split.full,
			)

	# Define custom compute_metrics function, returns macro F1 metric for CaseHOLD task
	def compute_metrics(p: EvalPrediction):
		preds = np.argmax(p.predictions, axis=1)
		metric = load_metric("f1")
		# Compute macro F1 for 5-class CaseHOLD task
		f1 = metric.compute(predictions=preds, references=p.label_ids, average='macro')
		return f1

	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		compute_metrics=compute_metrics,
	)

	if not custom_args.ptl:
		# Training
		if training_args.do_train:
			trainer.train(
				model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
			)
			trainer.save_model()
			# Re-save the tokenizer for model sharing
			if trainer.is_world_process_zero():
				tokenizer.save_pretrained(training_args.output_dir)

		# Evaluation on eval_dataset
		eval_results = {}
		if training_args.do_eval:
			logger.info("*** Evaluate ***")

			eval_result = trainer.evaluate()

			output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
			if trainer.is_world_process_zero():
				with open(output_eval_file, "w") as writer:
					logger.info("***** Eval results *****")
					for key, value in eval_result.items():
						logger.info("  %s = %s", key, value)
						writer.write("%s = %s\n" % (key, value))

			eval_results.update(eval_result)

		# Predict on eval_dataset
		if training_args.do_predict:
			logger.info("*** Predict ***")

			predictions = trainer.predict(test_dataset=eval_dataset).predictions
			predictions = np.argmax(predictions, axis=1)

			output_preds_file = os.path.join(training_args.output_dir, "predictions.csv")
			if trainer.is_world_process_zero():
				np.savetxt(output_preds_file, predictions, delimiter=',', fmt='%.4e')
				
		return eval_results
	# If ptl=True passed, compute per example/average pre-train loss on eval_dataset (by default loads full dataset from file named all.csv in data directory)
	# See utils_multiple_choice.py for more details on dataset processing and to change behavior
	else:
		logger.info("*** Compute per example pre-train loss ***")

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		pretrain_losses = []
		# Maintains original ordering of dataset
		eval_loader = trainer.get_eval_dataloader(eval_dataset)
		for batch in eval_loader:
			inputs = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			token_type_ids = batch['token_type_ids'].to(device)
			labels = batch['labels'].to(device)
			outputs = model(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
			loss = F.cross_entropy(outputs.logits, labels, reduction='none')
			pretrain_losses += loss.tolist()

		per_ex_pretrain_loss = np.array(pretrain_losses)
		avg_pretrain_loss = per_ex_pretrain_loss.mean().item()
		print("Average pre-train loss:", avg_pretrain_loss)

		output_test_file = os.path.join(training_args.output_dir, "per_ex_pretrain_loss.csv")
		if trainer.is_world_process_zero():
			np.savetxt(output_test_file, per_ex_pretrain_loss, delimiter=',', fmt='%.4e')

		return avg_pretrain_loss


def _mp_fn(index):
	# For xla_spawn (TPUs)
	main()


if __name__ == "__main__":
	main()
