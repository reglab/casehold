import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

import transformers
from transformers import (
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	EvalPrediction,
	HfArgumentParser,
	PretrainedConfig,
	Trainer,
	TrainingArguments,
	default_data_collator,
	set_seed,
)
from transformers.trainer_utils import is_main_process


task_to_keys = {
	"cola": ("sentence", None),
	"mnli": ("premise", "hypothesis"),
	"mrpc": ("sentence1", "sentence2"),
	"qnli": ("question", "sentence"),
	"qqp": ("question1", "question2"),
	"rte": ("sentence1", "sentence2"),
	"sst2": ("sentence", None),
	"stsb": ("sentence1", "sentence2"),
	"wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""

	task_name: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
	)
	max_seq_length: int = field(
		default=128,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
	)
	pad_to_max_length: bool = field(
		default=True,
		metadata={
			"help": "Whether to pad all samples to `max_seq_length`. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch."
		},
	)
	train_file: Optional[str] = field(
		default=None, metadata={"help": "A csv or a json file containing the training data."}
	)
	validation_file: Optional[str] = field(
		default=None, metadata={"help": "A csv or a json file containing the validation data."}
	)

	def __post_init__(self):
		if self.task_name is not None:
			self.task_name = self.task_name.lower()
			if self.task_name not in task_to_keys.keys():
				raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
		elif self.train_file is None or self.validation_file is None:
			raise ValueError("Need either a GLUE task or a training/validation file.")
		else:
			extension = self.train_file.split(".")[-1]
			assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
			extension = self.validation_file.split(".")[-1]
			assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


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
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
	)


def main():
	# See all possible arguments by passing --help to this script.

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
			f"Output directory ({training_args.output_dir}) already exists and is not empty. "
			"Use --overwrite_output_dir to overcome."
		)

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
	)

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
		+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	# Set the verbosity to info of the Transformers logger (on main process only):
	if is_main_process(training_args.local_rank):
		transformers.utils.logging.set_verbosity_info()
		transformers.utils.logging.enable_default_handler()
		transformers.utils.logging.enable_explicit_format()
	logger.info(f"Training/evaluation parameters {training_args}")

	# Set seed before initializing model.
	set_seed(training_args.seed)

	# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
	# or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
	#
	# For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
	# sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
	# label if at least two columns are provided.
	#
	# If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
	# single column. See preprocessing section below for more details.
	#
	# In distributed training, the load_dataset function guarantee that only one local process can concurrently
	# download the dataset.
	if data_args.task_name is not None:
		# Downloading and loading a dataset from the hub.
		datasets = load_dataset("glue", data_args.task_name)
	elif data_args.train_file.endswith(".csv"):
		# Loading a dataset from local csv files
		datasets = load_dataset(
			"csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
		)
	else:
		# Loading a dataset from local json files
		datasets = load_dataset(
			"json", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
		)
	# See more about loading any type of standard or custom dataset at
	# https://huggingface.co/docs/datasets/loading_datasets.html.

	# Labels
	# Labels for GLUE tasks
	if data_args.task_name is not None:
		is_regression = data_args.task_name == "stsb"
		# Regression tasks
		if is_regression:
			num_labels = 1
		# Classification tasks
		else:
			label_list = datasets["train"].features["label"].names
			num_labels = len(label_list)
	# Labels for custom tasks
	else:
		is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
		# Regression tasks
		if is_regression:
			num_labels = 1
		# Classification tasks
		else:
			label_list = datasets["train"].unique("label")
			label_list.sort()  # Sort for deterministic ordering
			num_labels = len(label_list)

	# Load pretrained model and tokenizer
	#
	# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
	# download model & vocab.
	config = AutoConfig.from_pretrained(
		model_args.config_name if model_args.config_name else model_args.model_name_or_path,
		num_labels=num_labels,
		finetuning_task=data_args.task_name,
		cache_dir=model_args.cache_dir,
	)
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		# Defaults to using fast tokenizer
		use_fast=model_args.use_fast_tokenizer,
	)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
	)

	# Preprocess dataset
	non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
	if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
		sentence1_key, sentence2_key = "sentence1", "sentence2"
	else:
		if len(non_label_column_names) >= 2:
			sentence1_key, sentence2_key = non_label_column_names[:2]
		else:
			sentence1_key, sentence2_key = non_label_column_names[0], None

	# Padding strategy
	if data_args.pad_to_max_length:
		padding = "max_length"
		max_length = data_args.max_seq_length
	else:
		# Pad dynamically at batch creation, to the max sequence length in each batch
		padding = False
		max_length = None

	# Some models have set the order of the labels to use, so set the specified order here
	label_to_id = None
	if (
		model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
		and data_args.task_name is not None
		and is_regression
	):
		# Some have all caps in their config, some don't.
		label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
		if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
			label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
		else:
			logger.warn(
				"Your model seems to have been trained with labels, but they don't match the dataset: ",
				f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
				"\nIgnoring the model labels as a result.",
			)
	elif data_args.task_name is None:
		label_to_id = {v: i for i, v in enumerate(label_list)}

	def preprocess_function(examples):
		# Tokenize the texts
		args = (
			(examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
		)
		result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

		# Map labels to IDs (not necessary for GLUE tasks)
		if label_to_id is not None and "label" in examples:
			result["label"] = [label_to_id[l] for l in examples["label"]]
		return result

	datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

	train_dataset = datasets["train"]
	eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
	# Get the corresponding test set for GLUE task
	if data_args.task_name is not None:
		test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

	# Log a few random samples from the training set:
	for index in random.sample(range(len(train_dataset)), 3):
		logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

	# Get the corresponding metric function for GLUE task
	if data_args.task_name is not None:
		metric = load_metric("glue", data_args.task_name)

	# Define custom compute_metrics function, returns F1 metric for Overruling and ToS binary classification tasks
	def compute_metrics(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
		metric = load_metric("f1")
		# Compute F1 for binary classification task
		f1 = metric.compute(predictions=preds, references=p.label_ids)
		return f1

	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset if training_args.do_eval else None,
		compute_metrics=compute_metrics,
		tokenizer=tokenizer,
		# Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
		data_collator=default_data_collator if data_args.pad_to_max_length else None,
	)

	if not custom_args.ptl:
		eval_results = {}

		# Training on train_dataset
		if training_args.do_train:
			trainer.train(
				model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
			)
			trainer.save_model()  # Saves the tokenizer too for easy model upload

		# Evaluation on eval_dataset
		if training_args.do_eval:
			logger.info("*** Evaluate ***")

			eval_result = trainer.evaluate(eval_dataset=eval_dataset)

			output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
			if trainer.is_world_process_zero():
				with open(output_eval_file, "w") as writer:
					logger.info(f"***** Eval results *****")
					for key, value in eval_result.items():
						logger.info(f"  {key} = {value}")
						writer.write(f"{key} = {value}\n")

			eval_results.update(eval_result)
		
		# Prediction on eval_dataset
		if training_args.do_predict:
			logger.info("*** Predict ***")

			predictions = trainer.predict(test_dataset=eval_dataset).predictions
			predictions = np.argmax(predictions, axis=1)

			output_preds_file = os.path.join(training_args.output_dir, "predictions.csv")
			if trainer.is_world_process_zero():
				np.savetxt(output_preds_file, predictions, delimiter=',', fmt='%.4e')

		return eval_results
	# If ptl=True is passed, compute per example/average pre-train loss on eval_dataset
	# To compute pre-train loss across full dataset, pass in full dataset file to command line argument, validation_file
	else:
		logger.info("*** Compute pre-train loss ***")

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