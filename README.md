## When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset of 53,000+ Legal Holdings

This is the repository for the paper, [When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset of 53,000+ Legal Holdings](https://arxiv.org/abs/2104.08671), accepted to ICAIL 2021.

It includes models, datasets, and code for computing pretrain loss and finetuning Legal-BERT and Custom Legal-BERT models on legal benchmark tasks: Overruling, Terms of Service, CaseHOLD.

### Download Models & Datasets
The legal benchmark task datasets and Legal-BERT/Custom Legal-BERT model files can be downloaded from the [casehold Google Drive folder](https://drive.google.com/drive/folders/18YZpKNzbgG3ZWWgmu0Xz6oK3nuv0M2iK?usp=sharing). For more information, see the [Description](https://docs.google.com/document/d/1K3LtZ5Z6Zxh9Xuf5Pu0P4UuPXa_rCuE6b2_gL1yLej8/edit?usp=sharing) of the folder.

Alternatively, the Legal-BERT/Custom Legal-BERT models can also be accessed directly from the Hugging Face model hub. To load a model from the model hub in a script, pass its Hugging Face model repository name to the `model_name_or_path` script argument. See `demo.ipynb` for more details.

**Hugging Face Model Repositories**

-   Legal-BERT: `zlucia/legalbert` (https://huggingface.co/zlucia/legalbert)
- Custom Legal-BERT: `zlucia/custom-legalbert` (https://huggingface.co/zlucia/custom-legalbert)

Download the legal benchmark task datasets and the Legal-BERT/Custom Legal-BERT models (optional, scripts can directly load models from Hugging Face model repositories) and unzip them under the top-level directory like:

	reglab/casehold
	├── data
	│ ├── casehold.csv
	│ └── overruling.csv
	├── models
	│ ├── custom-legalbert
	│ │ ├── config.json
	│ │ ├── pytorch_model.bin
	│ │ ├── special_tokens_map.json
	│ │ ├── tf_model.h5
	│ │ ├── tokenizer_config.json
	│ │ └── vocab.txt
	│ └── legalbert
	│ │ ├── config.json
	│ │ ├── pytorch_model.bin
	│ │ ├── special_tokens_map.json
	│ │ ├── tf_model.h5
	│ │ ├── tokenizer_config.json
	│ │ └── vocab.txt

To compute domain specificity scores (DS) for the tasks, take the average difference in pretrain loss between the BERT (double) model and the Legal-BERT model. The BERT (double) model is initialized with the base BERT model (uncased, 110M parameters), [bert-base-uncased](https://huggingface.co/bert-base-uncased), and pretrained for additional steps on the general domain BERT vocabulary for comparability to Legal-BERT/Custom Legal-BERT models. If you are interested in accessing the BERT (double) model files, please contact: [update with email].

### Requirements
This code was tested with Python 3.7 and Pytorch 1.8.1.

Install required packages and dependencies:

    pip install -r requirements.txt

Install transformers from source (required for tokenizers dependencies):

    pip install git+https://github.com/huggingface/transformers

### Model Descriptions
####  Training Data
The pretraining corpus was constructed by ingesting the entire Harvard Law case corpus from 1965 to the present (https://case.law/). The size of this corpus (37GB) is substantial, representing 3,446,187 legal decisions across all federal and state courts, and is larger than the size of the BookCorpus/Wikipedia corpus originally used to train BERT (15GB). We randomly sample 10% of decisions from this corpus as a holdout set, which we use to create the CaseHOLD dataset. The remaining 90% is used for pretraining. 

#### Legal-BERT Training Objective
This model is initialized with the base BERT model (uncased, 110M parameters), [bert-base-uncased](https://huggingface.co/bert-base-uncased), and trained for an additional 1M steps on the MLM and NSP objective, with tokenization and sentence segmentation adapted for legal text (cf. the paper).

#### Custom Legal-BERT Training Objective
This model is pretrained from scratch for 2M steps on the MLM and NSP objective, with tokenization and sentence segmentation adapted for legal text (cf. the paper). 

The model also uses a custom domain-specific legal vocabulary. The vocabulary set is constructed using [SentencePiece](https://arxiv.org/abs/1808.06226) on a subsample (approx. 13M) of sentences from our pretraining corpus, with the number of tokens fixed to 32,000.

### CaseHOLD Dataset Description
The CaseHOLD dataset (Case Holdings On Legal Decisions) provides 53,000+ multiple choice questions with prompts from a judicial decision and five potential holdings, one of which is correct, that could be cited.

We construct this dataset from a holdout set of the Harvard Case corpus. We extract the holding statement from citations (parenthetical text that begins with "holding'') as the correct answer and take the text before it as the citing text prompt. We insert a `<HOLDING>` token in the position of the citing text prompt where the holding statement was extracted. To select four incorrect answers for a citing text, we compute the TD-IDF similarity between the correct answer and the pool of other holding statements extracted from the corpus and select the most similar holding statements, to make the task more difficult. We set an upper threshold for similarity to rule out indistinguishable holding statements (here 0.75), which would make the task impossible.

### Results
The results from the paper for the baseline BiLSTM, base BERT model (uncased, 110M parameters), BERT (double), Legal-BERT, and Custom Legal-BERT, finetuned on the legal benchmark tasks, are displayed below.

![](figures/results.png)

### Demo
`demo.ipynb` provides examples of how to run the scripts to compute pretrain loss and finetune Legal-BERT/Custom Legal-BERT models on the legal benchmark tasks. These examples should be able to run on a GPU that has 16GB of RAM using the hyperparameters specified in the examples.

### Citation
If you are using this work, please cite it as:

	@inproceedings{zhengguha2021,
		title={When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset},
		author={Lucia Zheng and Neel Guha and Brandon R. Anderson and Peter Henderson and Daniel E. Ho},
		year={2021},
		eprint={2104.08671},
		archivePrefix={arXiv},
		primaryClass={cs.CL},
		booktitle={Proceedings of the 18th International Conference on Artificial Intelligence and Law},
		publisher={Association for Computing Machinery},
		note={(in press)}
	}

Lucia Zheng, Neel Guha, Brandon R. Anderson, Peter Henderson, and Daniel E. Ho. 2021. When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset. In *Proceedings of the 18th International Conference on Artificial Intelligence and Law (ICAIL '21)*, June 21-25, 2021,  São Paulo, Brazil. ACM Inc., New York, NY, (in press). arXiv: [2104.08671 \[cs.CL\]](https://arxiv.org/abs/2104.08671).


