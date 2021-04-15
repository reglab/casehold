## When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset of 53,000+ Legal Holdings

This is the repository for the paper, "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset of 53,000+ Legal Holdings" [update with link]. 

It includes datasets, models, and code for computing pre-train loss and finetuning Legal-BERT and Custom Legal-BERT models on legal benchmark tasks: Overruling, Terms of Service, CaseHOLD.

### Datasets and Models
The legal benchmark task datasets and Legal-BERT model files can be downloaded from the casehold Google Drive [folder](https://drive.google.com/drive/folders/18YZpKNzbgG3ZWWgmu0Xz6oK3nuv0M2iK?usp=sharing). For more information, see the [Description](https://docs.google.com/document/d/1K3LtZ5Z6Zxh9Xuf5Pu0P4UuPXa_rCuE6b2_gL1yLej8/edit?usp=sharing) in the folder.

Alternatively, the Legal-BERT models can also be accessed directly from the Hugging Face model hub. To load a model from the model hub in a script, pass its Hugging Face model repo name to the model_name_or_path argument. See `demo.ipynb` for more details.

To compute domain specificity scores (DS) for the tasks, you will need to compute pre-train loss on the Legal-BERT model and the BERT (double) model, which is pretrained for additional steps on the general domain BERT vocabulary for comparability to Legal-BERT models. If you are interested in accessing the BERT (double) model files, please contact: [update with email].

### Requirements
This code was tested with Python 3.7 and Pytorch 1.8.1.

Install required packages and dependencies:

    pip install -r requirements.txt

Install transformers from source (required for tokenizers dependencies):

    pip install git+https://github.com/huggingface/transformers

### Demo
`demo.ipynb` provides examples of how to run the scripts to compute pre-train loss/finetune Legal-BERT models on the legal benchmark tasks. These examples should be able to run on a GPU that has 16GB of RAM using the hyperparameters given.

### Citation
If you are using this work, please cite it as:

    [update with citation]