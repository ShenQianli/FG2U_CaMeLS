# (FG)2U implementation for CaMeLS

This is part of the official code repo for Memory-Efficient Gradient Unrolling for Large-Scale Bi-level Optimization (NeurIPS24).

ArXiv: https://arxiv.org/abs/2406.14095  
Please feel free to shoot an email to shenqianli@u.nus.edu for any questions.

The repo is built on top of the original CaMeLS repo https://github.com/nathanhu0/CaMeLS.

### Usage

The usage is almost the same as the original CaMeLS repo.
Please follow the step-by-step instructions in the original README https://github.com/nathanhu0/CaMeLS/blob/main/README.md.

The only difference is that ```forward_mode``` option is set to be ```True``` in ```conf/task/train.yaml```.   
The core implementation can be found in ```weight_model.py``` starting from line 125.

### Citation

If you use this code or find our work helpful, please consider citing:

```
@article{shen2024memory,
  title={Memory-Efficient Gradient Unrolling for Large-Scale Bi-level Optimization},
  author={Shen, Qianli and Wang, Yezhen and Yang, Zhouhao and Li, Xiang and Wang, Haonan and Zhang, Yang and Scarlett, Jonathan and Zhu, Zhanxing and Kawaguchi, Kenji},
  journal={arXiv preprint arXiv:2406.14095},
  year={2024}
}
```

To cite CaMeLS:

```
@misc{hu2023metalearning,
      title={Meta-Learning Online Adaptation of Language Models}, 
      author={Nathan Hu and Eric Mitchell and Christopher D. Manning and Chelsea Finn},
      year={2023},
      eprint={2305.15076},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



-----------------------------------------------------------
(The original README of CaMeLS is below.)

# CaMeLS: Context-aware Meta-learned Loss Scaling

## What is this repo?

This repo includes a reference implementation of the CaMeLS meta-learning-based approach to online adaptation as in [Meta-Learning Online Adaptation of Language Models](https://arxiv.org/abs/2305.15076). CaMeLS meta-learns an importance weighting model that identifies which tokens in a stream of documents are most important to update on during fine-tuning.

The files in this repo are:

- `run.py` - main function to be called for both training of CaMeLS and evaluation of online adaptation.
- `weight_model.py` - the core implementation of a CaMeLS weight model and baseline loss weighting strategies.
- `train_qa.py` - script to fine tune language models for question-answering on a given dataset.
- `util.py` - various helper functions.
- `exp_datasets`: dataset classes surrounding document-question pairs. 
- `subroutines.py`: definition of subroutines for QA tuning and top_k evaluation.


## Datasets
We conduct experiments using three datasets:

**[StreamingQA:](https://github.com/deepmind/streamingqa)** The CSV files containing the subset of the StreamingQA dataset we use for our experiments can be downloaded [here](https://drive.google.com/drive/folders/1Xod97TmnjmbGDiyOHfuEUZ14tSL3qA-X?usp=drive_link).

**SQuAD:** We directly uses the HuggingFace dataset class and do not require additional files to be manually downloaded.

**ArchivalQA:** The question-article pairings can be recreated by running the `ArchivalQA_processing.ipynb` notebook using the [ArchivalQA questions](https://drive.google.com/drive/u/0/folders/15JMtkJAqtZsKr_P-0jH4iFy2EOri4GbR) and the [NYT corpus](https://catalog.ldc.upenn.edu/LDC2008T19).

**OpenWebText:** we additionally used a small sample of openwebtext to enforce local updates during CaMeLS meta-training (i.e., to make sure that fine-tuning using the meta-learned importance weights doesn't harm the model for unrelated texts). It can be downloaded [here](https://drive.google.com/drive/folders/1Xod97TmnjmbGDiyOHfuEUZ14tSL3qA-X?usp=drive_link).

## QA model pretraining

The language models trained for question-answering are used for CaMeLS meta-training and online-adaptation evaluation. Running `train_qa.py` will load a pretrained language model from Huggingface and fine tune that model for question-answering on a dataset. For example:
`python train_qa.py dataset=squad model=gpt2-xl`

Key `train_qa.py` arguments:
- `dataset`: the dataset to fine tune on. One of `archivalqa`, `squad`, `streamingqa`.  
- `model`: the pretrained language model to load from HuggingFace. Valid values include: `gpt2-xl`, `EleutherAI/gpt-neo-1.3b`
, or `EleutherAI/gpt-j-6b`.


## CaMeLS meta-training and evaluation

Depending on the value of the `task` argument, `run.py` is used for both training of CaMeLS and evaluation of CaMeLS and baseline online adaptation methods. In both settings, the following common arguments are used:

Key `run.py` arguments:
- `model`: Weight model configuration corresponding to the loss weighting strategy. During evaluation, can be set to `uniform` (standard fine tuning), `ssm` (fine tuning on salient spans), or `CaMeLS`. Should be set to `model=CaMeLS` for all meta-training runs.
- `dataset`: dataset used for the given task.
- `base_model`: used to specify the base language model. To be passed as the `pretrained_model_name_or_path` argument of `AutoModelForCausalLM.from_pretrained()`. During training, the base model updated during the inner step of optimization. During evaluation, the base model is updated on a stream of documents then evaluated on question answering.
- `base_model_state_dict`: If not `None`, we set the base model parameters by loading a state dictionary from the specified path.

### CaMeLS meta-training

Running `run.py` with `task=train` lets us train CaMeLS weighting models. You will need to specify the model, dataset, and base model. For example:

```python run.py task=train weight_model=CaMeLS dataset=archivalqa base_model=distilgpt2 base_model_state_dict=/qa_tuned/distilgpt2/state_dict.pt```

Key `run.py task=train` arguments:
- `update_batch_size`: (defaults to 6) the number of documents sampled per training batch
- `sequential_update`: (defaults to `True`) How to update the base model in the inner step of meta-training. If true, the base model is sequentially updated on each document in the batch. This results in a gradient tape of `update_batch_size` many base model updates. If false, the base model is updated for a single step on all documents.
- `grad_acc_steps`: (defaults to 4) The number of batches to accumulate outer loss gradients for before updating the weighting model.
- `reset_base_freq`: (defaults to 24) The number of documents the base model is updated on before its state is reset to the starting state. 

### Online Adaptation Evaluation

Running `run.py` with `task=eval` lets us evaluate `CaMeLS` and other baseline loss scaling approaches for online adaptation of a base model. You will need to specify the loss weighting strategy, dataset, and base model. For example:

To evaluate using a saved CaMeLS weight model:
`python run.py task=eval dataset=streamingqa weight_model=CaMeLS weight_model_path=path/to/CaMeLS_checkpoint.pt base_model=path/to/qa/model_ckpt lr=2.5e-05`

To perform a learning rate sweep of the Salient Span Masking baseline: 
`python run.py -m task=eval dataset=streamingqa weight_model=ssm base_model=path/to/qa/model_ckpt lr=.0001,.000025,.00000625,.0000015625 test_path=path/to/val_data.csv`

Key `run.py task=eval` arguments:
- `downsample_to`: for evaluation, we typically consider many streams of documents sampled from the test split. `downsample_to = k` corresponds to sampling `k` documents from the test set
- `seed`: the random seed used to generate the test stream
- `qa_lt_final`:  whether to additionally "light tune" the adapted qa model before evaluation (to essentially "re-learn" the QA task, which may be forgotten during adaptation). Used for the uniform + qa_tune baseline
- `lr`: the learning rate used to update the base model on the stream of documents. We note that for all online adaptation methods, performance was very sensitive to learning rate. 

## A worked example
In this example we will train weight model on distilgpt2 fine tuned for QA on SQuAD. Then we will use the resulting weight model for online adaptation of gpt2-xl. At each step, the outputs are generated in subdirectory beginning with `outputs/TASK/DATASET/`

### Step 1: Set up environment
First, create a virtualenv and install the dependencies. 

    python3 -m venv ./env
    source ./env/bin/activate
    pip install -r requirements.txt

Additionally, change the `CACHE_DIR` dir variable in `util.py` if you wish.

For experiments using StreamingQA, download corresponding CSV files locally. Then change the value of `data_dir` in `conf/dataset/streamingqa` and `qa_conf/dataset/streamingqa` accordingly.

For metatraining using OpenWebText to enforce local updates, change the value of `web_text_csv` in `conf/task/train` to be the absolute path to the local OpenWebText csv.

### Step 2: Run QA pretraining for the base model to be used during CaMeLS pretraining and to be used for evaluation.
Next we train 2 language models for question answering. 

    python train_qa.py dataset=streamingqa model=distilgpt2
    python train_qa.py dataset=streamingqa model=gpt2-xl

The models with the lowest validation NLL, highest validation F1 score, and highest exact match are saved in the 'checkpoints' subdirectory. (For small models which fail to achieve significant F1 or EM scores, the model with the lowest NLL is selected. Otherwise, our experiments chose the models with highest F1). 

### Step 2: Train a CaMeLS weight model

We now use the smaller qa-tuned distilgpt2 as the _base_model_ for meta-training of a CaMeLS weighting model.

    python run.py task=train weight_model=CaMeLS dataset=streamingqa base_model=distilgpt2 base_model_state_dict={ABSOLUTE_PATH_TO_DISTILGPT2_STATE_DICT}

Sample model importance weights are generated periodically during training. Model checkpoints are saved every epoch and at each validation step with lower loss than all previous steps.

### Step 3: Evaluate a trained weight model for Online Adapation

Lastly, we will use our trained weighting model to adapt the gpt2-xl model which we previously fined tuned for question answering. We set `dowmsample_to=1665` to adapt on a stream of 1665 sampled articles.

    python run.py task=eval weight_model=CaMeLS weight_model_path={PATH_TO_WEIGHT_MODEL_CHECKPOINT} dataset=streamingqa base_model=gpt2-xl base_model_state_dict={ABSOLUTE_PATH_TO_GPT2XL_STATE_DICT} downsample_to=1665 lr=2.5e-5

To perform this same evaluation using a uniform fine tuning baseline

    python run.py task=eval weight_model=uniform qa_lt_final=True dataset=streamingqa base_model=gpt2-xl base_model_state_dict={ABSOLUTE_PATH_TO_GPT2XL_STATE_DICT} downsample_to=1665 lr=2.5e-5

For the uniform + post adaptation QA tuning:

    python run.py task=eval weight_model=uniform dataset=streamingqa base_model=gpt2-xl base_model_state_dict={ABSOLUTE_PATH_TO_GPT2XL_STATE_DICT} downsample_to=1665 lr=2.5e-5

For only fine tuning on salient spans (first line only needs to be run the first time):

    python -m spacy download en_core_web_sm
    python run.py task=eval weight_model=ssm dataset=streamingqa base_model=gpt2-xl base_model_state_dict={ABSOLUTE_PATH_TO_GPT2XL_STATE_DICT} downsample_to=1665 lr=2.5e-5

For the TFIDF + thresholding baseline:

   python run.py task=eval weight_model=TFIDF min_threshold=.05 dataset=streamingqa base_model=gpt2-xl base_model_state_dict={ABSOLUTE_PATH_TO_GPT2XL_STATE_DICT} downsample_to=1665 lr=2.5e-5

The model generations, per question F1 and EM values, and average F1 and EM scores are generated in an output csv file.
## Citing CaMeLS
If CaMeLS or this repository is useful in your own research, you can use the following BibTeX entry:

    @misc{hu2023metalearning,
      title={Meta-Learning Online Adaptation of Language Models}, 
      author={Nathan Hu and Eric Mitchell and Christopher D. Manning and Chelsea Finn},
      year={2023},
      eprint={2305.15076},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
