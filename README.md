# Pictalk_NLP

Implementation of the PrAACT method (described in [PrAACT: Predictive Augmentative and Alternative Communication with Transformers](https://www.sciencedirect.com/science/article/abs/pii/S0957417423029196)) in order to help the development of [Pictalk](https://github.com/Pictalk-speech-made-easy).

## Dependencies and installation

This project uses [Python 3.10.12](https://www.python.org/downloads/release/python-31012/). Use the package manager [pip](https://pypi.org/project/pip/) to install the dependencies :

```bash
pip install -r requirements.txt
```

## Usage

The code is divided in 3 main files, each implementing a step of the method described in the paper :

1. `corpus_annotation.py` : Annotates the aactext corpus as described in the paper and push it on [HuggingFace](https://huggingface.co/datasets/LucasMagnana/aactext_text). The annotation consists in lemmatizing the sentences and adding the lemmatized and the original sentence in the dataset. Only sentences that do not contain commas are processed, and the punctuation is removed from both the original and the lemmatized sentences.

2. `fine_tuning.py` : Finetunes a transformer model using the annotated corpus. This file is an adaptation of this HuggingFace [tutorial](https://huggingface.co/learn/nlp-course/chapter7/3). The main difference is that the preprocessing of the dataset is pushed on HuggingFace, as the process takes too much time on [Google Colab](https://colab.research.google.com/drive/1u99u3JnAhPDq17F2ojFNzqr_ghTs5lxS?authuser=2&hl=fr) due to low CPU power. The preprocessing is triggered by using the `--preprocess` argument.
> [!WARNING]  
> The preprocessing needs to be redone every time the corpus is changed and every time a new type of model is finetuned (as it uses the tokenizer of the model).
The default model used is the [Large Bert](https://huggingface.co/bert-large-uncased) but it can be changed using the `--model`/`-m` argument.

3. `vocabulary_encoding.py` : Computes the decoder layer of the final model as described in the paper. An embedding matrix is created using a dataset of pictograms and the embeded layer of a transformer. A linear layer using the matrix as its weights is then created and pushed to HuggingFace. Note that the model remains unchanged during the process. The vocabulary (i.e. the dataset of pictograms) used is [CACE-UTAC](https://www.utac.cat/descarregues/cace-utac). It has been translated in english and pushed on HuggingFace using `datasets/upload_ARASAAC_CACE.py`.

`complete_sentences.py` has been added in order to show how to use the final models. The decoder layer used is the one computed and pushed with `vocabulary_encoding.py`, but the `--no-encode` argument can be utilized to use the decoder layer of the model instead.

