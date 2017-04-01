# OVERVIEW

This code replicates a convolutional neural network architecture for learning to match question and answer sentences described in the paper:

Aliaksei Severyn and Alessandro Moschitti. *Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks*. SIGIR, 2015



# DEPENDENCIES

- python 3.6
- [pytorch](http://pytorch.org/)
- numpy
- sklearn
- nltk
- scipy


# EMBEDDINGS

The pre-initialized word2vec embeddings have to be downloaded from [here](https://drive.google.com/folderview?id=0B-yipfgecoSBfkZlY2FFWEpDR3M4Qkw5U055MWJrenE5MTBFVXlpRnd0QjZaMDQxejh1cWs&usp=sharing).


# TREC-QA
The addressed task is a popular answer sentence selection benchmark, where the goal is for each question to select relevant answer sentences. The dataset was first introduced by (Wang et al., 2007) and further elaborated by (Yao et al., 2013). It is freely availabe.

We provided the processed xml file in jacana-qa-naacl2013-data-results/ folder.

# BUILD Dataset

```
python3 parse.py

python3 overlap_features.py TrecQA/

```