# Unsupervised Entity Disambiguation using Pretrained Contextualized Word Embeddings

This repository contains the code for the equally named master thesis at [UHH's LT Group](https://www.inf.uni-hamburg.de/en/inst/ab/lt/home.html).

## Requirements
All code in this repository is written in Python (python 3.9.6 to be precise).
Python dependencies are managed with Pipenv, which needs to be installed.
Then, to install dependencies run:
```shell
pipenv install
```

To build the models and run the experiments, we need datasets. In the thesis, we use two models:
* AIDA YAGO
* Wikipedia Dump

### AIDA YAGO
For license issues, you need to provide the AIDA dataset yourself.
Place the full AIDA TSV in ``aida/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv``, the Makefile will split it into training and test files.

### Wikipedia
The dump, that we use, consists of one JSON object per line.
Each line is a JSON object with the keys ``id``, ``title`` and ``text`` corresponding to one Wikipedia page.

## Run
The Makefile holds all commands used to build the models and run the experiments.
Our pipeline consists of multiple Python files that all load and save pickles.
