#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizerFast, BertModel
from more_itertools import locate
from csv import QUOTE_NONE
from tqdm import tqdm
tqdm.pandas()
from utils.aida.train import aida
import sys

import pickle
# with open("pickles/unambiguous_labels.pickle", "rb") as f:
with open("pickles/unambiguous_mention.pickle", "rb") as f:
    _aida, unambiguous_mentions = pickle.load(f)
    # print(unambiguous_mentions)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(
    'bert-base-uncased',
    output_hidden_states = True
)
model.eval()

# unambiguous_mentions = unambiguous_mentions.iloc[:100]
print(unambiguous_mentions)

def build_sentence_embedding(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def build_sentence(mention):
    index = mention.name
    sentence_id = mention.sentence_id
    sentence = aida[aida["sentence_id"] == sentence_id]

    upper_sentence_bound = aida['token'][index:].isna().idxmax()
    lower_sentence_bound = aida['token'][:index].isna().sort_index(axis=0, ascending=False).idxmax()
    token_list = aida[lower_sentence_bound+1:upper_sentence_bound]['token'].to_list()
    mention_lower_bound = index - lower_sentence_bound - 1
    mention_upper_bound = mention_lower_bound + len(mention["full_mention"].split(" "))
    mention_range = (mention_lower_bound, mention_upper_bound)
    sentences.append(token_list)
    return pd.Series([token_list, mention_range])

sentences = []
sentences_and_mention_ranges = unambiguous_mentions.progress_apply(build_sentence, axis=1)
unambiguous_mentions[["sentence", "mention_range"]] = sentences_and_mention_ranges

tokens = tokenizer(
    sentences, is_split_into_words=True, return_tensors="pt", padding=True
)
with torch.no_grad():
    model_output = model(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        token_type_ids=tokens.token_type_ids
    )
sentence_emb = build_sentence_embedding(model_output, tokens.attention_mask)

import pickle
with open("pickles/sentence_embeddings.pickle", "wb") as f:
    pickle.dump([unambiguous_mentions, sentence_emb], f, pickle.HIGHEST_PROTOCOL)
