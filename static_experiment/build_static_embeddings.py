#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import gensim.downloader as gs_downloader
from gensim.models import FastText
from tqdm import tqdm
import torch
tqdm.pandas()

kv = gs_downloader.load('fasttext-wiki-news-subwords-300', return_path=True)
model = FastText(kv)


def static_embedding(tokens: list[str]):
    embeddings = []
    for token in tokens:
        embeddings.append(model.wv[token])
    embedding = np.average(embeddings, axis=0)
    embedding_tensor = torch.from_numpy(embedding)
    return embedding_tensor


def static_embedding_for_mention(mention):
    full_mention = mention["full_mention"]
    return static_embedding(full_mention.split(" "))


if __name__ == '__main__':
    with open("pickles/filtered_enh2_unambiguous_mention.pickle", "rb") as f:
        unambiguous_mentions = pickle.load(f)

    embeddings = unambiguous_mentions.progress_apply(static_embedding_for_mention, axis=1).to_list()

    with open("static_experiment/pickles/embeddings.pickle", "wb") as f:
        pickle.dump([unambiguous_mentions, embeddings], f, pickle.HIGHEST_PROTOCOL)
