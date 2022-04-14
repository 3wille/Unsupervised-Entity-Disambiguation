#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModel
from sentence_transformers import util
from utils import retok
import os
import pickle
from utils.general import batch
from load_wikipedia_extracts import request_wikipedia_pages
import tqdm


class Similarity():
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.t = retok.ReTokenizer(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )
        self.model.eval()
        self.top_k = 3

    def build_mention_embedding(self, sentence_tokens, mention_start_index, mention_end_index):
        tokens, ind, _l = self.t.retokenize_and_encode_indexed(sentence_tokens)
        if self.model_name == "bert-base-uncased":
            token_type_ids = tokens.token_type_ids
        else:
            token_type_ids = None
        with torch.no_grad():
            model_output = self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                token_type_ids=token_type_ids
            )
        last_layer_token_embeddings = model_output[0][0]

        mention_indexes = []
        for j in range(mention_start_index, mention_end_index + 1):
            mention_indexes.extend(ind[0][j][1])
        mention_embeddings = last_layer_token_embeddings[mention_indexes]
        mention_embedding = torch.mean(mention_embeddings, 0)
        mention_embedding_tensor = torch.stack([mention_embedding])
        return mention_embedding_tensor

    def calculate_similarities(
        self, sentence_tokens, mention_start_index, mention_end_index, target_tensor
    ):
        mention_embedding_tensor = self.build_mention_embedding(
            sentence_tokens, mention_start_index, mention_end_index
        )
        similarities = util.pytorch_cos_sim(mention_embedding_tensor, target_tensor)
        return similarities


def retrieve_wikipedia_pages(mentions_with_wikipedia):
    pickle_filename = "pickles/dev/wikipedia_page_extracts.pickle"
    file_present = os.path.isfile(pickle_filename)
    if file_present:
        with open(pickle_filename, "rb") as f:
            return pickle.load(f)
    page_descriptions = {}
    wikipedia_ids = []
    for _index, mention in mentions_with_wikipedia.iterrows():
        wikipedia_ids.append(str(mention["wikipedia_id"])[:-2])
    batched_ids = batch(wikipedia_ids, n=20)
    for wikipedia_ids in tqdm(batched_ids):
        response = request_wikipedia_pages(wikipedia_ids)
        if 'query' not in response.json().keys():
            print(f"no query, page_ids: {wikipedia_ids}")
            continue
        for page_id, page in response.json()['query']['pages'].items():
            if 'extract' not in page.keys():
                print(f"no extract, page id: {page_id}")
                print(page.keys())
                # print(page['description'])
                continue
            extract = page["extract"]
            page_descriptions[page_id] = extract
    with open(pickle_filename, "wb") as f:
        pickle.dump(page_descriptions, f, pickle.HIGHEST_PROTOCOL)
    return page_descriptions
