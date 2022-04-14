#!/usr/bin/env python
# coding: utf-8

from IPython import embed
import pickle
import torch
import logging
from sentence_transformers import util
from static_experiment.build_static_embeddings import static_embedding

with open("static_experiment/pickles/cluster_means.pickle", "rb") as f:
    cluster_means = pickle.load(f)
with open("static_experiment/pickles/llrs.pickle", "rb") as f:
    cluster_llrs = pickle.load(f)

cluster_ids, cluster_means_list = zip(*cluster_means.items())
cluster_means_tensor = torch.Tensor(
    len(cluster_means_list), len(cluster_means_list[0])
)
torch.stack(cluster_means_list, out=cluster_means_tensor)


def place_unknown(sentence_tokens, mention_start_index, mention_end_index, top_k):
    mention_tokens = sentence_tokens[mention_start_index:mention_end_index+1]
    mention_embedding_tensor = static_embedding(mention_tokens)
    similarities = util.pytorch_cos_sim(mention_embedding_tensor, cluster_means_tensor)

    top_similarities, indices = torch.topk(similarities, k=top_k, sorted=False)
    surface_form = sentence_tokens[mention_start_index:mention_end_index]
    result = {"sentence": " ".join(sentence_tokens), "surface_form": surface_form}
    for top_similarity_index, (_similarity, cluster_index) in enumerate(
        zip(top_similarities[0], indices[0])
    ):
        cluster_id = cluster_ids[cluster_index]
        logging.debug(f"cluster: {cluster_id}")
        logging.debug(cluster_llrs[cluster_id][:6])
        result[f"{top_similarity_index} top3 llr"] = cluster_llrs[cluster_id][:6]
        result[f"{top_similarity_index} cluster_id"] = cluster_id
    return result
