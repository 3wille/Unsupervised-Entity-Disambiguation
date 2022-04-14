#!/usr/bin/env python
# coding: utf-8

import pickle
from sklearn import cluster
import torch
import logging
from utils.similarity import Similarity
from IPython import embed


class PlaceUnknown():
    def __init__(
        self,
        model_name,
        cluster_means=None,
        cluster_llrs=None,
    ) -> None:
        self.similarity = Similarity(model_name)
        self.cluster_means = cluster_means
        self.cluster_llrs = cluster_llrs
        if cluster_means is None and cluster_llrs is None:
            self.load_defaults()

        self.cluster_ids, cluster_means_list = zip(*self.cluster_means.items())
        self.cluster_means_tensor = torch.Tensor(
            len(cluster_means_list), len(cluster_means_list[0])
        )
        torch.stack(cluster_means_list, out=self.cluster_means_tensor)


    def load_defaults(
        self,
        cluster_means_pickle="pickles/cluster_means.pickle",
        llr_pickle="pickles/cluster_llrs.pickle",
        clustering_pickle="pickles/mentions_word_embeds_graph.pickle"
    ):
        with open(cluster_means_pickle, "rb") as f:
            self.cluster_means = pickle.load(f)
        with open(llr_pickle, "rb") as f:
            self.cluster_llrs = pickle.load(f)


    def place_unknown(self, sentence_tokens, mention_start_index, mention_end_index, top_k):
        similarities = self.similarity.calculate_similarities(
            sentence_tokens, mention_start_index, mention_end_index, self.cluster_means_tensor
        )

        top_similarities, indices = torch.topk(similarities, k=top_k, sorted=False)
        surface_form = sentence_tokens[mention_start_index:mention_end_index+1]
        result = {"sentence": " ".join(sentence_tokens), "surface_form": surface_form}
        for top_similarity_index, (_similarity, cluster_index) in enumerate(
            zip(top_similarities[0], indices[0])
        ):
            # print(similarity)
            # print(index)
            cluster_id = self.cluster_ids[cluster_index]
            logging.debug(f"cluster: {cluster_id}")
            logging.debug(self.cluster_llrs[cluster_id][:6])
            result[f"{top_similarity_index} top3 llr"] = self.cluster_llrs[cluster_id][:6]
            result[f"{top_similarity_index} cluster_id"] = cluster_id
        return result
