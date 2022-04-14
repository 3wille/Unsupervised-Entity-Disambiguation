#!/usr/bin/env python
# coding: utf-8

import pickle
from chinese_whispers import aggregate_clusters
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'embeds_graph_pickle', nargs='?', default="pickles/mentions_word_embeds_graph.pickle"
)
parser.add_argument(
    'output_pickle', nargs='?', default="pickles/cluster_means.pickle"
)
args = parser.parse_args()

with open(args.embeds_graph_pickle, "rb") as f:
    mentions, embeddings, graph = pickle.load(f)

embeddings_by_mention = {}
for index, embedding in enumerate(embeddings):
    mention = mentions.iloc[index]
    embeddings_by_mention[mention.name] = embedding

clusters = aggregate_clusters(graph, label_key='cluster_label')
cluster_means = {}
for cluster_label, cluster_node_names in clusters.items():
    cluster_nodes = [graph.nodes[cluster_node_name] for cluster_node_name in cluster_node_names]
    cluster_mention_names = []
    for cluster_node in cluster_nodes:
        cluster_mention_names.extend(cluster_node["mention_names"])
    cluster_mention_names = list(set(cluster_mention_names))
    cluster_embeddings = [
        embeddings_by_mention[mention_name] for mention_name in cluster_mention_names
    ]

    embeddings_count = len(cluster_embeddings)
    embedding_size = len(cluster_embeddings[0])
    embeddings_tensor = torch.Tensor(embeddings_count, embedding_size)
    torch.stack(cluster_embeddings, out=embeddings_tensor)
    mean = torch.mean(embeddings_tensor, 0)
    cluster_means[cluster_label] = mean

with open(args.output_pickle, "wb") as f:
    pickle.dump(cluster_means, f, pickle.HIGHEST_PROTOCOL)
