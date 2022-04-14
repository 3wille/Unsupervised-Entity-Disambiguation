#!/usr/bin/env python
# coding: utf-8

from sentence_transformers import util
import torch
from chinese_whispers import chinese_whispers, aggregate_clusters
import networkx as nx
import pickle
from tqdm import tqdm
import argparse
from collections import Counter
from utils.general import batch_with_indices
from IPython import embed

tqdm.pandas()
parser = argparse.ArgumentParser()
parser.add_argument(
    'embeddings_pickle', nargs='?', default="pickles/filtered_word_embeddings.pickle"
)
parser.add_argument(
    'output_pickle', nargs='?', default="pickles/mentions_word_embeds_graph.pickle"
)
parser.add_argument(
    'node_accessor', nargs='?', default="full_mention"
)
args = parser.parse_args()

with open(args.embeddings_pickle, "rb") as f:
    all_mentions, all_embeddings = pickle.load(f)


def get_node_name(mention) -> str:
    if args.node_accessor == "index":
        return mention.name
    else:
        return mention[args.node_accessor]


used_mentions = 30000
mentions = all_mentions.iloc[:used_mentions]
embeddings = all_embeddings[:used_mentions]

embeddings_count = len(embeddings)
embedding_size = len(embeddings[0])
embeddings_tensor = torch.Tensor(embeddings_count, embedding_size)
torch.stack(embeddings, out=embeddings_tensor)

graph = nx.Graph()


def add_node_from_mention(mention):
    full_mention = get_node_name(mention)
    # print(full_mention)
    if full_mention not in graph.nodes:
        graph.add_node(full_mention, mention_names=[mention.name])
    else:
        graph.nodes[full_mention]["mention_names"].append(mention.name)
    # print(graph.nodes[full_mention])


print("Building graph nodes")
mentions.progress_apply(add_node_from_mention, axis=1)


def add_or_update_edge(graph: nx.Graph, current_mention, target_mention, weight):
    edge_key = (get_node_name(current_mention), get_node_name(target_mention))
    if edge_key not in graph.edges or graph.edges[edge_key]['weight'] < weight:
        graph.add_edge(
            get_node_name(current_mention),
            get_node_name(target_mention),
            weight=weight,
            current_mention_name=int(current_mention.name),
            target_mention_name=int(target_mention.name),
        )

print("Building graph edges")
batches = tqdm(batch_with_indices(embeddings_tensor, n=5000), desc="Batches", total=len(embeddings_tensor)/100)
for embeddings_batch, lower_border, upper_border in batches:
    similarities_list = util.pytorch_cos_sim(embeddings_batch, embeddings_tensor)
    for index, similarities in zip(tqdm(range(lower_border, upper_border), desc="In-Batch"), similarities_list):
        similarities[index] = 0
        similarities, indices = torch.sort(similarities, descending=True)
        current_mention = mentions.iloc[index]
        counter = 0
        for similarity, i in zip(similarities, indices):
            if counter == 3:
                break
            target_mention = mentions.iloc[i.item()]
            if current_mention["full_mention"] == target_mention["full_mention"]:
                continue

            add_or_update_edge(
                graph,
                current_mention,
                target_mention,
                similarity.item(),
            )
            counter += 1

print("Clustering")
chinese_whispers(graph, label_key='cluster_label', iterations=3, weighting='lin')
print("Clustering finished")

cluster_sizes = []
for label, cluster in sorted(aggregate_clusters(graph, label_key='cluster_label').items(), key=lambda e: len(e[1]), reverse=True):
    if len(cluster) > 1:
        cluster_mentions = {}
        for node_name in cluster:
            node = graph.nodes[node_name]
            cluster_mentions[node_name] = list(map(
                lambda mention_name: mentions.loc[mention_name]["sentence_id"],
                node["mention_names"],
            ))
        print('{}\t{}\n'.format(label, cluster_mentions))
    cluster_sizes.append(len(cluster))
print(len(cluster_sizes))
counter = Counter(cluster_sizes)
print(counter)

# import matplotlib.pyplot as plt
# colors = [1. / graph.nodes[node]['label'] for node in graph.nodes()]
# nx.draw_networkx(graph, cmap=plt.get_cmap('jet'), node_color=colors, font_color='grey')
# plt.show()

with open(args.output_pickle, "wb") as f:
    pickle.dump([mentions, embeddings, graph], f, pickle.HIGHEST_PROTOCOL)
