#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

from tqdm import tqdm
from chinese_whispers import aggregate_clusters
import nltk
from nltk.util import everygrams
from collections import Counter
import math
from pyvis.network import Network
import argparse
from matplotlib import cm
color_map = cm.get_cmap("jet")

nltk.download('omw-1.4')

pad_left = False
pad_right = False
ngram_n = 5
token_deny_list = ['(', ')', ',', '.']
nltk.download('wordnet')
nltk.download('punkt')
lemmatizer = nltk.stem.WordNetLemmatizer()


def build_cluster_ngrams(cluster_nodes, mentions, graph):
    cluster_ngrams = []
    for node_name in cluster_nodes:
        node = graph.nodes[node_name]
        node_mention_name = node["mention_names"]
        node_mentions = mentions.loc[node_mention_name]
        raw_descriptions = node_mentions[node_mentions["wikidata_description"].notna()]["wikidata_description"]
        if len(raw_descriptions) < 1:
            continue
        raw_description = raw_descriptions.iloc[0]
        if raw_description is None:
            continue
        node_ngrams = build_description_ngrams(raw_description)
        cluster_ngrams.extend(node_ngrams)
    return cluster_ngrams


def build_description_ngrams(raw_description):
    tokenized_description = nltk.tokenize.word_tokenize(raw_description)
    filtered_tokens = filter(lambda token: token not in token_deny_list, tokenized_description)
    filtered_lemmas = map(lemmatizer.lemmatize, filtered_tokens)
    node_ngrams = list(everygrams(list(filtered_lemmas), max_len=ngram_n, pad_left=pad_left, pad_right=pad_right))
    return node_ngrams


def term(n):
    return n*math.log(n+(n==0))
def llr(cluster, ngram, events):
    a_b = len(list(filter(lambda event: event[0] == cluster and event[1] == ngram, events)))
    a_nb = len(list(filter(lambda event: event[0] != cluster and event[1] == ngram, events)))
    na_b = len(list(filter(lambda event: event[0] == cluster and event[1] != ngram, events)))
    na_nb = len(list(filter(lambda event: event[0] != cluster and event[1] != ngram, events)))
    a = a_b + a_nb
    na = na_b + na_nb
    b = a_b + na_b
    nb = a_nb + na_nb
    s = a + na
    if s != b + nb:
        raise
    np.array([[a_b, na_b, b], [a_nb, na_nb, nb], [a, na, s]])
    llr_score = 2 * (term(s)-term(a)-term(b)+term(a_b)+term(na_nb)+term(na_b)+term(a_nb)-term(na)-term(nb))
    return llr_score


def calculate_llrs(mentions, graph):
    if graph is not None:
        clusters = aggregate_clusters(graph, label_key='cluster_label')
        clusters_with_ngrams = {cluster_label: build_cluster_ngrams(cluster_nodes, mentions, graph) for cluster_label, cluster_nodes in clusters.items()}
    elif 'hierarchical_clustering' in mentions.columns:
        clusters = mentions.groupby('hierarchical_clustering').groups
        clusters_with_ngrams = {}
        for cluster_label, mention_names in clusters.items():
            cluster_mentions = mentions.loc[mention_names]
            raw_descriptions = cluster_mentions[
                cluster_mentions["wikidata_description"].notna()
            ]["wikidata_description"]
            ngrams = [ngram for raw_description in raw_descriptions for ngram in build_description_ngrams(raw_description)]
            clusters_with_ngrams[cluster_label] = ngrams
    events = []
    cluster_llrs = {}
    for cluster, cluster_ngrams in tqdm(clusters_with_ngrams.items()):
        llrs = {ngram: llr(cluster, ngram, events) for ngram in cluster_ngrams}
        length_sorted_llrs = sorted(llrs.items(), key=lambda x: len(x[0]), reverse=True)
        sorted_llrs = sorted(length_sorted_llrs, key=lambda x: x[1], reverse=True)
        cluster_llrs[cluster] = sorted_llrs

    return cluster_llrs, clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'embeds_graph_pickle', nargs='?', default="pickles/mentions_word_embeds_graph.pickle"
    )
    parser.add_argument(
        'output_pickle', nargs='?', default="pickles/cluster_llrs.pickle"
    )
    args = parser.parse_args()

    with open(args.embeds_graph_pickle, "rb") as f:
        mentions, embeddings, graph = pickle.load(f)

    cluster_llrs, clusters = calculate_llrs(mentions, graph)

    # cluster_ngram_counts = {key: Counter(ngrams) for key, ngrams in clusters_with_ngrams.items()}
    # ngrams = []
    # for counter in cluster_ngram_counts.values():
    #     ngrams.extend(counter.keys())
    # ngram_set = set(ngrams)

    with open(args.output_pickle, "wb") as f:
        pickle.dump(cluster_llrs, f, pickle.HIGHEST_PROTOCOL)
    if graph is not None:
        print(graph)
        vis_clusters = Network(height='700px', width='700px')#, layout=True)
        for cluster_label, cluster_nodes in clusters.items():
            colormap_value = cluster_label % color_map.N
            color = color_map(colormap_value, bytes=True)
            color_str = f"rgba({color[0]},{color[1]},{color[2]},{color[3]-10}"
            label = cluster_llrs[cluster_label]
            if len(label) > 0:
                label = str(cluster_label) + ": " + str(label[0])
            vis_clusters.add_node(str(cluster_label), color=color_str, label=label)
            for node_name in cluster_nodes:
                node = graph.nodes[node_name]
                node_mentions = mentions.loc[node["mention_names"]]
                full_mention = node_mentions.iloc[0]["full_mention"]
                vis_clusters.add_node(full_mention, color=color_str)
                vis_clusters.add_edge(str(cluster_label), full_mention)
        vis_clusters.show_buttons()
        vis_clusters.toggle_physics(True)
        vis_clusters.show("graph.html")

if __name__ == "__main__":
    main()
