#!/usr/bin/env python
# coding: utf-8

import pickle
from chinese_whispers import chinese_whispers, aggregate_clusters
import math
from utils.general import batch
import requests
import requests_cache

requests_cache.install_cache('data/cache/wikipedia_extracts', backend='sqlite', expire_after=-1)
with open("pickles/mentions_word_embeds_graph.pickle", "rb") as f:
    mentions, embeddings, graph = pickle.load(f)
page_descriptions = {}


def request_wikipedia_pages(ids):
    filtered_ids = list(filter(lambda id: id not in page_descriptions.keys(), ids))
    if len(filtered_ids) == 0:
        return
    params = {
        "format": "json",
        "action": "query",
        "pageids": "|".join(map(str, filtered_ids)),
        "prop": "info|extracts|categories|pageprops|description|revisions",
        "exsentences": 1,
        "exintro": True,
        "explaintext": True,
    }
    response = requests.get("https://en.wikipedia.org/w/api.php", params=params)
    response_content = response.json()
    # print(response_content.keys())
    if 'error' in response_content:
        print(response_content['error'])
    if 'warnings' in response_content:
        print(response_content['warnings'])
    return response


if __name__ == '__main__':
    clusters = aggregate_clusters(graph, label_key='cluster_label')
    cluster_wikipedia_ids = {}
    wikipedia_ids = []
    for cluster_id, cluster_node_names in clusters.items():
        cluster_wikipedia_ids[cluster_id] = []
        for node_name in cluster_node_names:
            node = graph.nodes[node_name]
            node_mention_name = node["mention_names"]
            node_mentions = mentions.loc[node_mention_name]
            wikipedia_id = node_mentions["wikipedia_id"].iloc[0]
            if not math.isnan(wikipedia_id):
                cluster_wikipedia_ids[cluster_id].append(int(wikipedia_id))
                wikipedia_ids.append(int(wikipedia_id))

    batched_ids = batch(wikipedia_ids, n=20)
    for wikipedia_ids in batched_ids:
        response = request_wikipedia_pages(wikipedia_ids)
        for page_id, page in response.json()['query']['pages'].items():
            if 'extract' not in page.keys():
                print(f"no extract, page id: {page_id}")
                print(page.keys())
                # print(page['description'])
                continue
            extract = page["extract"]
            page_descriptions[int(page_id)] = extract
            # tokenized_description = nltk.tokenize.word_tokenize(raw_description)
            # filtered_tokens = filter(lambda token: token not in token_deny_list, tokenized_description)
            # filtered_lemmas = map(lemmatizer.lemmatize, filtered_tokens)
            # article_ngrams = list(everygrams(
            #     list(filtered_lemmas), max_len=ngram_n, pad_left=pad_left, pad_right=pad_right
            # ))

    with open("pickles/wikipedia_page_extracts.pickle", "wb") as f:
        pickle.dump(page_descriptions, f, pickle.HIGHEST_PROTOCOL)
