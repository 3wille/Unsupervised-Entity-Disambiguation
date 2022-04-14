#!/usr/bin/env python
# coding: utf-8

# from csv import QUOTE_NONE

from IPython import embed

from tqdm import tqdm
import pickle
import json
import argparse
import numpy as np
import pandas as pd
from IPython.display import display
from utils.aida.train import aida
from elasticsearch import Elasticsearch
from elasticsearch import RequestsHttpConnection
import logging
import requests_cache
# from elasticsearch import helpers as es_helpers

tqdm.pandas()
logging.basicConfig(
    filename='log/find_mentions.log', encoding='utf-8',
    level=logging.INFO
)

batch_size = 10
es_url = "http://ltdocker.informatik.uni-hamburg.de:10004/"


class MyConnection(RequestsHttpConnection):
    def __init__(self, *args, **kwargs):
        proxies = kwargs.pop('proxies', {})
        super(MyConnection, self).__init__(*args, **kwargs)
        self.session.proxies = proxies


def aida_mention_is_unambiguous(mention):
    full_mention = mention["full_mention"]
    return mention_is_unambiguous(full_mention)


def batch_mention_is_unambiguous(batch_surface_forms):
    queries = list(map(json.dumps, map(search_query, batch_surface_forms)))
    headers = ['{"index": "wikidata"}' for _ in queries]
    body = [None] * (len(queries)*2)
    body[::2] = headers
    body[1::2] = queries
    responses = es.msearch(body="\n".join(body), index="wikidata")['responses']
    wikidata_titles = {}
    for response, surface_form in zip(responses, batch_surface_forms):
        wikidata_titles[surface_form] = unambiguous_response_wikidata_title(response, surface_form)
    return wikidata_titles


def search_query(mention_surface_form):
    return {
        "size": 5000,
        "query": {
            "match": {
                "labels.en": {
                    "query":  mention_surface_form,
                }
            },
        },
        "_source": [
            "labels.en", "descriptions.en", "title",
            "instance_of"
        ]
    }


def mention_is_unambiguous(mention_surface_form):
    query = search_query(mention_surface_form)
    ambiguity_results = es.search(body=query, index="wikidata")
    return unambiguous_response_wikidata_title(ambiguity_results, mention_surface_form)
    # return len(exact_matches) == 1
# unambiguous_mentions = [mention for mention in full_mentions[:10] if mention_is_unambiguous(mention)]


def unambiguous_response_wikidata_title(response, mention_surface_form):
    hits = response["hits"]["hits"]
    exact_matches = list(filter(
        lambda hit: mention_surface_form in hit["_source"]["labels"]["en"], hits
    ))
    # print(list(exact_matches))
    if len(exact_matches) == 1:
        logging.info(f"mention unambiguous: {mention_surface_form}")
        return exact_matches[0]["_source"]["title"]
    else:
        logging.info(f"mention ambiguous: {mention_surface_form}")
        return np.nan


def process_chunk(data):
    all_mentions = data[data['bi'] == 'B']
    surface_forms = [*all_mentions.groupby(by='full_mention').groups.keys()]
    print(f"found {len(surface_forms)} unique surface forms")
    for i in tqdm(range(0, len(surface_forms), batch_size)):
        # batch = all_mentions.iloc[i:i+batch_size]["full_mention"]
        batch = surface_forms[i:i+batch_size]
        wikidata_titles_by_surface_form = batch_mention_is_unambiguous(batch)
        for surface_form, wikidata_title in wikidata_titles_by_surface_form.items():
            data.loc[(data["full_mention"] == surface_form) & (data["bi"] == "B"), "wikidata_title"] = wikidata_title

    # data["wikidata_id"] = np.nan
    # data.loc[all_mentions.index, "wikidata_id"] = all_mentions.progress_apply(aida_mention_is_unambiguous, axis=1)
    unambiguous_mentions = data[data['wikidata_title'].notna()]
    unambiguous_surface_forms.extend(unambiguous_mentions['full_mention'].tolist())
    display(unambiguous_mentions)
    if args.dataset == "aida_train":
        with open(args.output_pickle, "wb") as f:
            pickle.dump(unambiguous_mentions, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.output_pickle, "ab") as f:
            pickle.dump(unambiguous_mentions.to_dict('records'), f, pickle.HIGHEST_PROTOCOL)


parser = argparse.ArgumentParser()

parser.add_argument('--socks', dest='socks', action='store_true')
parser.add_argument('--no-socks', dest='socks', action='store_false')
parser.set_defaults(feature=True)

parser.add_argument(
    'dataset', nargs='?', default="aida_train", help="'aida_train' or a pickle",
)
parser.add_argument(
    'output_pickle', nargs='?', default="pickles/unambiguous_mention.pickle"
)
args = parser.parse_args()

if __name__ == '__main__':
    unambiguous_surface_forms = []
    if args.socks:
        es = Elasticsearch(
            [es_url], retry_on_timeout=True, connection_class=MyConnection,
            proxies={'http': 'socks5://localhost:1080'}
        )
    else:
        requests_cache.install_cache()
        es = Elasticsearch([es_url], retry_on_timeout=True)
    print(es.info())
    if args.dataset == "aida_train":
        data = aida
        process_chunk(data)
    else:
        with open(args.dataset, "rb") as input_file, open(args.output_pickle, 'rb') as output_file:
            while True:
                try:
                    # pair-wise load from input and output file to scan the number of inputs
                    # already processed, should break on EOF of output file so that next loop
                    # reads the first unprocessed input
                    pickle.load(output_file)
                except EOFError:
                    print("reached end")
                    break
                pickle.load(input_file)
                print("skipped")
            while True:
                try:
                    raw_data, _ = pickle.load(input_file)
                    data = pd.DataFrame(raw_data)
                    process_chunk(data)
                except EOFError:
                    break
