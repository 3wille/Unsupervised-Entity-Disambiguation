#!/usr/bin/env python
# coding: utf-8

# from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api, LdiResponseNotOk
from tqdm import tqdm
from termcolor import colored
# from nltk.util import ngrams
import requests_cache
import pickle
import numpy as np
import argparse

requests_cache.install_cache('data/cache/wikidata_descriptions', backend='sqlite', expire_after=-1)
tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument(
    'mentions_pickle', nargs='?', default="pickles/unambiguous_mention.pickle"
)
parser.add_argument(
    'output_pickle', nargs='?', default="pickles/described_unambiguous_mention.pickle"
)
args = parser.parse_args()

with open(args.mentions_pickle, "rb") as f:
    mentions = pickle.load(f)


def add_wikidata_description(mention):
    try:
        wikidata_dict = get_entity_dict_from_api(mention["wikidata_title"])
    except LdiResponseNotOk as err:
        print(colored(mention.name, color='red'))
        print(colored(err, color='red'))
        return np.nan
    descriptions = wikidata_dict["descriptions"]
    if "en" in descriptions:
        # print(descriptions["en"])
        return descriptions["en"]["value"]


mentions["wikidata_description"] = mentions.progress_apply(add_wikidata_description, axis='columns')
print(mentions)

with open(args.output_pickle, "wb") as f:
    pickle.dump(mentions, f, pickle.HIGHEST_PROTOCOL)
