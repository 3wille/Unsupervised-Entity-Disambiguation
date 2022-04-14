#!/usr/bin/env python
# coding: utf-8

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'mentions_pickle', nargs='?', default="pickles/described_unambiguous_mention.pickle"
)
parser.add_argument(
    'output_pickle', nargs='?', default="pickles/filtered_described_unambiguous_mention.pickle"
)
args = parser.parse_args()

with open(args.mentions_pickle, "rb") as f:
    unambiguous_mentions = pickle.load(f)

print(len(unambiguous_mentions))
filtered = unambiguous_mentions[unambiguous_mentions["wikidata_description"] != "Wikimedia disambiguation page"]
print(len(filtered))

with open(args.output_pickle, "wb") as f:
    pickle.dump(filtered, f, pickle.HIGHEST_PROTOCOL)
