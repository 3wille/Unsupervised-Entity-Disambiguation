#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
from IPython.display import display
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'input_pickle', nargs='?', default="wikipedia_dataset/unambiguous_mention.pickle"
)
parser.add_argument(
    'output_pickle', nargs='?', default="wikipedia_dataset/merged_unambiguous_mentions.pickle"
)
args = parser.parse_args()

data = []
with open(args.input_pickle, 'rb') as f:
    pickle_counter = 0
    while True:
        try:
            data_rows = pickle.load(f)
        except EOFError:
            break
        for data_row in data_rows:
            data_row['chunk_id'] = pickle_counter
        data.extend(data_rows)
        pickle_counter += 1

mentions = pd.DataFrame(data)
display(mentions)

with open(args.output_pickle, 'wb') as f:
    pickle.dump(mentions, f)
