#!/usr/bin/env python
# coding: utf-8

import json
import readline
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import signal
from sys import exit
from IPython.display import display
import logging
import spacy

logging.basicConfig(
    filename='log/build_wikipedia_corpus.log', encoding='utf-8',
    level=logging.DEBUG
)

sentence_id_counter = 0
stop = False

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf") # , disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp.add_pipe('sentencizer')

#######################################
#
# Dump aus 2017
# lines/pages: 5490664
#
#######################################


def current_mention(spacy_sentence, token):
    for ent in spacy_sentence.ents:
        if token.i in range(ent.start, ent.end):
            return ent


def process_page_json(line: dict):
    global sentence_id_counter
    global token_counter
    text = line['text']
    page_id = line['id']
    doc = nlp(text)
    for spacy_sentence in doc.sents:
        for token in spacy_sentence:
            token_counter += 1
            entity_tag = token.ent_iob_
            if entity_tag == 'O':
                entity_tag = np.nan
                full_mention = np.nan
            else:
                full_mention = current_mention(spacy_sentence, token).text
            row = {
                'token': token.text, 'bi': entity_tag, 'full_mention': full_mention,
                'sentence_id': sentence_id_counter, 'source_wikipedia_id': page_id,
                'token_id': token_counter,
            }
            data.append(row)
        data.append({
            'token': np.NaN, 'sentence_id': sentence_id_counter, 'source_wikipedia_id': page_id,
        })
        sentence_id_counter += 1


def handler(_signum, _frame):
    global stop
    stop = True


def dump():
    print("dumping")
    global data
    with open(args.output_pickle, "ab") as f:
        pickle.dump((data, read_lines_counter), f, pickle.HIGHEST_PROTOCOL)
    data = []
    print("dumped")


signal.signal(signal.SIGINT, handler)
parser = argparse.ArgumentParser()
parser.add_argument(
    'lines_to_read', nargs='?', default="10", type=int,
)
parser.add_argument(
    'load_pickle', nargs='?', default='True', type=str,
)
parser.add_argument(
    'input_txt', nargs='?', default='data/enwiki.txt'
)
parser.add_argument(
    'output_pickle', nargs='?', default="wikipedia_dataset/corpus.pickle"
)
args = parser.parse_args()
if args.load_pickle == 'True':
    with open(args.output_pickle, "rb") as f:
        while True:
            try:
                _, lines_offset = pickle.load(f)
            except EOFError:
                break
else:
    lines_offset = 0
data = []

token_counter = 0
read_lines_counter = 0
with open(args.input_txt, "r") as f:
    for line in tqdm(f, total=args.lines_to_read):
        if read_lines_counter == args.lines_to_read or stop:
            break
        if read_lines_counter < lines_offset:
            read_lines_counter += 1
            continue
        json_line = json.loads(line)
        process_page_json(json_line)
        if read_lines_counter % 100 == 0 and read_lines_counter != 0:
            dump()
        read_lines_counter += 1

df = pd.DataFrame(data)
display(df)
print(df.columns)

dump()
