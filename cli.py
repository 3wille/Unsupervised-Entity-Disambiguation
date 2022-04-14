#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
from tkinter import Place
from IPython.terminal.embed import embed
from IPython.display import display
import nltk
from prompt_toolkit import PromptSession, prompt
from utils.entity_tagger import entity_indexes
from place_unknown import PlaceUnknown
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from utils.aida.train import aida
from chinese_whispers import aggregate_clusters

parser = argparse.ArgumentParser()
parser.add_argument(
    'embeds_graph_pickle', nargs='?', default="pickles/mentions_word_embeds_graph.pickle"
)
parser.add_argument(
    'cluster_llrs_pickle', nargs='?', default="pickles/cluster_llrs.pickle"
)
args = parser.parse_args()
with open(args.embeds_graph_pickle, "rb") as f:
    mentions, embeddings, graph = pickle.load(f)
with open(args.cluster_llrs_pickle, "rb") as f:
    cluster_llrs = pickle.load(f)
clusters = aggregate_clusters(graph, label_key='cluster_label')
place_unknown = PlaceUnknown('bert-base-uncased').place_unknown

# Create prompt object.
menu_session = PromptSession()
sentence_session = PromptSession()
descriptor_session = PromptSession()


def main():
    print("")
    while(True):
        print("")
        try:
            text = menu_session.prompt(
                "Functions\n1: Sentence\n2: Cluster Descriptor\nSelect function (^d to exit): "
            )
        except EOFError:
            break
        if text == '1':
            sentence_prompt()
        elif text == '2':
            descriptor_prompt()


def sentence_prompt():
    print("")
    while(True):
        print("")
        try:
            text = sentence_session.prompt("Enter sentence (^d to exit): ")
        except EOFError:
            break
        # text = "Hugging Face Inc. is a company based in New York City."
        # tokens = nltk_word_tokenize(text)
        tokens, bert_encoding, mention_ranges = entity_indexes(text)
        display(tokens)
        display(bert_encoding)
        display(mention_ranges)
        for mention_range in mention_ranges:
            descriptions = place_unknown(tokens, mention_range[0], mention_range[1], 3)
            display(descriptions)


def descriptor_prompt():
    print("")
    while(True):
        print("")
        try:
            text = sentence_session.prompt("Enter cluster ID (^d to exit): ")
            cluster_id = int(text)
            llrs = cluster_llrs[cluster_id]
        except EOFError:
            break
        except ValueError:
            continue
        except KeyError:
            print("ID not found")
            continue
        display(llrs)
        mention_names = []
        for node_name in clusters[cluster_id]:
            mention_names.extend(graph.nodes[node_name]['mention_names'])
        display(mentions.loc[mention_names])
        while(True):
            try:
                mention_name_str = prompt("Mention ID (^d to exit): ")
                mention_name = int(mention_name_str)
                mention = mentions.loc[mention_name]
                aida_sentence = aida[aida["sentence_id"] == mention.sentence_id]
                display(" ".join(aida_sentence["token"][:-1].tolist()))
            except EOFError:
                break
            except (ValueError, KeyError):
                continue


main()
