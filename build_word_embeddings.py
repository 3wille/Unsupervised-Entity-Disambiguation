#!/usr/bin/env python
# coding: utf-8

import argparse
from IPython import embed
from IPython.display import display
import pickle
import torch
from transformers import BertModel, AutoModel
from tqdm import tqdm
from utils.aida.train import aida
from utils import retok
import pandas as pd
tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument(
    'model_name', nargs='?', default="bert-base-uncased"
)
parser.add_argument(
    'dataset', nargs='?', default="aida_train", help="'aida_train' or a pickle",
)
parser.add_argument(
    'mentions_pickle', nargs='?', default="pickles/filtered_described_unambiguous_mention.pickle"
)
parser.add_argument(
    'output_pickle', nargs='?', default="pickles/word_embeddings.pickle"
)
args = parser.parse_args()

with open(args.mentions_pickle, "rb") as f:
    unambiguous_mentions = pickle.load(f)

t = retok.ReTokenizer(args.model_name)
model = AutoModel.from_pretrained(
    args.model_name,
    output_hidden_states=True,
)
model.eval()

word_embeddings = []


def process_chunk(data, chunk_mentions):
    def build_sentence(mention):
        # index = mention.name
        sentence_id = mention.sentence_id
        sentence = data[data["sentence_id"] == sentence_id]
        tokens = sentence["token"][:-1].to_list()
        sentences.append(tokens)
    sentences = []
    chunk_mentions.apply(build_sentence, axis=1)
    # sentences_and_mention_ranges = unambiguous_mentions.progress_apply(build_sentence, axis=1)
    # unambiguous_mentions[["sentence", "mention_range"]] = sentences_and_mention_ranges

    for i, sentence_tokens in enumerate(tqdm(sentences)):
        try:
            # tokens = tokenizer(
            #     sentences, is_split_into_words=True, return_tensors="pt", padding=True
            # )
            mention = chunk_mentions.iloc[i]
            df_sentence = data[data["sentence_id"] == mention.sentence_id]
            unambiguous_mentions.loc[mention.name, 'embedding_id'] = len(word_embeddings)
            mention_start_index = int(mention.token_id - df_sentence.iloc[0].token_id)
            mention_end_index = int(mention_start_index + len(mention.full_mention.split(" ")) - 1)

            tokens, ind, l = t.retokenize_and_encode_indexed(sentence_tokens)
            if args.model_name == "bert-base-uncased":
                token_type_ids = tokens.token_type_ids
            else:
                token_type_ids = None
            with torch.no_grad():
                model_output = model(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    token_type_ids=token_type_ids
                )
            last_layer_token_embeddings = model_output[0][0]
            mention_indexes = []
            for j in range(mention_start_index, mention_end_index + 1):
                mention_indexes.extend(ind[0][j][1])
            mention_embeddings = last_layer_token_embeddings[mention_indexes]
            mention_embedding = torch.mean(mention_embeddings, 0)
            word_embeddings.append(mention_embedding)
            # sentence_emb = build_sentence_embedding(model_output, tokens.attention_mask)
        except Exception as e:
            print(e)
            embed()


if args.dataset == "aida_train":
    process_chunk(aida, unambiguous_mentions)
else:
    with open(args.dataset, "rb") as f:
        chunk_id = 0
        while True:
            try:
                raw_data, _ = pickle.load(f)
                data = pd.DataFrame(raw_data)
            except EOFError:
                break
            chunk_mentions = unambiguous_mentions[unambiguous_mentions['chunk_id']==chunk_id]
            print(f"chunk {chunk_id}")
            process_chunk(data, chunk_mentions)
            chunk_id += 1

unambiguous_mentions.embedding_id.astype('int')
with open(args.output_pickle, "wb") as f:
    pickle.dump([unambiguous_mentions, word_embeddings], f, pickle.HIGHEST_PROTOCOL)
