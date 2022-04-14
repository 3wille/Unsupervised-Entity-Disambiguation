#!/usr/bin/env python
# coding: utf-8

from IPython.terminal.embed import embed
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from IPython.display import display
from itertools import groupby
from operator import itemgetter

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForTokenClassification.from_pretrained(
  "dbmdz/bert-large-cased-finetuned-conll03-english"
)
model.eval()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

label_list = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]


def entity_indexes(sequence):
    # Bit of a hack to get the tokens with the special tokens
    # tokens = tokenizer.tokenize(sequence)
    encoding = tokenizer.encode_plus(sequence, return_tensors="pt").to(device)

    tokens = []
    word_indexes = list(set(
        [word_index for word_index in encoding.word_ids() if word_index is not None]
    ))
    for word_index in word_indexes:
        span = encoding.word_to_chars(word_index)
        tokens.append(sequence[span.start:span.end])

    for token_index, token in enumerate(encoding.tokens()):
        word_index = encoding.token_to_word(token_index)
        if word_index is not None:
            span = encoding.word_to_chars(word_index)
            # print(sequence[span.start:span.end])
    outputs = model(
        input_ids=encoding.input_ids,
    )[0]
    predictions = torch.argmax(outputs, dim=2)
    # print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])

    ranges = []
    # print(encoding.tokens())
    for k, g in groupby(enumerate(predictions[0]), key=lambda x: x[1]):
        if k == torch.tensor(0):
            continue
        bert_indexes = list(map(itemgetter(0), g))
        # print(bert_indexes)
        bert_indexes = list(map(encoding.token_to_word, bert_indexes))
        # print(bert_indexes)
        index_set = set(bert_indexes)
        ranges.append((min(index_set), max(index_set)))
    # print(ranges)
    return tokens, encoding, ranges


if __name__ == "__main__":
    sequences = [
        "Hugging Face Inc. is a company based in New York City.",
        "Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge which is " \
            "visible from the window.",
    ]
    for sequence in sequences:
        print(entity_indexes(sequence))

# for index, prediction in enumerate(predictions[0]):
#     word_index = inputs.token_to_word(index)
#     if word_index is None:
#         continue
#     char_span = inputs.word_to_chars(word_index)
#     word = sequence[char_span.start:char_span.end]
#     display((word, prediction))
