#!/usr/bin/env python
# coding: utf-8

from IPython import embed
import math
# from sentence_transformers import util
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from rouge_metric import PyRouge

from utils.similarity import Similarity, retrieve_wikipedia_pages
# from utils.aida.train import aida as aida_train
from utils.aida.dev import aida as aida_dev
import logging
with open("pickles/filtered_word_embeddings.pickle", "rb") as f:
    unanmbiguous_mentions, embeddings = pickle.load(f)
tqdm.pandas()
logging.basicConfig(
    filename='log/unclustered_experiment.log', encoding='utf-8',
    level=logging.DEBUG
)

top_k = 5
metrics = [
    'rouge-1', 'rouge-2', 'rouge-3', 'rouge-l', 'rouge-w-1.2', 'rouge-s4', 'rouge-su4'
]
rouge = PyRouge(rouge_n=(1, 2, 3), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)

unambiguous_mentions_with_wikipedia = unanmbiguous_mentions.loc[
    unanmbiguous_mentions["wikipedia_id"].notna(),
]
wikipedia_descriptions = retrieve_wikipedia_pages(unambiguous_mentions_with_wikipedia)

embeddings_tensor = torch.Tensor(
    len(embeddings), len(embeddings[0])
)
torch.stack(embeddings, out=embeddings_tensor)
mentions_with_wikipedia = aida_dev.loc[(
    (aida_dev["bi"] == "B") & (aida_dev["wikipedia_id"].notna())
)]
calculate_similarities = Similarity("bert-base-uncased").calculate_similarities

rouge_scores = []
for _index, mention in tqdm(
    mentions_with_wikipedia.iterrows(), total=mentions_with_wikipedia.shape[0]
):
    logging.debug(mention)
    sentence = aida_dev[aida_dev["sentence_id"] == mention.sentence_id]

    sentence_tokens = sentence["token"][:-1]
    mention_start_index = mention.name - sentence.iloc[0].name
    mention_end_index = mention_start_index + len(mention.full_mention.split(" ")) - 1
    logging.debug(sentence_tokens)
    logging.debug(f"{mention_start_index}-{mention_end_index}")

    similarities = calculate_similarities(
        sentence_tokens, mention_start_index, mention_end_index, embeddings_tensor,
    )
    top_similarities, indices = torch.topk(similarities, k=top_k, sorted=False)
    surface_form = sentence_tokens[mention_start_index:mention_end_index]
    result = {"sentence": " ".join(sentence_tokens), "surface_form": surface_form}
    similar_mentions = unanmbiguous_mentions.iloc[indices[0]]

    sim_mentions_wikidata_descriptions = similar_mentions["wikidata_description"]

    wikipedia_id = str(mention.wikipedia_id)[:-2]
    if wikipedia_id not in wikipedia_descriptions.keys():
        continue
        # TODO: manche IDs sind nicht mehr korrekt, die URLs könnten aber noch passen
    page_extract = wikipedia_descriptions[wikipedia_id]

    mention_rouge_scores = {metric: 0 for metric in metrics}
    for description in sim_mentions_wikidata_descriptions:
        if description is None or (type(description) == float and math.isnan(description)):
            continue
        try:
            description_rouge_scores = rouge.evaluate([description], [[page_extract]])
        except Exception as e:
            e
            embed()
        for metric, inner_dict in description_rouge_scores.items():
            value = inner_dict['p']
            if value > mention_rouge_scores[metric]:
                mention_rouge_scores[metric] = value
    # print(mention_rouge_scores)
    rouge_scores.append(mention_rouge_scores)
df_rouge_scores = pd.DataFrame(rouge_scores)
print(df_rouge_scores.mean(axis='index'))
