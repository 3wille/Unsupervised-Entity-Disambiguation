import os
import pickle
from tqdm import tqdm
import torch
from utils.aida.dev import aida
from utils.similarity import retrieve_wikipedia_pages
import pandas as pd
from rouge_metric import PyRouge
import logging
import argparse

from IPython import embed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="bert-base-uncased")
    parser.add_argument(
        'embeds_graph_pickle', nargs='?', default="pickles/mentions_word_embeds_graph.pickle"
    )
    parser.add_argument(
        'cluster_means_pickle', nargs='?', default="pickles/cluster_means.pickle"
    )
    parser.add_argument(
        'cluster_llrs_pickle', nargs='?', default="pickles/cluster_llrs.pickle"
    )
    parser.add_argument(
        'assignments_pickle', nargs='?', default='pickles/cluster_assignments.pickle'
    )
    args = parser.parse_args()

    if args.model in ["bert-base-uncased", "roberta-large"]:
        from place_unknown import PlaceUnknown
        with open(args.cluster_means_pickle, "rb") as f:
            cluster_means = pickle.load(f)
        with open(args.cluster_llrs_pickle, "rb") as f:
            cluster_llrs = pickle.load(f)
        place_unknown = PlaceUnknown(
            args.model, cluster_means, cluster_llrs,
        ).place_unknown
    elif args.model == "fast_text":
        from static_experiment.place_unknown import place_unknown
    with open(args.embeds_graph_pickle, "rb") as f:
        mentions, embeddings, graph = pickle.load(f)

    cluster_assignments = place_dev_set(place_unknown)

    with open(args.assignments_pickle, 'wb') as f:
        pickle.dump(cluster_assignments, f)


dev_set_mentions = aida.loc[((aida["bi"] == "B") & (aida["wikipedia_id"].notna()))]

top_k = 3
metrics = ['rouge-1', 'rouge-2', 'rouge-3', 'rouge-l', 'rouge-w-1.2', 'rouge-s4', 'rouge-su4']
rouge = PyRouge(rouge_n=(1, 2, 3), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)


def place_dev_set(place_unknown):
    cluster_assignments = {}
    rouge_scores = []

    for _index, mention in tqdm(
        dev_set_mentions.iterrows(), total=dev_set_mentions.shape[0]
    ):
        logging.debug(mention)
        sentence = aida[aida["sentence_id"] == mention.sentence_id]

        sentence_tokens = sentence["token"][:-1]
        mention_start_index = mention.name - sentence.iloc[0].name
        mention_end_index = mention_start_index + len(mention.full_mention.split(" ")) - 1
        logging.debug(sentence_tokens)
        logging.debug(f"{mention_start_index}-{mention_end_index}")

        result = place_unknown(
            sentence_tokens, mention_start_index, mention_end_index, top_k
        )
        cluster_id = result[f"1 cluster_id"]
        if cluster_id in cluster_assignments.keys():
            cluster_assignments[cluster_id].append(mention)
        else:
            cluster_assignments[cluster_id] = [mention]
        descriptions = []
        for i in range(top_k):
            i_descriptions = list(map(lambda x: " ".join(x[0]), result[f"{i} top3 llr"][:3]))
            descriptions.extend(i_descriptions)

        wikipedia_id = str(mention.wikipedia_id)[:-2]
        if wikipedia_id not in wikipedia_descriptions.keys():
            continue  # TODO: manche IDs sind nicht mehr korrekt, die URLs könnten aber noch passen
        page_extract = wikipedia_descriptions[wikipedia_id]

        mention_rouge_scores = {metric: 0 for metric in metrics}
        for description in descriptions:
            description_rouge_scores = rouge.evaluate([description], [[page_extract]])
            for metric, inner_dict in description_rouge_scores.items():
                value = inner_dict['p']
                if value > mention_rouge_scores[metric]:
                    mention_rouge_scores[metric] = value
        # print(mention_rouge_scores)
        rouge_scores.append(mention_rouge_scores)
    df_rouge_scores = pd.DataFrame(rouge_scores)
    print(df_rouge_scores.mean(axis='index'))
    return cluster_assignments, df_rouge_scores
    #     results.append(result)
    # df_results = pd.DataFrame(results)
    # print(df_results)
    # with open(df_results_filename, "wb") as f:
    #     pickle.dump(df_results, f, pickle.HIGHEST_PROTOCOL)
    # return df_results


logging.basicConfig(filename='log/eval_dev_set.log', encoding='utf-8', level=logging.DEBUG)
wikipedia_descriptions = retrieve_wikipedia_pages(dev_set_mentions)
if __name__ == '__main__':
    main()
