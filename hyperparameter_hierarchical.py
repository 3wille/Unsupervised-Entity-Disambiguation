from re import A
from eval_dev_set import place_dev_set
from hierarchical_clustering import Clusterer
from build_cluster_labels_llr import calculate_llrs
from place_unknown import PlaceUnknown
import pickle
import pandas as pd

from IPython import embed
from IPython.display import display

with open('pickles/word_embeddings.pickle', 'rb') as f:
    mentions, embeddings = pickle.load(f)

param_sets = {
    "euclidean": {
        "ward":[0.85, 0.925],
    },
    "cosine": {
        "average": [0.15, 0.3, 0.6],
        "complete": [0.15, 0.3, 0.7],
        "single": [0.15, 0.3, 0.4, 0.5],
    }
}

assignments = {}
scores = {}
skipped = []
for affinity, params in param_sets.items():
        scores[affinity] = {}
        for linkage, thresholds in params.items():
            scores[affinity][linkage] = {}
            for threshold in thresholds:
                try:
                    print(affinity)
                    print(linkage)
                    print(threshold)
                    current_mentions = mentions.copy()
                    clusterer = Clusterer(current_mentions, embeddings)
                    _, cluster_centers = clusterer.cluster(affinity=affinity, linkage=linkage, distance_threshold=threshold, n_clusters=None, memory="data/cache/sklearn")
                    cluster_llrs, clusters = calculate_llrs(current_mentions, None)
                    place_unknown = PlaceUnknown("bert-base-uncased", cluster_centers, cluster_llrs).place_unknown

                    cluster_assignments, rouge_scores = place_dev_set(place_unknown)
                    scores[affinity][linkage][threshold] = rouge_scores.mean(axis='index')
                # assignments[threshold] = cluster_assignments
                except:
                    skipped.append((affinity, linkage, threshold))
                    continue

d = {}
for affinity, linkages in scores.items():
    e = {}
    for linkage, thresholds in linkages.items():
        e[linkage] = pd.DataFrame(thresholds)
    d[affinity] = pd.concat(e, axis="columns")
scores_df = pd.concat(d, axis="columns")
display(scores_df)
print(f"skipped: {skipped}")
embed()
