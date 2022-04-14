#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import torch
import math
import itertools
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import matplotlib
from IPython.display import display
from IPython import embed
import argparse
from matplotlib.pyplot import cm
from sklearn.manifold import TSNE
import tikzplotlib

class Clusterer():
    def __init__(self, mentions, embeddings):
        self.mentions = mentions
        self.embeddings = embeddings
        embeddings_tensor = torch.stack(embeddings)
        print(embeddings_tensor.shape)
        grouped_mentions = mentions.groupby('full_mention')
        grouped_embeddings = []
        self.embedding_index_to_surface_form = {}
        for surface_form, indices in grouped_mentions.indices.items():
            group = torch.stack([embeddings_tensor[index] for index in indices])
            group_embedding = group.mean(dim=0)
            index = len(grouped_embeddings)
            grouped_embeddings.append(group_embedding)
            self.embedding_index_to_surface_form[index] = surface_form

        grouped_embeddings = torch.stack(grouped_embeddings)
        print(grouped_embeddings.shape)
        self.normalized_embeddings = grouped_embeddings / np.linalg.norm(grouped_embeddings, axis=1, keepdims=True)

    def cluster(self, **kwargs):
        clustering_model = AgglomerativeClustering(**kwargs)
        print("Starting to cluster")
        clustering_model = clustering_model.fit(self.normalized_embeddings)
        cluster_assignments = clustering_model.labels_
        print(len(cluster_assignments))
        embeddings_by_cluster = {label: [] for label in set(cluster_assignments)}
        for index, assignment in enumerate(cluster_assignments):
            surface_form = self.embedding_index_to_surface_form[index]
            self.mentions.loc[self.mentions['full_mention'] == surface_form, "hierarchical_clustering"] = assignment
            embeddings_by_cluster[assignment].append(self.embeddings[index])
        cluster_centers = {}
        for cluster_name, cluster_embeddings in embeddings_by_cluster.items():
            cluster_embeddings_tensor = torch.stack(cluster_embeddings)
            cluster_mean = torch.mean(cluster_embeddings_tensor, dim=0)
            cluster_centers[cluster_name] = cluster_mean

        # mentions["hierarchical_clustering"] = cluster_assignments
        self.mentions.hierarchical_clustering = self.mentions.hierarchical_clustering.astype('int')
        return clustering_model, cluster_centers

    # def plot_scatter(self, model):
    #     mentions = self.mentions
    #     pca = PCA(n_components=2)
    #     pca.fit(self.normalized_embeddings)
    #     reduced = pca.transform(self.normalized_embeddings)
    #     x, y = zip(*reduced)

    #     cluster_assignments = []
    #     for surface_form in self.embedding_index_to_surface_form.values():
    #         surface_form_mentions = mentions[mentions['full_mention']==surface_form]
    #         cluster = surface_form_mentions.iloc[0].hierarchical_clustering.astype("int")
    #         cluster_assignments.append(cluster)

    #     count_of_clusters = len(set(model.labels_))
    #     color_map = cm.rainbow(np.linspace(0, 1, count_of_clusters))
    #     colors = [color_map[assignment] for assignment in cluster_assignments]
    #     # plt.scatter(x, y, c=cluster_assignments, cmap='gist_rainbow')
    #     plt.scatter(x, y, c=colors)
    #     plt.show()

    def plot_tsne(self, model):
        mentions = self.mentions
        reduced = TSNE(n_components=2).fit_transform(self.normalized_embeddings)
        # x, y = zip(*reduced)

        reduced_by_cluster = {}
        for i, reduced_embedding in enumerate(reduced):
            surface_form = self.embedding_index_to_surface_form[i]
            surface_form_mentions = mentions[mentions['full_mention']==surface_form]
            cluster = surface_form_mentions.iloc[0].hierarchical_clustering.astype("int")
            if cluster in reduced_by_cluster.keys():
                reduced_by_cluster[cluster].append(reduced_embedding)
            else:
                reduced_by_cluster[cluster] = [reduced_embedding]

        count_of_clusters = len(set(model.labels_))
        markers = ['o', 'v', '^', '<', '>', 's', 'P', '*']
        color_map = cm.rainbow(np.linspace(0, 1, math.ceil(count_of_clusters/len(markers))))
        styles = list(itertools.product(markers, color_map))
        # for (x, y), cluster_assignment in zip(reduced, cluster_assignments):
        for cluster_assignment, reduced_embedding in reduced_by_cluster.items():
            # color = color_map[cluster_assignment]
            marker, color = styles[cluster_assignment]
            print(reduced_embedding)
            xy = np.asarray(reduced_embedding)
            x = xy[:, 0]
            y = xy[:, 1]
            marker_size = (matplotlib.rcParams['lines.markersize'] ** 2)/3
            plt.scatter(x, y, s=marker_size, c=color, label=cluster_assignment, marker=marker, edgecolors='black', linewidths=0.25)
            # plt.scatter(x, y, label=cluster_assignment, edgecolors='black', linewidths=0.25)
        # plt.legend(loc="lower center", ncol=7, fontsize='small', bbox_to_anchor=(0.5, -0.5))
        # tikzplotlib.save("../thesis/pyplots/hierarchical_clustering_tsne.tikz", strict=True)
        plt.savefig("../thesis/pyplots/hierarchical_clustering_tsne.pdf", bbox_inches='tight')
        plt.savefig("meetings/02_24_hierarchical_tsne.png")
        # plt.show()
        plt.close()


    def plot_dendogram2(self, model):
        mapped_tree = []
        for pair in model.children_:
            if pair[0] < len(self.embedding_index_to_surface_form):
                left = self.embedding_index_to_surface_form[pair[0]]
            else:
                left = pair[0]
            if pair[1] < len(self.embedding_index_to_surface_form):
                right = self.embedding_index_to_surface_form[pair[1]]
            else:
                right = pair[1]
            mapped_tree.append([left, right])

    def plot_dendrogram(self, model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        print(linkage_matrix.shape)
        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, labels=self.embedding_index_to_surface_form, **kwargs)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.savefig("meetings/02_17_hierarchical_dendo.png")
        # plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'embeddings_pickle', nargs='?', default='pickles/word_embeddings.pickle'
    )
    parser.add_argument(
        'cluster_output_pickle', nargs='?', default='hierarchical_clustering/clustering.pickle'
    )
    parser.add_argument(
        'cluster_center_output_pickle', nargs='?', default='hierarchical_clustering/cluster_centers.pickle'
    )
    args = parser.parse_args()
    with open(args.embeddings_pickle, 'rb') as f:
        mentions, embeddings = pickle.load(f)

    clusterer = Clusterer(mentions, embeddings)
    threshold = 0.6
    model, cluster_centers = clusterer.cluster(
        affinity='cosine',
        linkage='average',
        n_clusters=None,
        distance_threshold=threshold,
        memory="data/cache/sklearn"
    )

    display(mentions)
    with open(args.cluster_output_pickle, 'wb') as f:
        pickle.dump((mentions, embeddings, None), f)
    with open(args.cluster_center_output_pickle, 'wb') as f:
        pickle.dump(cluster_centers, f)

    # embed()
    clusterer.plot_tsne(model)
    # clusterer.plot_dendrogram(model, truncate_mode="level", p=5, color_threshold=threshold)

if __name__ == '__main__':
    main()
