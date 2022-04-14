.PHONY: eval eval_dev_set static_experiment unclustered_experiment roberta_experiment analysis

# all: pickles/unambiguous_mention.pickle pickles/enh_unambiguous_mention.pickle pickles/enh2_unambiguous_mention.pickle pickles/enh_unambiguous_mention.pickle pickles/word_embeddings.pickle pickles/mentions_word_embeds_graph.pickle pickles/cluster_llrs.pickle pickles/wikipedia_page_extracts.pickle pickles/cluster_means.pickle
# all: pickles/cluster_means.pickle pickles/wikipedia_page_extracts.pickle pickles/cluster_llrs.pickle

eval: eval_dev_set unclustered_experiment static_experiment roberta_experiment

eval_dev_set: eval_dev_set.py place_unknown.py utils/retok.py utils/similarity.py pickles/cluster_llrs.pickle pickles/cluster_means.pickle pickles/mentions_word_embeds_graph.pickle
	python eval_dev_set.py

unclustered_experiment: eval_unclustered_experiment.py pickles/word_embeddings.pickle utils/similarity.py
	python eval_unclustered_experiment.py

static_experiment: static_experiment/pickles/*.pickle eval_dev_set.py static_experiment/place_unknown.py
	ipython eval_dev_set.py fast_text static_experiment/pickles/clustering.pickle static_experiment/pickles/cluster_means.pickle static_experiment/pickles/llrs.pickle static_experiment/cluster_assignments.pickle

roberta_experiment: roberta_experiment/pickles/*.pickle eval_dev_set.py place_unknown.py
	ipython eval_dev_set.py roberta-large roberta_experiment/pickles/clustering.pickle roberta_experiment/pickles/cluster_means.pickle roberta_experiment/pickles/llrs.pickle roberta_experiment/cluster_assignments.pickle

wikipedia_experiment: wikipedia_dataset/clustering.pickle wikipedia_dataset/cluster_means.pickle wikipedia_dataset/llrs.pickle eval_dev_set.py place_unknown.py
	python eval_dev_set.py bert-base-uncased wikipedia_dataset/clustering.pickle wikipedia_dataset/cluster_means.pickle wikipedia_dataset/llrs.pickle wikipedia_dataset/cluster_assignments.pickle

unmerged_experiment: unmerged_experiment/*.pickle eval_dev_set.py place_unknown.py
	ipython eval_dev_set.py bert-base-uncased unmerged_experiment/clustering.pickle unmerged_experiment/cluster_means.pickle unmerged_experiment/llrs.pickle unmerged_experiment/assignments.pickle

hierarchical_clustering: hierarchical_clustering/*.pickle eval_dev_set.py place_unknown.py
	ipython eval_dev_set.py bert-base-uncased hierarchical_clustering/clustering.pickle hierarchical_clustering/cluster_centers.pickle hierarchical_clustering/llrs.pickle hierarchical_clustering/assignments.pickle

analysis: ../thesis/pyplots/cluster_sizes_barplot.tikz ../thesis/tables/rouge_by_cluster_sizes.tex

../thesis/pyplots/cluster_sizes_barplot.tikz ../thesis/tables/rouge_by_cluster_sizes.tex: analysis/cluster_sizes.py
	python analysis/cluster_sizes.py

pickles/unambiguous_mention.pickle: find_mentions.py
	python find_mentions.py --no-socks aida_train pickles/unambiguous_mention.pickle

pickles/described_unambiguous_mention.pickle: add_wikidata_description.py pickles/unambiguous_mention.pickle
	python add_wikidata_description.py

pickles/filtered_described_unambiguous_mention.pickle: filter_disambiguation_pages.py pickles/described_unambiguous_mention.pickle
	python filter_disambiguation_pages.py

pickles/word_embeddings.pickle: build_word_embeddings.py pickles/filtered_described_unambiguous_mention.pickle
	python build_word_embeddings.py

pickles/mentions_word_embeds_graph.pickle: cluster_embeddings.py pickles/word_embeddings.pickle
	python cluster_embeddings.py

pickles/cluster_llrs.pickle: build_cluster_labels_llr.py pickles/mentions_word_embeds_graph.pickle
	python build_cluster_labels_llr.py

pickles/wikipedia_page_extracts.pickle: load_wikipedia_extracts.py pickles/mentions_word_embeds_graph.pickle
	python load_wikipedia_extracts.py

pickles/cluster_means.pickle: calculate_cluster_centers.py pickles/mentions_word_embeds_graph.pickle
	python calculate_cluster_centers.py

data/aida-yago2-dataset-train.tsv:
	head -n 218505 aida/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv > data/aida-yago2-dataset-train.tsv

data/aida-yago2-dataset-dev.tsv:
	tail -n +218505 aida/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv | head -n 54803 > data/aida-yago2-dataset-dev.tsv

data/aida-yago2-dataset-test.tsv: data/aida-yago2-dataset-dev.tsv
	tail -n +54804 data/aida-yago2-dataset-dev.tsv > data/aida-yago2-dataset-test.tsv

static_experiment/pickles/embeddings.pickle: pickles/filtered_enh2_unambiguous_mention.pickle static_experiment/build_static_embeddings.py
	ipython static_experiment/build_static_embeddings.py

static_experiment/pickles/clustering.pickle: static_experiment/pickles/embeddings.pickle cluster_embeddings.py
	ipython cluster_embeddings.py static_experiment/pickles/embeddings.pickle static_experiment/pickles/clustering.pickle

static_experiment/pickles/llrs.pickle: static_experiment/pickles/clustering.pickle build_cluster_labels_llr.py
	ipython build_cluster_labels_llr.py static_experiment/pickles/clustering.pickle static_experiment/pickles/llrs.pickle

static_experiment/pickles/cluster_means.pickle: static_experiment/pickles/clustering.pickle calculate_cluster_centers.py
	python calculate_cluster_centers.py static_experiment/pickles/clustering.pickle static_experiment/pickles/cluster_means.pickle

roberta_experiment/pickles/word_embeddings.pickle: build_word_embeddings.py pickles/filtered_enh2_unambiguous_mention.pickle
	python build_word_embeddings.py roberta-large aida_train roberta_experiment/pickles/word_embeddings.pickle

roberta_experiment/pickles/clustering.pickle: cluster_embeddings.py roberta_experiment/pickles/word_embeddings.pickle
	ipython cluster_embeddings.py roberta_experiment/pickles/word_embeddings.pickle roberta_experiment/pickles/clustering.pickle

roberta_experiment/pickles/llrs.pickle: roberta_experiment/pickles/clustering.pickle build_cluster_labels_llr.py
	ipython build_cluster_labels_llr.py roberta_experiment/pickles/clustering.pickle roberta_experiment/pickles/llrs.pickle

roberta_experiment/pickles/cluster_means.pickle: roberta_experiment/pickles/clustering.pickle calculate_cluster_centers.py
	python calculate_cluster_centers.py roberta_experiment/pickles/clustering.pickle roberta_experiment/pickles/cluster_means.pickle

wikipedia_dataset/corpus.pickle: build_wikipedia_corpus.py
	python build_wikipedia_corpus.py 10

wikipedia_dataset/unambiguous_mention.pickle: find_mentions.py wikipedia_dataset/corpus.pickle
	python find_mentions.py --no-socks wikipedia_dataset/corpus.pickle wikipedia_dataset/unambiguous_mention.pickle

wikipedia_dataset/merged_unambiguous_mentions.pickle: merge_unambiguous_results.py wikipedia_dataset/unambiguous_mention.pickle
	python merge_unambiguous_results.py

wikipedia_dataset/described_unambiguous_mention.pickle: add_wikidata_description.py wikipedia_dataset/merged_unambiguous_mentions.pickle
	python add_wikidata_description.py wikipedia_dataset/merged_unambiguous_mentions.pickle wikipedia_dataset/described_unambiguous_mention.pickle

wikipedia_dataset/filtered_described_unambiguous_mention.pickle: filter_disambiguation_pages.py wikipedia_dataset/described_unambiguous_mention.pickle
	python filter_disambiguation_pages.py wikipedia_dataset/described_unambiguous_mention.pickle wikipedia_dataset/filtered_described_unambiguous_mention.pickle

wikipedia_dataset/embeddings.pickle: build_word_embeddings.py wikipedia_dataset/filtered_described_unambiguous_mention.pickle
	python build_word_embeddings.py bert-base-uncased wikipedia_dataset/corpus.pickle wikipedia_dataset/filtered_described_unambiguous_mention.pickle wikipedia_dataset/embeddings.pickle

wikipedia_dataset/clustering.pickle: cluster_embeddings.py wikipedia_dataset/embeddings.pickle
	python cluster_embeddings.py wikipedia_dataset/embeddings.pickle wikipedia_dataset/clustering.pickle

wikipedia_dataset/llrs.pickle: wikipedia_dataset/clustering.pickle build_cluster_labels_llr.py
	python build_cluster_labels_llr.py wikipedia_dataset/clustering.pickle wikipedia_dataset/llrs.pickle

wikipedia_dataset/cluster_means.pickle: wikipedia_dataset/clustering.pickle calculate_cluster_centers.py
	python calculate_cluster_centers.py wikipedia_dataset/clustering.pickle wikipedia_dataset/cluster_means.pickle

hierarchical_clustering/clustering.pickle hierarchical_clustering/cluster_centers.pickle: hierarchical_clustering.py pickles/word_embeddings.pickle
	python hierarchical_clustering.py

hierarchical_clustering/llrs.pickle: build_cluster_labels_llr.py hierarchical_clustering/clustering.pickle
	python build_cluster_labels_llr.py hierarchical_clustering/clustering.pickle hierarchical_clustering/llrs.pickle

unmerged_experiment/clustering.pickle: cluster_embeddings.py pickles/word_embeddings.pickle
	ipython cluster_embeddings.py pickles/word_embeddings.pickle unmerged_experiment/clustering.pickle index

unmerged_experiment/llrs.pickle: unmerged_experiment/clustering.pickle build_cluster_labels_llr.py
	ipython build_cluster_labels_llr.py unmerged_experiment/clustering.pickle unmerged_experiment/llrs.pickle

unmerged_experiment/cluster_means.pickle: unmerged_experiment/clustering.pickle calculate_cluster_centers.py
	python calculate_cluster_centers.py unmerged_experiment/clustering.pickle unmerged_experiment/cluster_means.pickle
