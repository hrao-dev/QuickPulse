# cluster_news.py
# Clusters news articles using HDBSCAN, labels clusters with TF-IDF n-grams and LDA topics,
# and falls back to a representative summary if the label is too vague.

import numpy as np
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import LatentDirichletAllocation
import hdbscan
import umap

def generate_embeddings(df, content_column):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df[content_column].tolist(), show_progress_bar=True)
    return np.array(embeddings)

def reduce_dimensions(embeddings, n_neighbors=10, min_dist=0.0, n_components=5, random_state=42):
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        return embeddings
    n_components = min(max(2, n_components), n_samples - 2)
    n_neighbors = min(max(2, n_neighbors), n_samples - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric='cosine'
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced

def cluster_with_hdbscan(embeddings, min_cluster_size=2, min_samples=1):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(embeddings)
    return labels, clusterer

def extract_tfidf_labels(df, content_column, cluster_labels, top_n=6):
    grouped = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        if label == -1: continue
        grouped[label].append(df.iloc[idx][content_column])
    tfidf_labels = {}
    for cluster_id, texts in grouped.items():
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50)
        tfidf_matrix = vectorizer.fit_transform(texts)
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        if len(avg_tfidf) == 0:
            tfidf_labels[cluster_id] = []
            continue
        top_indices = np.argsort(avg_tfidf)[::-1][:top_n]
        top_terms = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        tfidf_labels[cluster_id] = top_terms
    return tfidf_labels

def lda_topic_modeling(texts, n_topics=1, n_words=6):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
    X = vectorizer.fit_transform(texts)
    if X.shape[0] < n_topics:
        n_topics = max(1, X.shape[0])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    topic_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-n_words - 1:-1]
        words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        topic_words.extend(words)
    return topic_words

def get_representative_summary(df, cluster_indices, embeddings, centroid):
    cluster_embs = embeddings[cluster_indices]
    dists = cosine_distances(cluster_embs, centroid.reshape(1, -1)).flatten()
    min_idx = np.argmin(dists)
    return df.iloc[cluster_indices[min_idx]]["summary"]

def label_clusters_hybrid(df, content_column, summary_column, cluster_labels, embeddings, tfidf_labels, lda_labels, vague_threshold=15):
    cluster_label_map = {}
    cluster_primary_topics = {}
    cluster_related_topics = {}
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue
        topics = lda_labels.get(cluster_id, []) or tfidf_labels.get(cluster_id, [])
        topics = [t for t in topics if t]
        primary_topics = topics[:3]
        related_topics = topics[3:]
        label = ", ".join(primary_topics) if primary_topics else ""
        if not label or len(label) < vague_threshold:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            centroid = embeddings[cluster_indices].mean(axis=0)
            rep_summary = get_representative_summary(df, cluster_indices, embeddings, centroid)
            label = rep_summary[:80] + "..." if len(rep_summary) > 80 else rep_summary
        cluster_label_map[cluster_id] = label
        cluster_primary_topics[cluster_id] = primary_topics
        cluster_related_topics[cluster_id] = related_topics
    return cluster_label_map, cluster_primary_topics, cluster_related_topics

def cluster_and_label_articles(
    df, 
    content_column="content", 
    summary_column="summary", 
    min_cluster_size=2, 
    min_samples=1, 
    n_neighbors=10, 
    min_dist=0.0, 
    n_components=5, 
    top_n=6,
    lda_n_topics=1,
    lda_n_words=6,
    vague_threshold=15
):
    if df.empty:
        return None

    min_cluster_size = max(2, min(min_cluster_size, len(df) // 2)) if len(df) < 20 else min_cluster_size

    embeddings = generate_embeddings(df, content_column)
    reduced_embeddings = reduce_dimensions(embeddings, n_neighbors, min_dist, n_components)
    cluster_labels, clusterer = cluster_with_hdbscan(reduced_embeddings, min_cluster_size, min_samples)
    df['cluster_id'] = cluster_labels

    tfidf_labels = extract_tfidf_labels(df, content_column, cluster_labels, top_n=top_n)

    lda_labels = {}
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue
        cluster_texts = df[cluster_labels == cluster_id][content_column].tolist()
        if cluster_texts:
            topics = lda_topic_modeling(
                cluster_texts, n_topics=lda_n_topics, n_words=lda_n_words
            )
            lda_labels[cluster_id] = topics
        else:
            lda_labels[cluster_id] = []

    cluster_label_map, cluster_primary_topics, cluster_related_topics = label_clusters_hybrid(
        df, content_column, summary_column, cluster_labels, embeddings, tfidf_labels, lda_labels, vague_threshold=vague_threshold
    )

    df['cluster_label'] = [
        cluster_label_map.get(cid, "Noise/Other") if cid != -1 else "Noise/Other"
        for cid in cluster_labels
    ]
    df['lda_topics'] = [
        ", ".join(lda_labels.get(cid, [])) if cid != -1 else "" for cid in cluster_labels
    ]

    detected_topics = {
        label: {
            "size": int((df['cluster_label'] == label).sum())
        }
        for label in set(df['cluster_label']) if label != "Noise/Other"
    }

    return {
        "dataframe": df,
        "detected_topics": detected_topics,
        "number_of_clusters": len(detected_topics),
        "cluster_primary_topics": cluster_primary_topics,
        "cluster_related_topics": cluster_related_topics
    }