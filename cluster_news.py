import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def generate_embeddings(df, content_column):
    """
    Generate embeddings for the content using SentenceTransformer.
    """
    print("🔢 Generating embeddings for clustering...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df[content_column].tolist(), show_progress_bar=True)
    return embeddings


def determine_optimum_clusters(embeddings, min_clusters=2, max_clusters=10):
    """
    Determine the optimum number of clusters using silhouette analysis.
    """
    print("🔍 Determining the optimum number of clusters using silhouette analysis...")
    n_samples = len(embeddings)
    if n_samples < 2:
        raise ValueError("Not enough samples to perform clustering. At least 2 samples are required.")

    # Adjust max_clusters to ensure it does not exceed n_samples - 1
    max_clusters = min(max_clusters, n_samples - 1)

    best_num_clusters = min_clusters
    best_score = -1

    for n_clusters in range(min_clusters, max_clusters + 1):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, cluster_labels)
            print(f"Number of clusters: {n_clusters}, Silhouette Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_num_clusters = n_clusters
        except ValueError as e:
            print(f"Skipping {n_clusters} clusters due to error: {e}")

    print(f"✅ Optimum number of clusters determined: {best_num_clusters}")
    return best_num_clusters


def cluster_embeddings(embeddings, num_clusters):
    """
    Perform KMeans clustering on the embeddings.
    """
    print(f"📊 Clustering articles into {num_clusters} clusters using KMeans...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans.labels_, kmeans


def extract_tfidf_labels(df, content_column, cluster_labels):
    """
    Extract top TF-IDF keywords for each cluster.
    """
    print("🔠 Extracting TF-IDF-based keywords for cluster labels...")
    grouped = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        grouped[label].append(df.iloc[idx][content_column])

    tfidf_labels = {}
    for cluster_id, texts in grouped.items():
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50)
        tfidf_matrix = vectorizer.fit_transform(texts)
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        top_indices = np.argsort(avg_tfidf)[::-1][:3]
        top_terms = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        tfidf_labels[cluster_id] = ", ".join(top_terms)

    return tfidf_labels

def apply_topic_modeling(df, content_column, cluster_labels, num_topics=2):
    """
    Apply topic modeling (LDA) within each cluster to refine and describe topics.
    """
    print("🔍 Applying topic modeling within each cluster...")
    grouped = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        grouped[label].append(df.iloc[idx][content_column])

    topic_labels = {}
    for cluster_id, texts in grouped.items():
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(texts)

        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(tfidf_matrix)

        # Extract top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[:-4:-1]
            topics.append(", ".join([feature_names[i] for i in top_indices]))
        topic_labels[cluster_id] = " | ".join(topics)

    return topic_labels


def filter_similar_topics(topic_keywords_list, threshold=0.75):
    """
    Filter out similar topics based on cosine similarity of their embeddings.
    """
    print("🔄 Filtering similar topics...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    topic_sentences = [", ".join(kw) for kw in topic_keywords_list]
    embeddings = model.encode(topic_sentences)
    unique_indices = []
    for i, emb in enumerate(embeddings):
        if all(cosine_similarity([emb], [embeddings[j]])[0][0] < threshold for j in unique_indices):
            unique_indices.append(i)
    return [topic_keywords_list[i] for i in unique_indices]


def get_representative_summaries(df, summary_column, embeddings, cluster_labels, kmeans):
    """
    Get the most representative summary for each cluster based on proximity to the cluster centroid.
    """
    print("🔄 Refining cluster labels using representative summaries...")
    representatives = {}
    for i in range(kmeans.n_clusters):
        indices = [j for j, label in enumerate(cluster_labels) if label == i]
        if not indices:
            continue
        cluster_embeddings = embeddings[indices]
        centroid = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = indices[np.argmin(distances)]
        representatives[i] = df.iloc[closest_idx][summary_column]

    return representatives


def cluster_and_label_articles(df, content_column="content", summary_column="summary", min_clusters=2, max_clusters=10, max_topics=3):
    """
    Cluster articles using SentenceTransformer embeddings and label clusters using TF-IDF and Topic Modeling.
    Display detected topics for each cluster with Primary focus and Related topics.
    """
    if df.empty:
        print("No articles to cluster.")
        return None

    # Step 1: Generate embeddings
    embeddings = generate_embeddings(df, content_column)

    # Step 2: Determine the optimum number of clusters
    num_clusters = determine_optimum_clusters(embeddings, min_clusters, max_clusters)

    # Step 3: Perform clustering
    cluster_labels, kmeans = cluster_embeddings(embeddings, num_clusters)
    df['cluster_label'] = cluster_labels

    # Step 4: Extract TF-IDF matrix
    print("🔠 Extracting TF-IDF matrix for clusters...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df[content_column].tolist())
    feature_names = vectorizer.get_feature_names_out()

    # Step 5: Process each cluster
    print("🔍 Processing clusters for TF-IDF and topic modeling...")
    grouped = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        grouped[label].append(idx)

    refined_labels = [""] * num_clusters  # Initialize refined_labels with empty strings
    detected_topics = {}
    for cluster_id, indices in grouped.items():
        cluster_texts = tfidf_matrix[indices]

        # Extract TF-IDF keywords
        avg_tfidf = cluster_texts.mean(axis=0).A1
        top_indices = np.argsort(avg_tfidf)[::-1][:3]
        tfidf_keywords = [feature_names[i] for i in top_indices]

        # Generate a cluster label using the top TF-IDF keywords
        cluster_label_tfidf = ", ".join(tfidf_keywords)

        # Apply topic modeling
        lda = LatentDirichletAllocation(n_components=min(max_topics, len(indices)), random_state=42)
        lda.fit(cluster_texts)
        topics = []
        topic_weights = []
        for topic_idx, topic in enumerate(lda.components_):
            top_topic_indices = topic.argsort()[:-4:-1]
            topics.append(", ".join([feature_names[i] for i in top_topic_indices]))
            topic_weights.append(topic.sum())  # Sum of weights for ranking

        # Rank topics by importance
        ranked_topics = [x for _, x in sorted(zip(topic_weights, topics), reverse=True)]

        # Generate Primary focus and Related topics
        primary_focus = ranked_topics[0] if ranked_topics else "N/A"
        related_topics = ranked_topics[1:] if len(ranked_topics) > 1 else []

        # Store detected topics for user display
        detected_topics[cluster_label_tfidf] = {
            "primary_focus": primary_focus,
            "related_topics": related_topics,
        }

        # Assign the TF-IDF keywords as the cluster label
        refined_labels[cluster_id] = cluster_label_tfidf

    # Assign refined labels to clusters
    df['cluster_label'] = [refined_labels[label] for label in cluster_labels]

    print("✅ Clustering and labeling complete!")
    return {
        "dataframe": df,
        "detected_topics": detected_topics,
        "number_of_clusters": num_clusters,
    }