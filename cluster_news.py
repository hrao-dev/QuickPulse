# cluster_news.py
# Assigns each article to one of 6 fixed topic buckets using zero-shot
# classification (facebook/bart-large-mnli).
#
# KEY CHANGE from original: replaced HDBSCAN + UMAP + SentenceTransformer with
# a lightweight zero-shot classifier against fixed labels. This avoids loading
# 3 heavy models on CPU and removes the expensive embedding + dimensionality
# reduction step. The result is deterministic topic buckets instead of
# data-driven clusters, which is exactly what the briefing format needs.
#
# The original TF-IDF keyword extraction is kept — it runs fast on short text
# and produces the "key terms" chips shown on each card.

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# Load once at module level so repeated calls don't re-initialise the model.
_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

TOPIC_LABELS = [
    "Artificial Intelligence",
    "Business & Finance",
    "Politics & World Affairs",
    "Science & Technology",
    "Health & Medicine",
    "Environment & Climate",
]

# Articles whose top score falls below this threshold go into "Other".
CONFIDENCE_THRESHOLD = 0.25


def classify_articles(articles: list[dict]) -> list[dict]:
    """
    Add a 'topic' key to each article dict by running zero-shot classification
    on 'title + snippet'.
    Returns the same list with 'topic' and 'topic_score' fields populated.
    """
    if not articles:
        return articles

    texts = [
        f"{a.get('title', '')}. {a.get('snippet', '')}".strip()
        for a in articles
    ]

    print(f"Classifying {len(texts)} articles into topic buckets...")
    results = _classifier(texts, TOPIC_LABELS, multi_label=False)

    for article, result in zip(articles, results):
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        article["topic"] = top_label if top_score >= CONFIDENCE_THRESHOLD else "Other"
        article["topic_score"] = round(top_score, 3)

    return articles


def extract_keywords(texts: list[str], top_n: int = 5) -> list[str]:
    """
    Return the top-n TF-IDF unigrams/bigrams for a list of short texts.
    Used to generate the keyword chips shown on each briefing card.
    """
    if not texts:
        return []

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=200,
        )
        matrix = vectorizer.fit_transform(texts)
        avg_scores = np.asarray(matrix.mean(axis=0)).flatten()
        top_indices = avg_scores.argsort()[::-1][:top_n]
        return [vectorizer.get_feature_names_out()[i] for i in top_indices]
    except ValueError:
        # Too few terms (e.g. single very short article)
        return []


def group_by_topic(articles: list[dict]) -> dict[str, list[dict]]:
    """
    Return a dict mapping topic label → list of article dicts.
    Only includes the 6 canonical topic labels (no 'Other').
    Topics with zero articles are omitted.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        topic = article.get("topic", "Other")
        if topic in TOPIC_LABELS:
            buckets[topic].append(article)
    return dict(buckets)


def build_topic_dataframe(articles: list[dict]) -> pd.DataFrame:
    """Convenience wrapper — returns articles as a DataFrame with topic column."""
    return pd.DataFrame(articles)
