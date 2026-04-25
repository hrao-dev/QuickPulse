# divergence.py 

import numpy as np
from typing import Optional

# Thresholds — tunable
_CONSENSUS_THRESHOLD = 0.35
_CONTESTED_THRESHOLD = 0.60
_MIN_ARTICLES = 3  # need at least 3 articles to measure divergence meaningfully


def compute_divergence(
    articles: list[dict],
    embedding_key: str = "_embedding",
) -> dict:
    """
    Given a list of articles (already classified + embedded), compute how
    much the sources diverge in their coverage angle.

    Articles must have an '_embedding' key (np.ndarray) added by classify_articles().
    If embeddings are missing, falls back to a neutral result.

    Returns:
    {
        "score": float,          # 0.0 (full consensus) → 1.0 (full divergence)
        "label": str,            # "Consensus" | "Mixed" | "Contested"
        "mean_similarity": float,
        "variance": float,
        "n_articles": int,
        "outlier_titles": list[str],  # titles most divergent from the group center
    }
    """
    embeddings = [
        a[embedding_key] for a in articles
        if embedding_key in a and a[embedding_key] is not None
    ]

    if len(embeddings) < _MIN_ARTICLES:
        return _neutral_result(len(articles))

    emb_matrix = np.array(embeddings)  # shape: (N, 384)

    # Pairwise cosine similarity — since vectors are already L2-normalized,
    # this is just the dot product matrix
    sim_matrix = emb_matrix @ emb_matrix.T  # shape: (N, N)

    # Extract upper triangle (exclude diagonal self-similarity = 1.0)
    n = len(embeddings)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]

    mean_sim = float(np.mean(pairwise_sims))
    variance = float(np.var(pairwise_sims))

    # Divergence score: low mean similarity + high variance = contested
    # Weight variance more heavily — two camps is worse than gradual spread
    score = (1.0 - mean_sim) * 0.6 + variance * 0.4
    score = float(np.clip(score, 0.0, 1.0))

    if score < _CONSENSUS_THRESHOLD:
        label = "Consensus"
    elif score < _CONTESTED_THRESHOLD:
        label = "Mixed"
    else:
        label = "Contested"

    # Find outlier articles: those with lowest mean similarity to all others
    mean_sim_per_article = sim_matrix.mean(axis=1)
    outlier_indices = np.argsort(mean_sim_per_article)[:2]  # 2 most divergent
    outlier_titles = [articles[i].get("title", "") for i in outlier_indices]

    return {
        "score": round(score, 4),
        "label": label,
        "mean_similarity": round(mean_sim, 4),
        "variance": round(variance, 4),
        "n_articles": len(embeddings),
        "outlier_titles": outlier_titles,
    }


def _neutral_result(n: int) -> dict:
    return {
        "score": 0.0,
        "label": "Insufficient data",
        "mean_similarity": 1.0,
        "variance": 0.0,
        "n_articles": n,
        "outlier_titles": [],
    }


def annotate_topic_buckets(
    topic_buckets: dict[str, list[dict]],
) -> dict[str, dict]:
    """
    Run divergence analysis on every topic bucket.
    Returns a dict of topic → divergence result.
    """
    return {
        topic: compute_divergence(articles)
        for topic, articles in topic_buckets.items()
    }