# cluster_news.py
# FAST topic classification — no model loading, no API calls.
#
# Uses a keyword-rule approach: for each article we score it against
# keyword lists for each topic and assign the highest-scoring bucket.
# This runs in milliseconds for 50 articles vs 5-10 min for bart-large-mnli.
#
# TF-IDF keyword extraction is kept for the chips shown on briefing cards.

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TOPIC_LABELS = [
    "Artificial Intelligence",
    "Business & Finance",
    "Politics & World Affairs",
    "Science & Technology",
    "Health & Medicine",
    "Environment & Climate",
]

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Artificial Intelligence": [
        "ai", "artificial intelligence", "machine learning", "deep learning",
        "llm", "large language model", "gpt", "chatgpt", "openai", "anthropic",
        "gemini", "claude", "copilot", "neural network", "generative ai",
        "diffusion model", "transformer", "nlp", "computer vision", "automation",
        "robot", "robotics", "algorithm", "model training", "inference",
        "nvidia", "gpu", "agi", "agent", "multimodal",
    ],
    "Business & Finance": [
        "stock", "market", "economy", "inflation", "interest rate", "fed",
        "federal reserve", "gdp", "recession", "earnings", "revenue", "profit",
        "ipo", "acquisition", "merger", "startup", "venture capital", "funding",
        "investment", "investor", "nasdaq", "s&p", "dow jones", "crypto",
        "bitcoin", "ethereum", "bank", "finance", "financial", "trade",
        "tariff", "export", "import", "supply chain", "ceo", "quarterly",
        "wall street", "hedge fund", "bond", "treasury", "dollar",
    ],
    "Politics & World Affairs": [
        "election", "president", "prime minister", "congress", "senate",
        "parliament", "government", "policy", "legislation", "bill", "law",
        "war", "conflict", "military", "nato", "un", "united nations",
        "sanctions", "diplomacy", "treaty", "vote", "campaign", "democrat",
        "republican", "political", "minister", "white house", "kremlin",
        "ukraine", "russia", "china", "israel", "gaza", "taiwan", "iran",
        "north korea", "eu", "european union", "protest", "coup", "summit",
    ],
    "Science & Technology": [
        "research", "study", "scientist", "discovery", "space", "nasa",
        "spacex", "rocket", "satellite", "quantum", "physics", "biology",
        "chemistry", "experiment", "lab", "breakthrough", "innovation",
        "semiconductor", "chip", "apple", "google", "microsoft", "meta",
        "amazon", "software", "hardware", "cybersecurity", "hack", "data",
        "cloud", "5g", "broadband", "internet", "app", "smartphone",
        "telescope", "astronomy", "genome", "crispr", "biotech",
    ],
    "Health & Medicine": [
        "health", "medicine", "medical", "hospital", "doctor", "patient",
        "treatment", "drug", "vaccine", "cancer", "disease", "virus",
        "pandemic", "fda", "clinical trial", "surgery", "diagnosis",
        "mental health", "therapy", "pharmaceutical", "pfizer", "moderna",
        "obesity", "diabetes", "alzheimer", "heart", "stroke", "covid",
        "outbreak", "epidemic", "nutrition", "fitness", "wellness",
        "healthcare", "insurance", "medicare", "medicaid",
    ],
    "Environment & Climate": [
        "climate", "climate change", "global warming", "carbon", "emissions",
        "renewable energy", "solar", "wind energy", "electric vehicle", "ev",
        "tesla", "fossil fuel", "oil", "gas", "coal", "pollution",
        "environment", "sustainability", "biodiversity", "species", "ocean",
        "wildfire", "flood", "drought", "hurricane", "deforestation",
        "paris agreement", "cop", "greenhouse", "methane", "recycling",
        "plastic", "water", "conservation", "glacier", "sea level",
    ],
}

_TOPIC_PATTERNS: dict[str, list[str]] = {
    topic: [kw.lower() for kw in kws]
    for topic, kws in _TOPIC_KEYWORDS.items()
}


def _score_text(text: str) -> dict[str, int]:
    text_lower = text.lower()
    return {
        topic: sum(1 for kw in keywords if kw in text_lower)
        for topic, keywords in _TOPIC_PATTERNS.items()
    }


def classify_articles(articles: list[dict]) -> list[dict]:
    """
    Assign each article a 'topic' by scoring title+snippet against keyword lists.
    Runs in <100ms for 50 articles. No model loading.
    """
    for article in articles:
        text = f"{article.get('title', '')} {article.get('snippet', '')}".strip()
        scores = _score_text(text)
        best_topic = max(scores, key=lambda t: scores[t])
        best_score = scores[best_topic]
        article["topic"] = best_topic if best_score > 0 else "Other"
        article["topic_score"] = best_score
    return articles


def extract_keywords(texts: list[str], top_n: int = 5) -> list[str]:
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
        return []


def group_by_topic(articles: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        topic = article.get("topic", "Other")
        if topic in TOPIC_LABELS:
            buckets[topic].append(article)
    return dict(buckets)


def build_topic_dataframe(articles: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(articles)
