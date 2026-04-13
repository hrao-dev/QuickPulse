# cluster_news.py

from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

TOPIC_LABELS = [
    "Artificial Intelligence",
    "Business & Finance",
    "Politics & World Affairs",
    "Science & Technology",
    "Health & Medicine",
    "Environment & Climate",
]

# 8-10 representative sentences per topic.
# These get averaged into a single prototype vector.
_TOPIC_SEEDS = {
    "Artificial Intelligence": [
        "OpenAI released a new large language model with improved reasoning.",
        "Researchers trained a neural network on multimodal data.",
        "The AI startup raised funding to develop generative models.",
        "Machine learning algorithms are being used to automate decision making.",
        "Anthropic announced Claude, a new AI assistant built on constitutional AI.",
        "Deep learning models achieved state of the art on computer vision benchmarks.",
        "GPU demand is surging as AI model training requires more compute.",
        "Robotics companies are integrating large language models into physical agents.",
    ],
    "Business & Finance": [
        "The Federal Reserve raised interest rates to combat inflation.",
        "The company reported record quarterly earnings beating analyst expectations.",
        "Venture capital funding for startups dropped sharply this quarter.",
        "The S&P 500 fell two percent amid recession fears.",
        "Merger talks between the two tech giants collapsed over antitrust concerns.",
        "Bitcoin surged to a new all-time high as institutional investors piled in.",
        "Supply chain disruptions are driving up costs for manufacturers.",
        "The IPO raised three billion dollars in its market debut.",
    ],
    "Politics & World Affairs": [
        "The president signed the legislation after it passed both chambers.",
        "NATO allies pledged additional military support to Ukraine.",
        "Tensions between China and Taiwan escalated after military exercises.",
        "The election results were disputed as opposition parties demanded a recount.",
        "Sanctions were imposed on Iran following nuclear program developments.",
        "The prime minister announced early elections amid a political crisis.",
        "United Nations peacekeepers deployed to the conflict zone.",
        "Diplomatic talks between the two nations broke down over territorial disputes.",
    ],
    "Science & Technology": [
        "NASA's James Webb telescope captured images of a distant galaxy.",
        "Researchers at CERN made a breakthrough in quantum physics.",
        "A new CRISPR technique could eliminate inherited genetic diseases.",
        "SpaceX successfully launched and landed its reusable rocket.",
        "The semiconductor shortage is easing as new chip fabs come online.",
        "Cybersecurity researchers discovered a critical vulnerability in widely used software.",
        "A clinical study found a new biomarker for early cancer detection.",
        "The quantum computer solved an optimization problem in seconds.",
    ],
    "Health & Medicine": [
        "The FDA approved a new drug for treatment-resistant depression.",
        "A clinical trial showed the vaccine was highly effective against the virus.",
        "Obesity rates continue to rise driven by ultra-processed food consumption.",
        "Researchers identified a genetic mutation linked to Alzheimer's disease.",
        "The hospital system announced a merger to reduce costs and expand access.",
        "Mental health crisis services are overwhelmed following the pandemic.",
        "A new surgical technique reduced recovery time for hip replacements.",
        "Pfizer reported positive phase three results for its cancer therapy.",
    ],
    "Environment & Climate": [
        "Global carbon emissions reached a record high despite climate pledges.",
        "Wildfires devastated millions of acres across California and Australia.",
        "The offshore wind farm will power half a million homes.",
        "Scientists warned that Arctic sea ice is melting faster than projected.",
        "Electric vehicle sales surpassed internal combustion engines in Norway.",
        "The COP summit ended with a deal to phase out fossil fuel subsidies.",
        "Plastic pollution in the ocean has reached catastrophic levels.",
        "Drought conditions across the midwest are threatening crop yields.",
    ],
}

# Load model once — all-MiniLM-L6-v2 is ~80MB, fast on CPU
_model = SentenceTransformer("all-MiniLM-L6-v2")

def _build_prototypes() -> dict[str, np.ndarray]:
    """Average seed embeddings into one prototype vector per topic."""
    prototypes = {}
    for topic, seeds in _TOPIC_SEEDS.items():
        embeddings = _model.encode(seeds, normalize_embeddings=True)
        prototypes[topic] = embeddings.mean(axis=0)
        # Re-normalize the mean vector
        norm = np.linalg.norm(prototypes[topic])
        if norm > 0:
            prototypes[topic] /= norm
    return prototypes

# Built once at import time
_PROTOTYPES = _build_prototypes()

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Dot product of two normalized vectors = cosine similarity."""
    return float(np.dot(vec_a, vec_b))

def classify_articles(articles: list[dict]) -> list[dict]:
    """
    Classify each article by cosine similarity to topic prototype embeddings.
    Falls back to 'Other' if max similarity is below 0.25.
    """
    texts = [
        f"{a.get('title', '')} {a.get('snippet', '')}".strip()
        for a in articles
    ]
    # Batch encode — much faster than one at a time
    embeddings = _model.encode(texts, normalize_embeddings=True, batch_size=32)

    for article, emb in zip(articles, embeddings):
        scores = {
            topic: _cosine_similarity(emb, proto)
            for topic, proto in _PROTOTYPES.items()
        }
        best_topic = max(scores, key=lambda t: scores[t])
        best_score = scores[best_topic]

        article["topic"] = best_topic if best_score >= 0.25 else "Other"
        article["topic_score"] = round(best_score, 4)
        article["topic_scores"] = {t: round(s, 4) for t, s in scores.items()}
    return articles

def extract_keywords(texts: list[str], top_n: int = 5) -> list[str]:
    # unchanged — TF-IDF still fine for keyword chips
    if not texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=200)
        matrix = vectorizer.fit_transform(texts)
        avg_scores = np.asarray(matrix.mean(axis=0)).flatten()
        top_indices = avg_scores.argsort()[::-1][:top_n]
        return [vectorizer.get_feature_names_out()[i] for i in top_indices]
    except ValueError:
        return []

def group_by_topic(articles: list[dict]) -> dict[str, list[dict]]:
    from collections import defaultdict
    buckets = defaultdict(list)
    for article in articles:
        topic = article.get("topic", "Other")
        if topic in TOPIC_LABELS:
            buckets[topic].append(article)
    return dict(buckets)