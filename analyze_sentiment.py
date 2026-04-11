# analyze_sentiment.py
# Zero-shot sentiment classification via facebook/bart-large-mnli.
# Classifier is loaded lazily on first call — avoids blocking app startup.

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _classifier


def analyze_summary(summary: str):
    """
    Classify sentiment of a summary string.
    Returns (sentiment: str, score: float).
    """
    try:
        if not summary or not summary.strip():
            return "Neutral", 0.0

        classifier = _get_classifier()
        candidate_labels = ["positive", "neutral", "negative"]
        result = classifier(summary, candidate_labels)
        sentiment = result["labels"][0].capitalize()
        score = float(result["scores"][0])
        return sentiment, score

    except Exception as e:
        return "Neutral", 0.0
