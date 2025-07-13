# analyze_sentiment.py
# This script analyzes the sentiment of the summarized content using the Hugging Face Transformers library.

from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def analyze_summary(summary):
    """
    Analyze the sentiment of the given summary using zero-shot classification.
    Returns a tuple of (sentiment, score).
    """
    try:
        if not summary.strip():
            return "No input provided.", 0.0

        candidate_labels = ["positive", "neutral", "negative"]
        result = classifier(summary, candidate_labels)
        sentiment = result['labels'][0].capitalize()
        score = float(result['scores'][0])
        return sentiment, score
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}", 0.0