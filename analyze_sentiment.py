# analyze_sentiment.py
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def analyze_summary(summary):
    try:
        if not summary.strip():
            return "No input provided.", 0.0
        candidate_labels = ["positive", "neutral", "negative"]
        result    = classifier(summary, candidate_labels)
        sentiment = result['labels'][0].capitalize()
        score     = float(result['scores'][0])
        return sentiment, score
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}", 0.0
