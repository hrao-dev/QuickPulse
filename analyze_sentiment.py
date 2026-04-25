# analyze_sentiment.py
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_classifier():
    # ~67MB vs 1.63GB — same job, 24x lighter
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512,
    )

# Maps model labels to readable names
LABEL_MAP = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}

def analyze_summary(summary):
    try:
        if not summary or not summary.strip():
            return "No input provided.", 0.0
        classifier = load_classifier()
        result = classifier(summary)[0]
        label = LABEL_MAP.get(result['label'].lower(), result['label'].capitalize())
        return label, round(result['score'], 4)
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}", 0.0