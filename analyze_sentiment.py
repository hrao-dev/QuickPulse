# analyze_sentiment.py

# This script analyzes the sentiment of the summarized content using the Hugging Face Transformers library.


from transformers import pipeline


# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model = "distilbert-base-uncased-finetuned-sst-2-english")

def analyze_summary(summary):
    """
    Analyze the sentiment of the given summary.
    Returns a tuple of (sentiment, score).
    """
    try:
        if not summary.strip():
            return "No input provided.", 0.0
        
        result = sentiment_analyzer(summary)[0]
        sentiment = result['label']
        score = result['score']
        
        return sentiment, score
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}", 0.0
# Example usage