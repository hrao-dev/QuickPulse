# analyze_sentiment.py
# This file is kept only for backward-compatibility if anything imports it.
# Sentiment analysis is now handled inside briefing.py as part of the Groq
# synthesis call (one call per topic returns briefing + sentiment + entities).
# The heavy facebook/bart-large-mnli zero-shot pipeline is no longer loaded
# here — it is used only in cluster_news.py for topic classification.

def analyze_summary(summary: str) -> tuple[str, float]:
    """
    DEPRECATED — sentiment is now returned by briefing.generate_briefing().
    This stub exists so any legacy import doesn't break.
    Returns ('Neutral', 0.0) as a safe default.
    """
    return "Neutral", 0.0
