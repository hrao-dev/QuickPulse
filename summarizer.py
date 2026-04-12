# summarizer.py
# REPLACED — per-article summarization via local FlanT5 is removed.
# Multi-document topic synthesis is now done by briefing.py via the Groq API.
# This stub is kept so any legacy import doesn't raise an ImportError.

def generate_summary(content: str) -> str:
    """
    DEPRECATED — use briefing.generate_briefing() instead.
    Returns the first 300 chars of content as a fallback.
    """
    return content[:300].strip() + "..." if len(content) > 300 else content.strip()
