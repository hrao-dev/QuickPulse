# summarizer.py
# Summarizes article content using a fine-tuned FLAN-T5 model.
# Model is loaded lazily on first call — keeps app startup instant.

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "summarization",
            model="harao-ml/flant5-finetuned-summarize",
        )
    return _pipeline


def split_text(text, max_tokens=512):
    """Yield word-based chunks of up to max_tokens words."""
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i : i + max_tokens])


def clean_text(text):
    """Collapse whitespace and drop suspiciously long tokens (e.g. base64 blobs)."""
    text = " ".join(text.split())
    text = " ".join(word for word in text.split() if len(word) < 100)
    return text


def generate_summary(content: str) -> str:
    """
    Return a summary string for the given article content.
    Falls back gracefully on any error so the pipeline never hard-crashes.
    """
    try:
        if not content or not content.strip():
            return "No content available."

        summarizer = _get_pipeline()
        cleaned = clean_text(content)
        chunks = [c for c in split_text(cleaned) if c.strip()]

        if not chunks:
            return "No content available."

        # BUG FIX: original code ran summarizer(chunk) per chunk AND then
        # summarizer(text) on the full text — doubling inference time — then
        # returned `summary` (the full-text result) while discarding `cons_summary`.
        # We now do one pass over the chunks only.
        parts = [
            summarizer(chunk, do_sample=False)[0]["summary_text"]
            for chunk in chunks
        ]
        return " ".join(parts)

    except Exception as e:
        return f"Summary unavailable: {e}"
