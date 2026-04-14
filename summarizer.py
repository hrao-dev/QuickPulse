# summarizer.py
# Summarizes article content using a fine-tuned FlanT5 model.
# Falls back to truncating the input if the text is already short.

from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="harao-ml/flant5-finetuned-summarize",
)

def _word_count(text):
    return len(text.split())

def clean_text(text):
    text = ' '.join(text.split())
    text = ' '.join(word for word in text.split() if len(word) < 100)
    return text

def split_text(text, max_tokens=400):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

def _safe_summarize(chunk):
    wc = _word_count(chunk)
    if wc < 20:
        # Too short to summarize — just return as-is
        return chunk
    # max_length must be strictly less than input length
    max_len = max(20, min(128, wc // 2))
    min_len = max(5, max_len // 4)
    try:
        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )
        return result[0]['summary_text']
    except Exception:
        return chunk  # fallback: return original chunk

def generate_summary(content):
    try:
        if not content or not content.strip():
            return "No content available."
        cleaned = clean_text(content)
        chunks = [c for c in split_text(cleaned) if c.strip()]
        if not chunks:
            return "No content to summarize."
        parts = [_safe_summarize(chunk) for chunk in chunks]
        summary = ' '.join(p for p in parts if p).strip()
        return summary if summary else cleaned[:300]
    except Exception as e:
        return content[:300] if content else "Error generating summary."
