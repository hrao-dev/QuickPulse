# summarizer.py
# Summarizes article content using a fine-tuned FlanT5 model.

from transformers import pipeline

# Load summarization pipeline once at module level
summarizer = pipeline(
    "summarization",
    model="harao-ml/flant5-finetuned-summarize",
)

def split_text(text, max_tokens=400):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

def clean_text(text):
    text = ' '.join(text.split())
    text = ' '.join(word for word in text.split() if len(word) < 100)
    return text

def _safe_summarize(chunk):
    """Summarize a single chunk, dynamically capping max_length to avoid warnings."""
    word_count = len(chunk.split())
    # max_length must be less than input length for summarization;
    # clamp it to half the input length, with a floor of 20 and ceiling of 128
    max_len = max(20, min(128, word_count // 2))
    min_len = max(10, max_len // 4)
    try:
        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )
        return result[0]['summary_text']
    except Exception as e:
        return ""

def generate_summary(content):
    try:
        if not content or not content.strip():
            return "No input provided."
        cleaned_text = clean_text(content)
        chunks = list(split_text(cleaned_text))
        if not chunks:
            return "No content to summarize."
        summaries = [_safe_summarize(chunk) for chunk in chunks if chunk.strip()]
        summary = ' '.join(s for s in summaries if s).strip()
        return summary if summary else "Could not generate summary."
    except Exception as e:
        return f"Error generating summary: {str(e)}"
