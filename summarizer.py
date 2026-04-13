# summarizer.py
# Summarizes article content using a fine-tuned flan-t5 model.

from transformers import pipeline

# Load summarization pipeline once at module level
summarizer = pipeline("summarization", model="harao-ml/flant5-finetuned-summarize")

def split_text(text, max_tokens=512):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

def clean_text(text):
    text = ' '.join(text.split())
    text = ' '.join(word for word in text.split() if len(word) < 100)
    return text

def generate_summary(content):
    """
    Summarize content by splitting into chunks, summarizing each, then
    joining results. Falls back gracefully on errors.
    """
    try:
        if not content or not content.strip():
            return "No input provided."

        cleaned_text = clean_text(content)
        chunks = list(split_text(cleaned_text))

        # BUG FIX 1: the original code computed cons_summary from chunks but then
        # immediately discarded it by re-running summarizer on the full raw text.
        # For long articles this second call exceeds the model token limit and fails.
        # Fix: only use chunk-based summarization and join the results.
        #
        # BUG FIX 2: added a minimum word guard so the model is never called on
        # chunks that are too short (causes errors in the flan-t5 pipeline).
        MIN_WORDS = 10
        chunk_summaries = []
        for chunk in chunks:
            if chunk.strip() and len(chunk.split()) >= MIN_WORDS:
                result = summarizer(
                    chunk,
                    do_sample=False,
                    min_length=10,
                    max_length=130,
                    truncation=True,
                )
                chunk_summaries.append(result[0]['summary_text'])

        if not chunk_summaries:
            return "Content too short to summarize."

        return ' '.join(chunk_summaries)

    except Exception as e:
        return f"Error generating summary: {str(e)}"
