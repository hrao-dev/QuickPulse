# summarizer.py
from transformers import pipeline
import streamlit as st

FINAL_PASS_THRESHOLD = 150  # words — only re-summarize if combined exceeds this

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="harao-ml/flant5-finetuned-summarize",
        truncation=True,
    )

def split_text(text, max_tokens=400):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

def clean_text(text):
    text = ' '.join(text.split())
    text = ' '.join(word for word in text.split() if len(word) < 100)
    return text

def _safe_lengths(word_count):
    """Return (max_length, min_length) that are always valid and proportional."""
    max_out = max(10, min(200, word_count // 2))
    min_out = min(5, max_out - 1)
    return max_out, min_out

def generate_summary(content):
    try:
        if not content or not content.strip():
            return "No input provided."

        summarizer = load_summarizer()
        cleaned = clean_text(content)
        chunks = [c for c in split_text(cleaned) if c.strip()]

        if not chunks:
            return "Could not generate summary."

        chunk_summaries = []
        for chunk in chunks:
            max_out, min_out = _safe_lengths(len(chunk.split()))
            result = summarizer(
                chunk,
                do_sample=False,
                max_length=max_out,
                min_length=min_out,
            )
            chunk_summaries.append(result[0]['summary_text'])

        combined = ' '.join(chunk_summaries)

        if len(chunks) > 1 and len(combined.split()) > FINAL_PASS_THRESHOLD:
            max_out, min_out = _safe_lengths(len(combined.split()))
            final = summarizer(
                combined,
                do_sample=False,
                max_length=max_out,
                min_length=min_out,
                truncation=True,
            )
            return final[0]['summary_text']

        return combined

    except Exception as e:
        return f"Error generating summary: {str(e)}"