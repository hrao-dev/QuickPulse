# summarizer.py
from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="harao-ml/flant5-finetuned-summarize",
        truncation=True,
        max_length=200,
        min_length=30,
    )

def split_text(text, max_tokens=400):  # 400 not 512, leaves room for special tokens
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

def clean_text(text):
    text = ' '.join(text.split())
    text = ' '.join(word for word in text.split() if len(word) < 100)
    return text

def generate_summary(content):
    try:
        if not content or not content.strip():
            return "No input provided."

        summarizer = load_summarizer()
        cleaned = clean_text(content)
        chunks = list(split_text(cleaned))

        # Summarize each chunk, then join and summarize again if needed
        chunk_summaries = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            input_len = len(chunk.split())
            max_out = max(30, min(200, input_len // 2))  # avoid max > input warning
            result = summarizer(
                chunk,
                do_sample=False,
                max_length=max_out,
                min_length=min(20, max_out - 1),
            )
            chunk_summaries.append(result[0]['summary_text'])

        combined = ' '.join(chunk_summaries)

        # If multi-chunk, do a final summarization pass on the combined summaries
        if len(chunks) > 1 and combined.strip():
            input_len = len(combined.split())
            max_out = max(30, min(200, input_len // 2))
            final = summarizer(
                combined,
                do_sample=False,
                max_length=max_out,
                min_length=min(20, max_out - 1),
                truncation=True,
            )
            return final[0]['summary_text']

        return combined if combined.strip() else "Could not generate summary."

    except Exception as e:
        return f"Error generating summary: {str(e)}"