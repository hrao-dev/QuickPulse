# summarizer.py
from transformers import pipeline

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
    try:
        if not content.strip():
            return "No input provided."
        cleaned = clean_text(content)
        chunks  = list(split_text(cleaned))
        summary = summarizer(content, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"
