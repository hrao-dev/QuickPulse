# summarizer.py
# This script summarizes the content of each article of the specified topic using the Hugging Face Transformers library.

from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline(
    "summarization",
    model="harao-ml/flant5-finetuned-summarize",
    max_new_tokens=128,
)

# Function to split text into smaller chunks
def split_text(text, max_tokens=400):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

# Function to clean text
def clean_text(text):
    text = ' '.join(text.split())
    text = ' '.join(word for word in text.split() if len(word) < 100)
    return text

def generate_summary(content):
    try:
        if not content or not content.strip():
            return "No input provided."
        cleaned_text = clean_text(content)
        chunks = list(split_text(cleaned_text))
        if not chunks:
            return "No content to summarize."
        summaries = []
        for chunk in chunks:
            if chunk.strip():
                result = summarizer(chunk, do_sample=False, truncation=True)
                summaries.append(result[0]['summary_text'])
        summary = ' '.join(summaries)
        return summary if summary.strip() else "Could not generate summary."
    except Exception as e:
        return f"Error generating summary: {str(e)}"
