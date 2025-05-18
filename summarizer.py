# summarizer.py
# This script summarizes the content of each article of the specified topic using the Hugging Face Transformers library.

from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="harao-ml/flant5-finetuned-summarize")

# Load once globally

#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#tokenizer = AutoTokenizer.from_pretrained("flant5-base")
#model = AutoModelForSeq2SeqLM.from_pretrained("flant5-base")
#summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Function to split text into smaller chunks
def split_text(text, max_tokens=512):
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
        if not content.strip():
                return "No input provided."
        text = content
        cleaned_text = clean_text(text)
        chunks = list(split_text(cleaned_text))
        cons_summary = ''.join([summarizer(chunk, do_sample=False)[0]['summary_text'] for chunk in chunks if chunk.strip()]) if chunks else ''
        summary = summarizer(text, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"
