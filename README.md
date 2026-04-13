---
title: QuickPulse
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# QuickPulse

Today's news, distilled by topic. Fetch live headlines by topic or URL, clustered into topic buckets with synthesised summaries, sentiment signals, and CSV export — powered by FlanT5 + HDBSCAN.

Part of the [harao-ml](https://huggingface.co/harao-ml) NLP portfolio alongside [SumUp](https://huggingface.co/spaces/harao-ml/SumUp) and [DocQuest](https://huggingface.co/spaces/harao-ml/DocQuest).

## What it does

QuickPulse fetches live articles from NewsAPI (by topic or top headlines), extracts full content, summarises each article with a fine-tuned FlanT5 model, analyses sentiment via zero-shot classification, and clusters articles using HDBSCAN + UMAP embeddings. Results are displayed grouped by topic cluster and sentiment, with Plotly charts and a CSV export.

## Architecture

```
NewsAPI (top headlines or topic search)
        ↓
  gather_news.py
  fetch articles, extract metadata
        ↓
  extract_news.py
  full-text extraction via newspaper4k
        ↓
  summarizer.py
  per-article summarisation (harao-ml/flant5-finetuned-summarize)
        ↓
  analyze_sentiment.py
  zero-shot sentiment (facebook/bart-large-mnli)
        ↓
  cluster_news.py
  sentence-transformers embeddings → UMAP → HDBSCAN → TF-IDF/LDA labels
        ↓
  app.py  (Gradio)
  clustered digest · sentiment donut · topic bar chart · CSV export
```

## Environment variables

| Name | Description |
| --- | --- |
| `api_key` | [NewsAPI.org](https://newsapi.org) API key |

Set it as a secret in your HF Space settings.

## Local development

```bash
pip install -r requirements.txt
export api_key=YOUR_NEWSAPI_KEY
python app.py
```

## Deployment on HF Spaces

1. Confirm SDK is set to **Gradio** in the Space settings (or via README metadata above).
2. Add `api_key` as a Space secret.
3. Push — HF will build automatically.
