---
title: QuickPulse
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# QuickPulse

Live news, clustered by topic and scored by sentiment — instantly.

Part of the [harao-ml](https://huggingface.co/harao-ml) NLP portfolio alongside [SumUp](https://huggingface.co/spaces/harao-ml/SumUp) and [DocQuest](https://huggingface.co/spaces/harao-ml/DocQuest).

## What it does

QuickPulse fetches up to 50 live articles from NewsAPI, scrapes full content, summarizes each article with a fine-tuned FlanT5 model, runs zero-shot sentiment classification, and clusters everything into topic groups using HDBSCAN + UMAP + sentence embeddings. Results are displayed in a dark-themed Streamlit UI with interactive Plotly charts.

- Search any topic or pull today's top headlines
- Paste a list of URLs to analyze your own sources
- Filter results by sentiment (Positive / Neutral / Negative)
- Topic frequency bar chart + sentiment donut chart
- Per-article AI summaries with sentiment scores

## Architecture

```
NewsAPI  ──────────────────────────────────────────────────────┐
  gather_news.py                                               │
  fetch up to 50 article URLs                                  │
        ↓                                                      │
  extract_news.py                                              │
  newspaper3k full-text scraping                               │  or paste URLs directly
        ↓                                                      │
  summarizer.py                                                │
  FlanT5 (harao-ml/flant5-finetuned-summarize)                 │
        ↓                                                      │
  analyze_sentiment.py                                         │
  facebook/bart-large-mnli zero-shot classification            │
        ↓                                                      │
  cluster_news.py                                              │
  sentence-transformers → UMAP → HDBSCAN                       │
  TF-IDF + LDA cluster labeling                               │
        ↓                                                      │
  app.py  (Streamlit)                                          │
  sidebar controls · metric cards · Plotly charts · digest ───┘
```

## Files

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI — sidebar controls, charts, clustered digest |
| `gather_news.py` | NewsAPI integration (top headlines + topic search) |
| `extract_news.py` | Full-text article extraction via newspaper3k |
| `summarizer.py` | Per-article summarization with fine-tuned FlanT5 |
| `analyze_sentiment.py` | Zero-shot sentiment scoring with BART-MNLI |
| `cluster_news.py` | HDBSCAN clustering, UMAP reduction, TF-IDF/LDA labeling |
| `requirements.txt` | Python dependencies |

## Environment variables

| Name | Description |
| --- | --- |
| `api_key` | [NewsAPI.org](https://newsapi.org) API key |

Set this as a secret in your HF Space settings (**Settings → Variables and secrets**).

## Local development

```bash
pip install -r requirements.txt
export api_key=YOUR_NEWSAPI_KEY
streamlit run app.py
```

## Deployment on HF Spaces

1. Confirm the Space SDK is set to **Streamlit** (the frontmatter above handles this automatically).
2. Add `api_key` as a Space secret.
3. Push all files — HF will build automatically.
