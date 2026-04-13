---
title: QuickPulse
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# QuickPulse

Today's news, distilled by topic. Multi-source headlines clustered into 6 topic buckets — with synthesised briefings, sentiment signals, and named entity extraction powered by the Groq API.

Part of the [harao-ml](https://huggingface.co/harao-ml) NLP portfolio alongside [SumUp](https://huggingface.co/spaces/harao-ml/SumUp) and [DocQuest](https://huggingface.co/spaces/harao-ml/DocQuest).

## What it does

QuickPulse fetches up to 50 live headlines from NewsAPI, classifies them into 6 topic buckets using zero-shot classification, then makes one Groq LLM call per topic to produce:

- A 3-sentence synthesised briefing (multi-document, not per-article)
- A topic-level sentiment signal (Mostly Positive / Mixed / Mostly Negative)
- Up to 5 named entities (companies, people, countries)
- TF-IDF keyword chips
- Links to the top 3 source articles

Results are cached for 60 minutes so subsequent loads are instant.

## Architecture

```
NewsAPI (headlines + snippets)
        ↓
  gather_news.py
  fetch 50 articles, title + snippet only — no full-text scraping
        ↓
  cluster_news.py
  zero-shot classification → 6 fixed topic buckets
  TF-IDF keyword extraction per bucket
        ↓
  briefing.py
  one Groq API call per topic (6 calls total)
  returns: briefing · sentiment · entities · top story
        ↓
  cache.json  (60-min TTL)
        ↓
  app.py  (Streamlit)
  2-column briefing cards · sentiment donut · volume bar chart
```

## Topics

| | Topic |
|---|---|
| 🤖 | Artificial Intelligence |
| 📈 | Business & Finance |
| 🌍 | Politics & World Affairs |
| 🔬 | Science & Technology |
| 🏥 | Health & Medicine |
| 🌱 | Environment & Climate |

## How this fits the portfolio

| App | NLP capability |
|---|---|
| [SumUp](https://huggingface.co/spaces/harao-ml/SumUp) | Single-document abstractive summarisation |
| [DocQuest](https://huggingface.co/spaces/harao-ml/DocQuest) | Retrieval-augmented document Q&A |
| **QuickPulse** | **Multi-document synthesis + topic clustering + sentiment** |

## Files

| File | Status | Notes |
|---|---|---|
| `app.py` | Rewritten | Gradio → Streamlit; briefing card UI |
| `briefing.py` | New | Groq synthesis, entity extraction, 60-min cache |
| `gather_news.py` | Updated | Removed full-article scraping; uses NewsAPI snippets |
| `cluster_news.py` | Refactored | HDBSCAN/UMAP → zero-shot classification into fixed buckets |
| `extract_news.py` | Stripped | Removed newspaper3k scraping and CSV export |
| `analyze_sentiment.py` | Stub | Sentiment moved into `briefing.py` Groq call |
| `summarizer.py` | Stub | Per-article FlanT5 replaced by topic-level Groq synthesis |
| `requirements.txt` | Updated | Removed torch-heavy deps; added `groq`, `streamlit` |

## Environment variables

| Name | Description |
|---|---|
| `api_key` | [NewsAPI.org](https://newsapi.org) API key |
| `GROQ_API_KEY` | [Groq](https://console.groq.com) API key (free tier) |

Set both as secrets in your HF Space settings.

## Local development

```bash
pip install -r requirements.txt
export api_key=YOUR_NEWSAPI_KEY
export GROQ_API_KEY=YOUR_GROQ_KEY
streamlit run app.py
```

## Deployment on HF Spaces

1. In your Space settings, confirm SDK is set to **Streamlit**.
2. Add `api_key` and `GROQ_API_KEY` as Space secrets.
3. Push — HF will build automatically.