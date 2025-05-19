# 📰 Quick Pulse 

#### Get a multi-perspective summary of today's news — powered by Gen AI.

---

## Overview

Quick Pulse combines advanced summarization models with sentiment scoring and topic modeling to provide a real-time snapshot of media narratives. It enables users to track evolving themes and emotional tone across diverse news sources all in one interactive interface powered by Gradio.

---

## ✨ Features

| Capability                    | Details                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| **Automated news harvesting** | Pulls the latest articles via **NewsAPI** (or Google News RSS as fallback).                       |
| **Abstractive summarization** | Condenses full‑text articles with a fine‑tuned **FLAN‑T5** model.                                 |
| **Sentiment analysis**        | Classifies each summary as *Positive*, *Neutral*, or *Negative* using **DistilBERT SST‑2**.       |
| **Semantic topic clustering** | Groups related stories with **Sentence‑BERT** embeddings + **K‑Means** + silhouette optimisation. |
| **Topic labeling**            | Generates human‑readable topic names via TF‑IDF & LDA for instant context.                        |
| **Interactive Gradio UI**     | Search by topic, upload files, paste URLs, filter by sentiment, download CSV results.             |
| **CSV export**                | One‑click download of clustered articles for further analysis.                                    |
| **Stateless deployment**      | Pure Python; no database required. Works locally, on HF Spaces, or inside Docker.                 |

---

## 🖥️ Live Demo

Experience QuickPulse instantly in your browser:

**[→ Hugging Face Space](https://huggingface.co/spaces/harao-ml/QuickPulse)**

---

## 🏗️ Architecture

```
┌──────────────┐     ┌────────────┐     ┌──────────────┐
│  News Feeds  │──▶│  Extractor  │──▶│ Summarizer ✂ │
└──────────────┘     └────────────┘     └────┬─────────┘
                                             │ summary
                                             ▼
                                        ┌───────────────┐
                                        │ Sentiment 🧭 │
                                        └────┬──────────┘
                                             │
                                             ▼
                                        ┌───────────────┐
                                        │ Embeddings 🧩 │
                                        └────┬──────────┘
                                             │ vectors
                                             ▼
                                        ┌───────────────┐
                                        │ K‑Means 📊    │
                                        └────┬──────────┘
                                             │ cluster id
                                             ▼
                                        ┌───────────────┐
                                        │  Topic LDA 🗂 │
                                        └───────────────┘
```

*Implementation highlights:* `feedparser`, `requests`, `sentence-transformers`, `scikit‑learn`, `transformers`.

---

