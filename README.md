# 📰 Quick Pulse 

#### Get a multi-perspective summary of today's news — powered by Gen AI.

---

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

## 📚 Use Cases
- #### Market Intelligence & Trend Tracking
Stay on top of fast-moving trends across tech, finance, and science by summarizing Reddit and Hacker News discussions into digestible insights.

- #### Research and Academic Monitoring
Aggregate public sentiment and community discussion around emerging technologies, scientific papers, or open-source tools.

- #### Product Discovery & Competitive Analysis
Identify popular tools, frameworks, and user pain points discussed in real-world tech communities.

- #### Developer News Feeds
Build a personalized, summarized feed of what the developer community is talking about—without endless scrolling.

- #### Internal Dashboards & Reporting
Integrate into internal tools for weekly digests, sentiment summaries, or team briefings on community conversations.

- #### Startup Scouting & Ecosystem Analysis
Use semantic clustering to spot early-stage products or startups gaining traction in niche discussions.
