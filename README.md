# QuickPulse

> Fetch, cluster & analyze live news — sentiment-scored, topic-clustered, AI-summarized.

[![Live Demo on HF Spaces](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/harao-ml/QuickPulse)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is QuickPulse?

**QuickPulse** is a live news intelligence dashboard that automatically fetches, clusters, and analyzes real-time news so you can stay informed without the noise. Instead of reading dozens of articles, QuickPulse groups related stories into topic clusters, scores their sentiment, and delivers AI-generated summaries — all in one view.

---

## Live Demo

**[Try QuickPulse on Hugging Face Spaces](https://huggingface.co/spaces/harao-ml/QuickPulse)**

---

## Features

| Feature | Description |
|---|---|
| **Live Multi-Source News** | Pulls real-time articles from multiple news sources simultaneously |
| **Topic Clustering** | Automatically groups related articles into named clusters (e.g. `ai, trading, stock`) |
| **Sentiment Scoring** | Labels every article Positive / Neutral / Negative with a confidence score (0–1) |
| **AI Summaries** | Expandable per-article AI-generated summaries powered by BART |
| **Clustered News Digest** | Full digest view organized by topic cluster with sentiment badges |
| **Top Headlines Mode** | Quick scan of the highest-scoring articles across all clusters |
| **Sentiment Filter** | Toggle to show only Positive, Neutral, or Negative articles |
| **Interactive Charts** | Bar chart of article volume per topic cluster + sentiment donut chart |

---

## How It Works

### Dashboard Overview
The main dashboard shows a summary of all fetched articles — total article count, number of topic clusters, and an overall sentiment breakdown (% Positive, % Negative). A horizontal bar chart shows article volume per topic cluster, color-coded by sentiment. A donut chart visualizes the overall Positive / Neutral / Negative split.

### Clustered News Digest
Articles are grouped into topic clusters identified by their top keywords (e.g. `ai, engineering, dangote` or `article, guardiola, pep`). Each cluster shows:
- A **POSITIVE / NEUTRAL / NEGATIVE** badge with article count
- Individual article titles with source and relevance score
- Expandable **Summary** for each article

### Filtering
Use the sidebar filter buttons to show only Positive, Neutral, or Negative articles across all clusters. Use the topic input to focus on a specific subject area.

---

## Tech Stack

| Layer | Technology |
|---|---|
| News Data | [NewsAPI](https://newsapi.org/) + HOBSCAN |
| Semantic Embeddings | [sentence-transformers](https://www.sbert.net/) |
| Summarization | BART-MNLI via Hugging Face |
| Sentiment Analysis | FlairNLP / sentence-transformers |
| Frontend UI | [Gradio](https://gradio.app/) |
| Hosting | [Hugging Face Spaces](https://huggingface.co/spaces) |

---

## 🚀 Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/QuickPulse.git
cd QuickPulse
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```
NEWS_API_KEY=your_newsapi_key_here
```

> Get a free API key at [newsapi.org](https://newsapi.org/)

### 4. Run the app

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

---

## Deploying to Hugging Face Spaces

This app is built to run on HF Spaces with a Gradio SDK.

1. Fork this repo
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
3. Select **Gradio** as the SDK
4. Push this repo to your Space:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/QuickPulse
git push hf main
```

5. Add your `NEWS_API_KEY` as a **Secret** in the Space Settings under *Repository Secrets*

---

## Project Structure

```
QuickPulse/
├── app.py               # Main application entry point
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── assets/              # Screenshots and static assets
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
