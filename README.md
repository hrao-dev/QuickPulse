# QuickPulse

Live multi-source news aggregator with HDBSCAN clustering, FLAN-T5 summarisation, and zero-shot sentiment analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  HF Docker Space                                │
│                                                 │
│  ┌─────────────┐     reads      ┌────────────┐  │
│  │ Streamlit   │ ─────────────▶ │ cache/     │  │
│  │ app.py      │                │ articles   │  │
│  └─────────────┘                │ .json      │  │
│                                 └─────┲──────┘  │
│  ┌─────────────────────────────┐      ┃ writes  │
│  │ APScheduler (every 30 min) │      ┃         │
│  │ + on-demand button trigger  │──────┛         │
│  └──────────────┬──────────────┘                │
│                 │                               │
│         pipeline.run()                          │
│         gather → extract (parallel)             │
│         → summarize → sentiment → cluster       │
└─────────────────────────────────────────────────┘
```

**Key principle:** the Streamlit render path never blocks on an API call.  
The background scheduler keeps the cache fresh; the UI reads it instantly.

---

## Files changed vs original

| File | Change |
|---|---|
| `app.py` | Full rewrite — Gradio → Streamlit, cache-first render, scheduler bootstrap |
| `summarizer.py` | Lazy model load; fixed redundant double-inference bug |
| `analyze_sentiment.py` | Lazy model load |
| `extract_news.py` | Sequential loop → `ThreadPoolExecutor` (parallel scraping) |
| `pipeline.py` | **New** — orchestration layer; writes `cache/articles.json` |
| `requirements.txt` | `gradio` → `streamlit`; added `apscheduler` |
| `Dockerfile` | **New** — HF Docker Space config, port 7860 |

**Unchanged:** `gather_news.py`, `cluster_news.py`, `input_topic.py`

---

## Deployment

1. In your HF Space settings, set **SDK → Docker**.
2. Set the secret `api_key` to your NewsAPI key.
3. Push this repo — HF will build the Docker image automatically.
4. (Optional) Add a free UptimeRobot monitor pointing at  
   `https://harao-ml-quickpulse.hf.space/` every 5 minutes to keep the Space warm.

---

## Environment variables

| Name | Description |
|---|---|
| `api_key` | NewsAPI.org API key |

---

## Local development

```bash
pip install -r requirements.txt
export api_key=YOUR_NEWSAPI_KEY
streamlit run app.py
```
