# gather_news.py
# Fetches headlines and snippets from NewsAPI.
# KEY CHANGE from original: we no longer call extract_news_articles() to scrape
# full article HTML. Instead we use the description/snippet NewsAPI already
# returns. This eliminates 25-50 sequential HTTP fetches and is the single
# biggest latency fix.

import os
import requests


def fetch_newsapi_top_headlines(max_articles: int = 50) -> list[dict]:
    """Return top headlines with title + snippet only (no full-text fetch)."""
    api_key = os.environ.get("api_key")
    if not api_key:
        print("Warning: api_key env var not set.")
        return []

    params = {
        "apiKey": api_key,
        "language": "en",
        "pageSize": max_articles,
    }
    try:
        response = requests.get(
            "https://newsapi.org/v2/top-headlines", params=params, timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching top headlines: {e}")
        return []

    articles = response.json().get("articles", [])
    return _normalize(articles)


def fetch_newsapi_everything(topic: str, max_articles: int = 50) -> list[dict]:
    """Return articles matching a topic with title + snippet only."""
    api_key = os.environ.get("api_key")
    if not api_key:
        print("Warning: api_key env var not set.")
        return []

    params = {
        "apiKey": api_key,
        "language": "en",
        "q": topic,
        "pageSize": max_articles,
        "sortBy": "publishedAt",
    }
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything", params=params, timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching topic news: {e}")
        return []

    articles = response.json().get("articles", [])
    return _normalize(articles)


def fetch_articles(topic: str | None = None, max_articles: int = 50) -> list[dict]:
    """Unified entry point. Topic → /everything, no topic → /top-headlines."""
    if topic and topic.strip():
        return fetch_newsapi_everything(topic.strip(), max_articles=max_articles)
    return fetch_newsapi_top_headlines(max_articles=max_articles)


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(raw_articles: list[dict]) -> list[dict]:
    """
    Map raw NewsAPI article dicts to the internal schema used by the rest of
    the pipeline:
        title, snippet, url, source, author, publishedAt
    """
    normalized = []
    for art in raw_articles:
        title = (art.get("title") or "").strip()
        snippet = (art.get("description") or art.get("content") or "").strip()

        # Skip placeholder entries NewsAPI sometimes returns
        if not title or title == "[Removed]":
            continue

        normalized.append(
            {
                "title": title,
                "snippet": snippet,
                "url": art.get("url", "#"),
                "source": art.get("source", {}).get("name", "Unknown"),
                "author": art.get("author") or "Unknown",
                "publishedAt": art.get("publishedAt") or "Unknown",
            }
        )

    print(f"Fetched {len(normalized)} articles from NewsAPI.")
    return normalized
