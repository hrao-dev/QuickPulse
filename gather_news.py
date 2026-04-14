# gather_news.py
# Fetches news from NewsAPI. Uses the API-provided title, description, and content
# directly — no full-article scraping — so it works reliably on HF Spaces.

import re
import requests
import os

def _clean_newsapi_content(text):
    """Strip the '[+N chars]' truncation marker NewsAPI appends."""
    if not text:
        return ""
    return re.sub(r'\s*\[\+\d+ chars\]\s*$', '', text).strip()

def _build_article_text(article):
    """
    Combine title + description + content into the richest possible text blob.
    NewsAPI truncates 'content' at ~200 chars but description is usually a full sentence.
    """
    parts = []
    title = (article.get("title") or "").strip()
    description = _clean_newsapi_content(article.get("description") or "")
    content = _clean_newsapi_content(article.get("content") or "")

    if title:
        parts.append(title)
    if description and description != title:
        parts.append(description)
    if content and content not in description:
        parts.append(content)

    return " ".join(parts)

def _normalize(article):
    """Convert a raw NewsAPI article dict into the internal format."""
    return {
        "url": article.get("url", ""),
        "title": (article.get("title") or "No title").strip(),
        "source": (article.get("source") or {}).get("name", "Unknown"),
        "author": article.get("author") or "Unknown",
        "publishedAt": article.get("publishedAt") or "Unknown",
        "text": _build_article_text(article),
    }

def fetch_newsapi_top_headlines(min_length=40, max_articles=25):
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
        response = requests.get("https://newsapi.org/v2/top-headlines", params=params, timeout=15)
    except Exception as e:
        print(f"Error: NewsAPI request failed: {e}")
        return []

    if response.status_code != 200:
        print(f"Error: NewsAPI Top Headlines returned status {response.status_code}: {response.text[:200]}")
        return []

    articles = response.json().get("articles", [])
    if not articles:
        print("No articles found in NewsAPI Top Headlines.")
        return []

    results = []
    for art in articles:
        normalized = _normalize(art)
        if len(normalized["text"]) >= min_length:
            results.append(normalized)

    print(f"NewsAPI Top Headlines: {len(articles)} fetched, {len(results)} usable.")
    return results

def fetch_newsapi_everything(topic, min_length=40, max_articles=50):
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
        response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
    except Exception as e:
        print(f"Error: NewsAPI request failed: {e}")
        return []

    if response.status_code != 200:
        print(f"Error: NewsAPI Everything returned status {response.status_code}: {response.text[:200]}")
        return []

    articles = response.json().get("articles", [])
    if not articles:
        print(f"No articles found for topic: {topic}")
        return []

    results = []
    for art in articles:
        normalized = _normalize(art)
        if len(normalized["text"]) >= min_length:
            results.append(normalized)

    print(f"NewsAPI Everything ({topic}): {len(articles)} fetched, {len(results)} usable.")
    return results

def fetch_articles(topic=None, min_length=40, max_articles=25):
    if topic and topic.strip():
        return fetch_newsapi_everything(topic, min_length=min_length, max_articles=max_articles)
    return fetch_newsapi_top_headlines(min_length=min_length, max_articles=max_articles)
