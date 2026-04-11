# pipeline.py
# Orchestrates the full QuickPulse data pipeline:
#   gather → extract (parallel) → summarize → sentiment → cluster → write cache
#
# This module is called by:
#   - the APScheduler background job (auto-refresh every 30 min)
#   - the Streamlit "Refresh now" button (on-demand)
#
# The Streamlit UI never calls this directly on the render path —
# it only reads from cache. This is the key architectural fix.

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import analyze_sentiment
import cluster_news
import extract_news
import gather_news
import summarizer

CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "articles.json"
CACHE_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _deduplicate(articles: list) -> list:
    seen_urls: set = set()
    seen_title_source: set = set()
    seen_title_summary: set = set()
    deduped = []
    for art in articles:
        url = art.get("url")
        title = art.get("title", "").strip().lower()
        source = art.get("source", "").strip().lower()
        summary = art.get("summary", "").strip().lower()
        k_ts = (title, source)
        k_tsum = (title, summary)
        if url and url in seen_urls:
            continue
        if k_ts in seen_title_source:
            continue
        if k_tsum in seen_title_summary:
            continue
        deduped.append(art)
        if url:
            seen_urls.add(url)
        seen_title_source.add(k_ts)
        seen_title_summary.add(k_tsum)
    return deduped


def _enrich_articles(raw_articles: list) -> list:
    """Add summary + sentiment to each article dict."""
    enriched = []
    for art in raw_articles:
        content = art.get("text") or art.get("content", "")
        if not content:
            continue
        summary = summarizer.generate_summary(content)
        sentiment, score = analyze_sentiment.analyze_summary(summary)
        enriched.append({
            "title":       art.get("title", "No title"),
            "url":         art.get("url", ""),
            "source":      art.get("source", "Unknown"),
            "author":      art.get("author", "Unknown"),
            "publishedAt": art.get("publishedAt", "Unknown"),
            "content":     content,
            "summary":     summary,
            "sentiment":   sentiment,
            "score":       score,
        })
    return enriched


def _cluster(enriched: list) -> dict | None:
    """Run the HDBSCAN + LDA clustering pipeline. Returns result dict or None."""
    if not enriched:
        return None
    df = pd.DataFrame(enriched)
    return cluster_news.cluster_and_label_articles(
        df,
        content_column="content",
        summary_column="summary",
    )


def _result_to_serialisable(result: dict) -> dict:
    """
    Convert the cluster_and_label_articles result dict into something
    json.dumps can handle (DataFrames → list of dicts, numpy ints → int).
    """
    df = result["dataframe"]

    # numpy int64 keys in topic dicts need converting
    def _fix_keys(d):
        return {str(k): v for k, v in d.items()}

    return {
        "articles":              df.to_dict(orient="records"),
        "detected_topics":       result.get("detected_topics", {}),
        "number_of_clusters":    int(result.get("number_of_clusters", 0)),
        "cluster_primary_topics": _fix_keys(result.get("cluster_primary_topics", {})),
        "cluster_related_topics": _fix_keys(result.get("cluster_related_topics", {})),
    }


# ── public API ───────────────────────────────────────────────────────────────

def run(topic: str | None = None, urls: list | None = None) -> dict:
    """
    Run the full pipeline for a topic search, URL list, or top headlines.
    Returns the serialisable result dict (also written to CACHE_FILE).
    """
    logger.info(f"Pipeline starting — topic={topic!r}, urls={bool(urls)}")

    # 1. Gather article metadata
    if urls:
        raw_articles = extract_news.extract_news_articles(urls)
    elif topic and topic.strip():
        raw_articles = gather_news.fetch_newsapi_everything(topic)
    else:
        raw_articles = gather_news.fetch_newsapi_top_headlines()

    if not raw_articles:
        logger.warning("Pipeline: no articles fetched.")
        return {}

    raw_articles = sorted(
        raw_articles, key=lambda x: x.get("publishedAt", ""), reverse=True
    )

    # 2. Enrich (summarize + sentiment)
    enriched = _enrich_articles(raw_articles)

    # 3. Deduplicate
    enriched = _deduplicate(enriched)
    if not enriched:
        logger.warning("Pipeline: all articles filtered by dedup.")
        return {}

    # 4. Cluster
    result = _cluster(enriched)
    if result is None:
        logger.warning("Pipeline: clustering returned None.")
        return {}

    # 5. Serialise + write cache
    payload = _result_to_serialisable(result)
    payload["meta"] = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "topic":        topic or "top_headlines",
        "article_count": len(enriched),
    }

    try:
        CACHE_FILE.write_text(json.dumps(payload, default=str), encoding="utf-8")
        logger.info(
            f"Pipeline complete — {len(enriched)} articles, "
            f"{payload['number_of_clusters']} clusters. Cache written."
        )
    except Exception as e:
        logger.error(f"Failed to write cache: {e}")

    return payload


def load_cache() -> dict:
    """
    Read the last-written cache file.
    Returns an empty dict (never raises) so the UI always has something to render.
    """
    try:
        if CACHE_FILE.exists():
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to read cache: {e}")
    return {}
