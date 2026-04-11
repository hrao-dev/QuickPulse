# extract_news.py
# Extracts full article content from URLs using newspaper3k.
# REFACTOR: switched from sequential for-loop to ThreadPoolExecutor so
# scraping 25-50 URLs runs in parallel instead of serially.

import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from newspaper import Article


def extract_full_content(url: str, min_length: int = 100):
    """Download and parse a single article. Returns dict or None."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        title = article.title.strip() if article.title else "Untitled"
        if len(text) < min_length:
            logging.warning(f"Content too short from {url}.")
            return None
        return {"url": url, "original_url": url, "text": text, "title": title}
    except Exception as e:
        logging.error(f"Failed to extract {url}: {e}")
        return None


def extract_news_articles(urls: list, min_length: int = 100, max_workers: int = 10):
    """
    Scrape all URLs in parallel (default 10 workers).
    Returns list of article dicts that passed the min_length filter.
    """
    extracted = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_full_content, url, min_length): url
            for url in urls
        }
        for future in as_completed(futures):
            result = future.result()
            if result and result.get("text"):
                extracted.append(result)
    return extracted


def create_dataframe(articles: list) -> pd.DataFrame:
    return pd.DataFrame(articles)


def save_to_csv(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False)
