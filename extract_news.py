# extract_news.py
# Extracts full content from user-supplied URLs using newspaper4k.
# Falls back gracefully when sites block scraping (403, timeout, proxy errors).

import logging
import pandas as pd

try:
    from newspaper import Article, Config
    _config = Config()
    _config.browser_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    _config.request_timeout = 10
    _config.fetch_images = False
    _config.memoize_articles = False
    _NEWSPAPER_AVAILABLE = True
except ImportError:
    _NEWSPAPER_AVAILABLE = False

def extract_full_content(url, min_length=40):
    if not _NEWSPAPER_AVAILABLE:
        return None
    try:
        article = Article(url, config=_config)
        article.download()
        article.parse()
        text = article.text.strip()
        title = article.title.strip() if article.title else "Untitled"
        if len(text) < min_length:
            logging.warning(f"Content too short from {url} ({len(text)} chars).")
            return None
        return {"url": url, "text": text, "title": title}
    except Exception as e:
        logging.error(f"Failed to extract content from {url}: {str(e)}")
        return None

def extract_news_articles(urls, min_length=40):
    extracted = []
    for url in urls:
        article = extract_full_content(url, min_length=min_length)
        if article and article.get("text"):
            article["original_url"] = url
            extracted.append(article)
    if not extracted:
        logging.warning(f"No articles extracted from {len(urls)} URLs (likely blocked by egress proxy or paywalls).")
    return extracted

def create_dataframe(articles):
    return pd.DataFrame(articles)

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
