# extract_news.py
# This script extracts full content from news articles using the newspaper4k library.

import logging
import pandas as pd
from newspaper import Article

def extract_full_content(url, min_length=100):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        title = article.title.strip() if article.title else "Untitled"
        if len(text) < min_length:
            logging.warning(f"Extracted content is too short from {url}.")
            return None
        return {"url": url, "text": text, "title": title}
    except Exception as e:
        logging.error(f"Failed to extract content from {url}: {str(e)}")
        return None

def extract_news_articles(urls, min_length=100):
    extracted_articles = []
    for url in urls:
        article = extract_full_content(url, min_length=min_length)
        if article and article.get("text"):
            article["original_url"] = url
            extracted_articles.append(article)
    return extracted_articles

def create_dataframe(articles):
    return pd.DataFrame(articles)

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
