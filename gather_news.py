# gather_news.py
# News Source Integration
# This script integrates with various news sources to fetch the latest articles from the specified news sources,
# extracts relevant information such as title, URL, Source, Author and Publish date, and extracts full content.

import requests
import os
from extract_news import extract_news_articles, create_dataframe, save_to_csv

def fetch_newsapi_top_headlines(min_length=100, max_articles=25):
    api_url = 'https://newsapi.org/v2/top-headlines'
    api_key = os.environ.get("api_key")
    params = {
        'apiKey': api_key,
        'language': 'en',
        'pageSize': max_articles
    }
    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        print(f"Error: Failed to fetch news from NewsAPI Top Headlines. Status code: {response.status_code}")
        return []
    articles = response.json().get("articles", [])
    if not articles:
        print("No articles found in NewsAPI Top Headlines.")
        return []
    meta_by_url = {}
    urls = []
    for article in articles:
        article_url = article.get("url", "#")
        meta = {
            "url": article_url,
            "title": article.get("title", ""),
            "source": article.get("source", {}).get("name", ""),
            "author": article.get("author", "Unknown"),
            "publishedAt": article.get("publishedAt", "Unknown"),
        }
        meta_by_url[article_url] = meta
        urls.append(article_url)
    print(f"Fetched {len(urls)} article URLs from NewsAPI Top Headlines.")
    extracted_articles = extract_news_articles(urls, min_length=min_length)
    merged_articles = []
    for art in extracted_articles:
        meta = meta_by_url.get(art.get("original_url"))
        if not meta:
            meta = {
                "title": art.get("title", "Untitled"),
                "source": "",
                "author": "Unknown",
                "publishedAt": "Unknown"
            }
        merged = {
            "url": art.get("url"),
            "title": art.get("title") if art.get("title") and art.get("title") != "Untitled" else meta["title"],
            "source": meta["source"],
            "author": meta["author"],
            "publishedAt": meta["publishedAt"],
            "text": art.get("text", ""),
        }
        merged_articles.append(merged)
    print(f"Usable articles after extraction (NewsAPI Top Headlines): {len(merged_articles)}")
    return merged_articles

def fetch_newsapi_everything(topic, min_length=100, max_articles=50):
    api_url = 'https://newsapi.org/v2/everything'
    api_key = os.environ.get("api_key")
    params = {
        'apiKey': api_key,
        'language': 'en',
        'q': topic,
        'pageSize': max_articles,
        'sortBy': 'publishedAt'
    }
    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        print(f"Error: Failed to fetch news from NewsAPI Everything. Status code: {response.status_code}")
        return []
    articles = response.json().get("articles", [])
    if not articles:
        print("No articles found in NewsAPI Everything.")
        return []
    meta_by_url = {}
    urls = []
    for article in articles:
        article_url = article.get("url", "#")
        meta = {
            "url": article_url,
            "title": article.get("title", ""),
            "source": article.get("source", {}).get("name", ""),
            "author": article.get("author", "Unknown"),
            "publishedAt": article.get("publishedAt", "Unknown"),
        }
        meta_by_url[article_url] = meta
        urls.append(article_url)
    print(f"Fetched {len(urls)} article URLs from NewsAPI Everything.")
    extracted_articles = extract_news_articles(urls, min_length=min_length)
    merged_articles = []
    for art in extracted_articles:
        meta = meta_by_url.get(art.get("original_url"))
        if not meta:
            meta = {
                "title": art.get("title", "Untitled"),
                "source": "",
                "author": "Unknown",
                "publishedAt": "Unknown"
            }
        merged = {
            "url": art.get("url"),
            "title": art.get("title") if art.get("title") and art.get("title") != "Untitled" else meta["title"],
            "source": meta["source"],
            "author": meta["author"],
            "publishedAt": meta["publishedAt"],
            "text": art.get("text", ""),
        }
        merged_articles.append(merged)
    print(f"Usable articles after extraction (NewsAPI Everything): {len(merged_articles)}")
    return merged_articles

def fetch_articles(topic=None, min_length=100, max_articles=25):
    if topic and topic.strip():
        return fetch_newsapi_everything(topic, min_length=min_length, max_articles=max_articles)
    else:
        return fetch_newsapi_top_headlines(min_length=min_length, max_articles=max_articles)
