# gather_news.py


# News Source Integration
# This script integrates with various news sources to fetch the latest articles from the specified news sources, extracts relevant information such as title, URL,Source,Author and Publish date.

import config
import requests
import feedparser

def fetch_articles_newsapi(topic):
    """
    Fetch articles from NewsAPI based on the provided topic.
    """
    url = 'https://newsapi.org/v2/everything'
    params = {
        'apiKey': config.api_key,
        'language': 'en',
        'q': topic,
        'pageSize': 20
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return f"Error: Failed to fetch news. Status code: {response.status_code}"

        articles = response.json().get("articles", [])
        if not articles:
            return "No articles found."

        # Extract relevant information from each article
        extracted_articles = []
        for article in articles:
            extracted_articles.append({
                "title": article.get("title", "No title"),
                "url": article.get("url", "#"),
                "source": article.get("source", {}).get("name", "Unknown"),
                "author": article.get("author", "Unknown"),
                "publishedAt": article.get("publishedAt", "Unknown")
            })

        return extracted_articles
    except Exception as e:
        return f"Error fetching news: {str(e)}"
    
def fetch_articles_google(topic):
    """
    Fetch articles from Google News RSS feed based on the provided topic.
    """
    rss_url = f'https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en'
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            return "No articles found."

        # Extract relevant information from each article
        extracted_articles = []
        for entry in feed.entries[:20]:  # Limit to top 20 articles
            extracted_articles.append({
                "title": entry.title,
                "url": entry.link,
                "source": entry.source.title if hasattr(entry, 'source') else "Unknown",
                "author": entry.author if hasattr(entry, 'author') else "Unknown",
                "publishedAt": entry.published if hasattr(entry, 'published') else "Unknown"
            })

        return extracted_articles
    except Exception as e:
        return f"Error fetching news: {str(e)}"
    