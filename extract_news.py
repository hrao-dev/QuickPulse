# extract_news.py

# This script is designed to extract news articles from various sources, including NewsAPI and Google News RSS using the URLs saved from the gather_news.py file. 
# It includes functions for extracting  clean,full-text content from the articles, and storing the metadata into a file.


# Article Scraping & Text Extraction    

from newspaper import Article
import pandas as pd
import logging
import requests
from bs4 import BeautifulSoup


# * For each URL from NewsAPI or RSS, * Create Article(url)* Call .download(), .parse(), .text and * Optionally use .nlp() to get summary and keywords

def extract_full_content(url, min_length=300):
    """
    Extract full content and title from the given URL using newspaper3k.
    Always returns a tuple (content, title) or (None, None).
    """
    try:
        article = Article(url)
        article.download()
        article.parse()

        text = article.text.strip()
        title = article.title.strip() if article.title else "Untitled"

        # Filter out short content
        if len(text) < min_length:
            logging.warning(f"Extracted content is too short from {url}.")
            return None, None

        return text, title

    except Exception as e:
        logging.error(f"Failed to extract content from {url}: {str(e)}")
        return None, None

    
def extract_full_content_rss(url, min_length=300):
    """
    Extract full content and title from an RSS article using BeautifulSoup.
    Always returns a tuple: (text, title) or (None, None).
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logging.error(f"Error fetching URL {url}: {response.status_code}")
            return None, None

        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs]).strip()

        if len(text) < min_length:
            logging.warning(f"Extracted content is too short from {url}.")
            return None, None

        return text, title

    except Exception as e:
        logging.error(f"Error extracting content from {url}: {str(e)}")
        return None, None


# * Handle common edge cases such as * Paywalled content (skip or tag) and * Duplicate links or broken URLs 
def is_paywalled(url):
    """
    * Check if the URL is paywalled
    """
    paywall_indicators = ['paywall', 'subscription', 'premium']
    return any(indicator in url for indicator in paywall_indicators)

def is_paywalled_content(article):
    """
    * Check if the article is paywalled
    """
    if not article:
        return False
    if not article.get("text"):
        return False
    if is_paywalled(article.get("url", "")):
        return True
    return False

def is_duplicate(url, existing_urls):
    """
    * Check if the URL is a duplicate
    """
    return url in existing_urls

def is_broken(url):
    """
    * Check if the URL is broken
    """
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code != 200
    except requests.RequestException:
        return True
    
def is_valid_url(url):
    """
    * Check if the URL is valid
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def is_valid_url_content(url):
    """
    * Check if the URL is valid
    """
    if not url:
        return False
    if not is_valid_url(url):
        return False
    if is_paywalled(url):
        return False
    if is_broken(url):
        return False
    return True

# Additional functions to check if the article have empty content or blocked sites

def is_empty_content(article):
    """
    * Check if the article content is empty
    """
    if not article:
        return True
    if not article.get("text"):
        return True
    return False

def is_blocked_site(url):
    """
    * Check if the URL is from a blocked site
    """
    blocked_sites = ['example.com', 'blockedsite.com']  # Add your blocked sites here
    return any(blocked_site in url for blocked_site in blocked_sites)

def is_blocked_content(article):
    """
    * Check if the article is from a blocked site
    """
    if not article:
        return False
    if not article.get("text"):
        return False
    if is_blocked_site(article.get("url", "")):
        return True
    return False

#  Extract news articles from the given URLs

def extract_news_articles(urls):
    """
    * Extract news articles from the given URLs
    """
    extracted_articles = []
    existing_urls = set()

    for url in urls:
        if not is_valid_url_content(url):
            logging.warning(f"Skipping invalid or paywalled URL: {url}")
            continue
        if is_duplicate(url, existing_urls):
            logging.warning(f"Skipping duplicate URL: {url}")
            continue
        existing_urls.add(url)

        article = extract_full_content(url)
        if not article:
            logging.warning(f"Failed to extract content from {url}")
            continue

        if is_paywalled_content(article):
            logging.warning(f"Skipping paywalled content from URL: {url}")
            continue

        extracted_articles.append(article)

    return extracted_articles

def extract_news_articles_rss(urls):
    """
    * Extract news articles from the given RSS URLs
    """ 
    extracted_articles = []
    existing_urls = set()

    for url in urls:
        if not is_valid_url_content(url):
            logging.warning(f"Skipping invalid or paywalled URL: {url}")
            continue
        if is_duplicate(url, existing_urls):
            logging.warning(f"Skipping duplicate URL: {url}")
            continue
        existing_urls.add(url)

        article = extract_full_content_rss(url)
        if not article:
            logging.warning(f"Failed to extract content from {url}")
            continue

        if is_paywalled_content(article):
            logging.warning(f"Skipping paywalled content from URL: {url}")
            continue

        extracted_articles.append(article)

    return extracted_articles

# Metadata Structuring and Storage 
# Functions to create a dataframe with all the metadata for extracted fields title,url,source,author, published_at and full_text for each extracted article and save it to a csv file

def create_dataframe(articles):
    """
    Create a pandas DataFrame from the list of articles.
    """
    return pd.DataFrame(articles)

def save_to_csv(df, filename):
    """
    Save the DataFrame to a CSV file.
    """
    df.to_csv(filename, index=False)

def save_to_json(df, filename):
    """
    Save the DataFrame to a JSON file.
    """
    df.to_json(filename, orient="records", lines=True)