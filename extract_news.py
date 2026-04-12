# extract_news.py
# Slimmed down from the original.
# Full-article scraping (newspaper3k) is removed — we no longer need it because
# gather_news.py uses NewsAPI snippets directly.
# save_to_csv() is removed — CSV export is no longer a feature.
# create_dataframe() is kept for any downstream code that needs a DataFrame.

import pandas as pd


def create_dataframe(articles: list[dict]) -> pd.DataFrame:
    """Convert a list of article dicts to a DataFrame."""
    return pd.DataFrame(articles)
