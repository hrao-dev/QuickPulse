## This script provides a Gradio interface for gathering, clustering, summarizing, and analyzing news articles with sentiment analysis and topic modeling.

import gather_news
import pandas as pd
import cluster_news
import summarizer
import analyze_sentiment
import extract_news
import gradio as gr
import plotly.express as px

def plot_topic_frequency(result):
    df = result["dataframe"]
    topic_counts = df["cluster_label"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    fig = px.bar(topic_counts, x="Topic", y="Count", title="Topic Frequency", color="Topic")
    fig.update_layout(showlegend=False, height=350)
    return fig

def plot_sentiment_trends(result):
    df = result["dataframe"]
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Sentiment Distribution")
    fig.update_traces(textinfo='label+percent')
    fig.update_layout(height=350)
    return fig

def render_top_clusters_table(result, top_n=5):
    df = result["dataframe"]
    cluster_counts = df["cluster_label"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Articles"]
    top_clusters = cluster_counts.head(top_n)
    return top_clusters

def fetch_and_process_latest_news(sentiment_filters):
    articles = gather_news.fetch_newsapi_top_headlines()
    return process_and_display_articles(articles, sentiment_filters, "Top Headlines")

def fetch_and_process_topic_news(topic, sentiment_filters):
    articles = gather_news.fetch_newsapi_everything(topic)
    return process_and_display_articles(articles, sentiment_filters, topic or "Topic")

def process_and_display_articles(articles, sentiment_filters, topic_label):
    if not articles:
        return sentiment_filters, "", "", "", "", "", None, None, None, gr.update(visible=False)

    articles = sorted(articles, key=lambda x: x.get("publishedAt", ""), reverse=True)
    extracted_articles = extract_summarize_and_analyze_articles(articles)
    deduped_articles = deduplicate_articles(extracted_articles)
    if not deduped_articles:
        return sentiment_filters, "", "", "", "", "", None, None, None, gr.update(visible=False)

    df = pd.DataFrame(deduped_articles)
    result = cluster_news.cluster_and_label_articles(df, content_column="content", summary_column="summary")
    cluster_md_blocks = display_clusters_as_columns_grouped_by_sentiment(result, sentiment_filters)
    csv_file, _ = save_clustered_articles(result["dataframe"], topic_label)

    # Analytics
    topic_fig = plot_topic_frequency(result)
    sentiment_fig = plot_sentiment_trends(result)
    top_clusters_table = render_top_clusters_table(result)

    return sentiment_filters, *cluster_md_blocks, csv_file, topic_fig, sentiment_fig, top_clusters_table, gr.update(visible=True)

def extract_summarize_and_analyze_articles(articles):
    extracted_articles = []
    for article in articles:
        content = article.get("text") or article.get("content")
        if not content:
            continue
        title = article.get("title", "No title")
        summary = summarizer.generate_summary(content)
        sentiment, score = analyze_sentiment.analyze_summary(summary)
        extracted_articles.append({
            "title": title,
            "url": article.get("url"),
            "source": article.get("source", "Unknown"),
            "author": article.get("author", "Unknown"),
            "publishedAt": article.get("publishedAt", "Unknown"),
            "content": content,
            "summary": summary,
            "sentiment": sentiment,
            "score": score
        })
    return extracted_articles

def deduplicate_articles(articles):
    seen_urls = set()
    seen_title_source = set()
    seen_title_summary = set()
    deduped = []
    for art in articles:
        url = art.get("url")
        title = art.get("title", "").strip().lower()
        source = art.get("source", "").strip().lower()
        summary = art.get("summary", "").strip().lower()
        key_title_source = (title, source)
        key_title_summary = (title, summary)
        if url and url in seen_urls:
            continue
        if key_title_source in seen_title_source:
            continue
        if key_title_summary in seen_title_summary:
            continue
        deduped.append(art)
        if url:
            seen_urls.add(url)
        seen_title_source.add(key_title_source)
        seen_title_summary.add(key_title_summary)
    return deduped

def extract_summarize_and_analyze_content_from_urls(urls):
    articles = extract_news.extract_news_articles(urls)
    return extract_summarize_and_analyze_articles(articles)

def display_clusters_as_columns_grouped_by_sentiment(result, sentiment_filters=None):
    df = result["dataframe"]
    cluster_primary_topics = result.get("cluster_primary_topics", {})
    cluster_related_topics = result.get("cluster_related_topics", {})
    df["sentiment"] = df["sentiment"].str.capitalize()

    if sentiment_filters:
        df = df[df["sentiment"].isin(sentiment_filters)]

    if df.empty:
        return ["### ⚠️ No matching articles."] + [""] * 4

    clusters = df.groupby("cluster_label")
    markdown_blocks = []

    for cluster_label, articles in clusters:
        cluster_md = f"<div style='border:2px solid #e0e0e0; border-radius:10px; margin-bottom:18px; padding:18px; background: #f9f9fa;'>"
        cluster_md += f"<h3 style='color:#2d6cdf;'>🧩 Cluster: {cluster_label}</h3>"

        lda_topics = articles["lda_topics"].iloc[0] if "lda_topics" in articles else ""
        if lda_topics:
            cluster_md += f"<b style='color:#0d47a1;'>Main Themes:</b> <span style='color:#1976d2'>{lda_topics}</span><br>"

        primary = cluster_primary_topics.get(cluster_label, [])
        if primary:
            cluster_md += f"<b style='color:#1b5e20;'>Primary Topics:</b> <span style='color:#388e3c'>{', '.join(primary)}</span><br>"

        related = cluster_related_topics.get(cluster_label, [])
        if related:
            cluster_md += f"<b style='color:#616161;'>Related Topics:</b> <span style='color:#757575'>{', '.join(related)}</span><br>"

        cluster_md += f"<b>Articles:</b> {len(articles)}<br><br>"

        for sentiment in ["Positive", "Neutral", "Negative"]:
            sentiment_articles = articles[articles["sentiment"] == sentiment]
            if not sentiment_articles.empty:
                color = {"Positive": "#e8f5e9", "Neutral": "#e3f2fd", "Negative": "#ffebee"}[sentiment]
                border = {"Positive": "#43a047", "Neutral": "#1976d2", "Negative": "#c62828"}[sentiment]
                sentiment_label = {
                    "Positive": "Positive News",
                    "Neutral": "Neutral News",
                    "Negative": "Negative News"
                }[sentiment]
                cluster_md += (
                    f"<div style='background:{color}; border-left:6px solid {border}; border-radius:6px; margin-bottom:10px; padding:10px;'>"
                    f"<span style='font-size:1.2em;'><b>{sentiment_label} ({len(sentiment_articles)})</b></span><br>"
                )
                for _, article in sentiment_articles.iterrows():
                    cluster_md += (
                        f"<div style='margin:10px 0 10px 0; padding:10px; border-bottom:1px solid #e0e0e0;'>"
                        f"<span style='font-weight:bold; color:#37474f;'>📰 {article['title']}</span><br>"
                        f"<span style='font-size:0.95em;'>"
                        f"<b>Source:</b> {article['source']}<br>"
                        f"<details><summary style='cursor:pointer; color:#1976d2;'><strong>Summary</strong></summary>"
                        f"<div style='margin-left:10px; color:#424242;'>{article['summary']}</div></details>"
                        f"<a href='{article['url']}' target='_blank' style='color:#1976d2;'>Read Full Article</a>"
                        f"</span></div>"
                    )
                cluster_md += "</div>"
        cluster_md += "</div>"
        markdown_blocks.append(cluster_md)

    while len(markdown_blocks) < 5:
        markdown_blocks.append("")

    return markdown_blocks[:5]

def save_clustered_articles(df, topic):
    if df.empty:
        return None, None
    csv_file = f"{topic.replace(' ', '_')}_clustered_articles.csv"
    df.to_csv(csv_file, index=False)
    return csv_file, None

def update_ui_with_columns(topic, urls, sentiment_filters):
    extracted_articles = []

    if topic and topic.strip():
        return fetch_and_process_topic_news(topic, sentiment_filters)

    if urls:
        url_list = [url.strip() for url in urls.split("\n") if url.strip()]
        extracted_articles.extend(extract_summarize_and_analyze_content_from_urls(url_list))

    if not extracted_articles:
        return sentiment_filters, "", "", "", "", "", None, None, None, gr.update(visible=False)

    deduped_articles = deduplicate_articles(extracted_articles)
    df = pd.DataFrame(deduped_articles)
    result = cluster_news.cluster_and_label_articles(df, content_column="content", summary_column="summary")
    cluster_md_blocks = display_clusters_as_columns_grouped_by_sentiment(result, sentiment_filters)
    csv_file, _ = save_clustered_articles(result["dataframe"], topic or "batch_upload")
    topic_fig = plot_topic_frequency(result)
    sentiment_fig = plot_sentiment_trends(result)
    top_clusters_table = render_top_clusters_table(result)
    return sentiment_filters, *cluster_md_blocks, csv_file, topic_fig, sentiment_fig, top_clusters_table, gr.update(visible=True)

def clear_interface():
    return (
        "",                                 # topic_input
        ["Positive", "Neutral", "Negative"],# sentiment_filter
        "",                                 # urls_input
        "", "", "", "", "",                 # cluster columns 0–4
        gr.update(value=None),              # csv_output (reset download file)
        None, None, None,                   # topic_fig, sentiment_fig, top_clusters_table
        gr.update(visible=False)            # Hide Clustered News Digest section
    )

with gr.Blocks(theme=gr.themes.Base(), css="""
.gr-markdown { margin: 10px; }
.analytics-card {background: #f5f7fa; border-radius: 10px; padding: 18px; margin-bottom: 18px;}
""") as demo:
    gr.Markdown(
        "<h1 style='text-align:center;'>📰 Quick Pulse</h1>"
        "<h3 style='text-align:center; color:#1976d2;'>AI-Powered News Summarization with Real-Time Sentiment and Topic Insights</h3>"
        "<p style='text-align:center;'>From headlines to insight, Quick Pulse summarizes news stories, captures emotional context, clusters related topics, and provides analytics at a glance.</p>"
    )

    with gr.Row():
        with gr.Column(scale=2):
            topic_input = gr.Textbox(label="Enter Topic", placeholder="e.g. climate change")
            sentiment_filter = gr.CheckboxGroup(choices=["Positive", "Neutral", "Negative"], value=["Positive", "Neutral", "Negative"], label="Sentiment Filter")
            with gr.Accordion("🔗 Enter Multiple URLs", open=False):
                urls_input = gr.Textbox(label="Enter URLs (newline separated)", lines=4)
            with gr.Row():
                submit_button = gr.Button(" Generate Digest", scale=1)
                latest_news_button = gr.Button("Fetch & Summarize Top News", scale=1)
                clear_button = gr.Button(" Clear", scale=1)
            csv_output = gr.File(label="📁 Download Clustered Digest CSV")
        with gr.Column(scale=3):
            with gr.Row():
                topic_fig = gr.Plot(label="Topic Frequency")
                sentiment_fig = gr.Plot(label="Sentiment Trends")
            top_clusters_table = gr.Dataframe(label="Top Clusters")

    gr.Markdown("---")

    clustered_digest_section = gr.Group(visible=False)
    with clustered_digest_section:
        gr.Markdown("<h3 style='color:#1976d2;'>Clustered News Digest</h3>")
        with gr.Row():
            column_0 = gr.Markdown()
            column_1 = gr.Markdown()
            column_2 = gr.Markdown()
            column_3 = gr.Markdown()
            column_4 = gr.Markdown()

    submit_button.click(
        fn=update_ui_with_columns,
        inputs=[topic_input, urls_input, sentiment_filter],
        outputs=[
            sentiment_filter,
            column_0, column_1, column_2, column_3, column_4,
            csv_output,
            topic_fig, sentiment_fig, top_clusters_table,
            clustered_digest_section
        ]
    )

    latest_news_button.click(
        fn=fetch_and_process_latest_news,
        inputs=[sentiment_filter],
        outputs=[
            sentiment_filter,
            column_0, column_1, column_2, column_3, column_4,
            csv_output,
            topic_fig, sentiment_fig, top_clusters_table,
            clustered_digest_section
        ]
    )

    clear_button.click(
        fn=clear_interface,
        inputs=[],
        outputs=[
            topic_input, sentiment_filter, urls_input,
            column_0, column_1, column_2, column_3, column_4,
            csv_output,
            topic_fig, sentiment_fig, top_clusters_table,
            clustered_digest_section
        ]
    )

if __name__ == "__main__":
    demo.launch()