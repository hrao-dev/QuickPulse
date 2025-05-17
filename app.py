import gradio as gr
import pandas as pd
import cluster_news
import extract_news
import summarizer
import analyze_sentiment
import gather_news

# ------------------ Utilities ------------------

def fetch_content(topic):
    articles = gather_news.fetch_articles_newsapi(topic)
    if isinstance(articles, str):
        articles = gather_news.fetch_articles_google(topic)
        if isinstance(articles, str):
            return None
    try:
        articles = sorted(articles, key=lambda x: x.get("publishedAt", ""), reverse=True)[:10]
    except Exception:
        return None
    return articles

def fetch_and_process_latest_news(sentiment_filters):
    topic = "Top Headlines"
    articles = gather_news.fetch_articles_newsapi("top headlines")
    if isinstance(articles, str) or not articles:
        return sentiment_filters, "### No latest news available", "", "", "", "", None

    articles = sorted(articles, key=lambda x: x.get("publishedAt", ""), reverse=True)[:10]
    extracted_articles = extract_summarize_and_analyze_articles(articles)

    if not extracted_articles:
        return sentiment_filters, "### No content to display", "", "", "", "", None

    df = pd.DataFrame(extracted_articles)
    result = cluster_news.cluster_and_label_articles(df, content_column="content", summary_column="summary")
    cluster_md_blocks = display_clusters_as_columns(result, sentiment_filters)
    csv_file, _ = save_clustered_articles(result["dataframe"], topic)

    return sentiment_filters, *cluster_md_blocks, csv_file

def extract_summarize_and_analyze_articles(articles):
    extracted_articles = []
    for article in articles:
        url = article.get("url")
        if url:
            content, _ = extract_news.extract_full_content(url)
            if content:
                summary = summarizer.generate_summary(content)
                sentiment, score = analyze_sentiment.analyze_summary(summary)
                extracted_articles.append({
                    "title": article.get("title", "No title"),
                    "url": url,
                    "source": article.get("source", "Unknown"),
                    "author": article.get("author", "Unknown"),
                    "publishedAt": article.get("publishedAt", "Unknown"),
                    "content": content,
                    "summary": summary,
                    "sentiment": sentiment,
                    "score": score
                })
    return extracted_articles

def extract_summarize_and_analyze_content_from_file(files):
    extracted_articles = []
    for file in files:
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read()
            if content.strip():
                summary = summarizer.generate_summary(content)
                sentiment, score = analyze_sentiment.analyze_summary(summary)
                extracted_articles.append({
                    "title": "Custom File",
                    "url": "N/A",
                    "source": "Uploaded File",
                    "author": "Unknown",
                    "publishedAt": "Unknown",
                    "content": content,
                    "summary": summary,
                    "sentiment": sentiment,
                    "score": score
                })
    return extracted_articles

def extract_summarize_and_analyze_content_from_urls(urls):
    extracted_articles = []
    for url in urls:
        content, title = extract_news.extract_full_content(url)
        if content:  # Only proceed if content is successfully extracted
            summary = summarizer.generate_summary(content)
            sentiment, score = analyze_sentiment.analyze_summary(summary)
            extracted_articles.append({
                "title": title if title else "Untitled Article",
                "url": url,
                "source": "External Link",
                "author": "Unknown",
                "publishedAt": "Unknown",
                "content": content,
                "summary": summary,
                "sentiment": sentiment,
                "score": score
            })
    return extracted_articles

def display_clusters_as_columns(result, sentiment_filters=None):
    df = result["dataframe"]
    detected_topics = result.get("detected_topics", {})
    df["sentiment"] = df["sentiment"].str.capitalize()

    if sentiment_filters:
        df = df[df["sentiment"].isin(sentiment_filters)]

    if df.empty:
        return ["### ⚠️ No matching articles."] + [""] * 4

    clusters = df.groupby("cluster_label")
    markdown_blocks = []

    for cluster_label, articles in clusters:
        cluster_md = f"### 🧩 Cluster {cluster_label}\n"
        if cluster_label in detected_topics:
            topics = detected_topics[cluster_label]
            cluster_md += f"**Primary Topic:** {topics['primary_focus']}\n\n"
            if topics["related_topics"]:
                cluster_md += f"**Related Topics:** {', '.join(topics['related_topics'])}\n\n"
        cluster_md += f"**Articles:** {len(articles)}\n\n"
        for _, article in articles.iterrows():
            cluster_md += (
                f"#### 📰 {article['title']}\n"
                f"- **Source:** {article['source']}\n"
                f"- **Sentiment:** {article['sentiment']}\n"
                f"<details><summary><strong>Summary</strong></summary>\n"
                f"{article['summary']}\n"
                f"</details>\n"
                f"- [Read Full Article]({article['url']})\n\n"
            )
        
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

# ------------------ Pipeline Trigger ------------------

def update_ui_with_columns(topic, files, urls, sentiment_filters):
    extracted_articles = []

    if topic.strip():
        articles = fetch_content(topic)
        if articles:
            extracted_articles.extend(extract_summarize_and_analyze_articles(articles))

    if files:
        extracted_articles.extend(extract_summarize_and_analyze_content_from_file(files))

    if urls:
        url_list = [url.strip() for url in urls.split("\n") if url.strip()]
        extracted_articles.extend(extract_summarize_and_analyze_content_from_urls(url_list))

    if not extracted_articles:
        return sentiment_filters, "### No content to display", "", "", "", "", None

    df = pd.DataFrame(extracted_articles)
    result = cluster_news.cluster_and_label_articles(df, content_column="content", summary_column="summary")
    cluster_md_blocks = display_clusters_as_columns(result, sentiment_filters)
    csv_file, _ = save_clustered_articles(result["dataframe"], topic or "batch_upload")

    return sentiment_filters, *cluster_md_blocks, csv_file

def clear_interface():
    return (
        "",                                 # topic_input
        ["Positive", "Neutral", "Negative"],# sentiment_filter
        gr.update(value=None),              # uploaded_files (reset file upload)
        "",                                 # urls_input
        "", "", "", "", "",                 # cluster columns 0–4
        gr.update(value=None)               # csv_output (reset download file)
    )


# ------------------ Gradio UI ------------------

with gr.Blocks(theme=gr.themes.Base(), css=".gr-markdown { margin: 10px; }") as demo:
    
    # Header Section
    gr.Markdown("# 📰 Quick Pulse")
    gr.Markdown("### AI-Powered News Summarization with Real-Time Sentiment and Topic Insights")
    gr.Markdown(
        "From headlines to insight, Quick Pulse summarizes news stories, captures emotional context, and clusters related topics to provide structured intelligence—faster than ever")

    # Input Section
    gr.Markdown("---")  # Horizontal line for separation
    with gr.Accordion("🗞️ Latest Top Headlines", open=False):
        latest_news_button = gr.Button("Fetch & Summarize Top 10 Headlines")

    with gr.Row():
        topic_input = gr.Textbox(label="Enter Topic", placeholder="e.g. climate change")
        sentiment_filter = gr.CheckboxGroup(choices=["Positive", "Neutral", "Negative"], value=["Positive", "Neutral", "Negative"], label="Sentiment Filter")
        csv_output = gr.File(label="📁 Download Clustered Digest CSV")

    with gr.Accordion("📂 Upload Articles (.txt files)", open=False):
        uploaded_files = gr.File(label="Upload .txt Files", file_types=[".txt"], file_count="multiple")

    with gr.Accordion("🔗 Enter Multiple URLs", open=False):
        urls_input = gr.Textbox(label="Enter URLs (newline separated)", lines=4)

    with gr.Row():
        submit_button = gr.Button(" Generate Digest")
        clear_button = gr.Button(" Clear")

    with gr.Row():
        column_0 = gr.Markdown()
        column_1 = gr.Markdown()
        column_2 = gr.Markdown()
        column_3 = gr.Markdown()
        column_4 = gr.Markdown()

    submit_button.click(
        fn=update_ui_with_columns,
        inputs=[topic_input, uploaded_files, urls_input, sentiment_filter],
        outputs=[
            sentiment_filter,
            column_0, column_1, column_2, column_3, column_4,
            csv_output
        ]
    )

    latest_news_button.click(
        fn=fetch_and_process_latest_news,
        inputs=[sentiment_filter],
        outputs=[
            sentiment_filter,
            column_0, column_1, column_2, column_3, column_4,
            csv_output
        ]
    )

    clear_button.click(
    fn=clear_interface,
    inputs=[],
    outputs=[
        topic_input,          # 1
        sentiment_filter,     # 2
        uploaded_files,       # 3
        urls_input,           # 4
        column_0,             # 5
        column_1,             # 6
        column_2,             # 7
        column_3,             # 8
        column_4,             # 9
        csv_output            # 10
    ]
)



if __name__ == "__main__":
    demo.launch()
