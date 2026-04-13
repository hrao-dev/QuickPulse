## QuickPulse — Gradio interface for news gathering, clustering, summarization and sentiment analysis.

import asyncio
import gather_news
import pandas as pd
import cluster_news
import summarizer
import analyze_sentiment
import extract_news
import gradio as gr
import plotly.express as px

# ── Fix Python 3.13 asyncio cleanup noise (ValueError: Invalid file descriptor: -1) ──
try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ── DARK THEME TOKENS ──────────────────────────────────────────────────────────
_DARK_CSS = """
*, *::before, *::after { box-sizing: border-box; }

body, html, .gradio-container, .gradio-container > .main,
.gradio-container > .main > .wrap, .gradio-container > div, #root, .app {
  background: #0d0f12 !important;
  background-color: #0d0f12 !important;
  color: #e8eaf0 !important;
  font-family: Inter, system-ui, sans-serif !important;
}

.gradio-container .block, .gradio-container .form, .gradio-container .panel,
.gradio-container .wrap, .gradio-container div.svelte-1ipelgc,
.gradio-container .gap, .gradio-container .padded, .gradio-container .tabitem,
.contain, .gap {
  background: #13161b !important;
  background-color: #13161b !important;
  border-color: #252932 !important;
}

.gradio-container input, .gradio-container textarea, .gradio-container select {
  background: #1a1e26 !important;
  color: #e8eaf0 !important;
  border: 1px solid #252932 !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
}
.gradio-container input:focus, .gradio-container textarea:focus {
  border-color: #65C23A !important;
  box-shadow: 0 0 0 2px rgba(101,194,58,0.15) !important;
  outline: none !important;
}
.gradio-container label, .gradio-container .label-wrap span {
  color: #9aa0ad !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}

.gradio-container button.primary, .gradio-container button[variant="primary"] {
  background: #65C23A !important;
  color: #0d0f12 !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.03em !important;
  transition: background 0.2s, transform 0.1s !important;
}
.gradio-container button.primary:hover { background: #78d44e !important; transform: translateY(-1px) !important; }
.gradio-container button.secondary, .gradio-container button[variant="secondary"] {
  background: transparent !important;
  color: #9aa0ad !important;
  border: 1px solid #252932 !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  transition: border-color 0.2s, color 0.2s !important;
}
.gradio-container button.secondary:hover { border-color: #65C23A !important; color: #65C23A !important; }

.gradio-container .accordion {
  background: #13161b !important;
  border: 1px solid #252932 !important;
  border-radius: 8px !important;
}
.gradio-container .accordion-header { color: #9aa0ad !important; }

.gradio-container .checkbox-group label {
  color: #e8eaf0 !important;
  font-size: 0.85rem !important;
  text-transform: none !important;
  letter-spacing: normal !important;
  font-weight: 400 !important;
}

.gradio-container .plot-container, .gradio-container .js-plotly-plot {
  background: #13161b !important;
  border-radius: 10px !important;
  border: 1px solid #252932 !important;
}

.gradio-container table { background: #13161b !important; border-collapse: collapse !important; }
.gradio-container th {
  background: #1a1e26 !important; color: #65C23A !important;
  font-size: 0.72rem !important; font-weight: 700 !important;
  letter-spacing: 0.08em !important; text-transform: uppercase !important;
  border-bottom: 1px solid #252932 !important; padding: 10px 14px !important;
}
.gradio-container td {
  color: #9aa0ad !important; border-bottom: 1px solid #1a1e26 !important;
  padding: 9px 14px !important; font-size: 0.85rem !important;
}
.gradio-container tr:hover td { background: #1a1e26 !important; color: #e8eaf0 !important; }

.gradio-container .file-preview {
  background: #1a1e26 !important; border: 1px solid #252932 !important;
  border-radius: 8px !important; color: #9aa0ad !important;
}

.gradio-container .prose, .gradio-container .gr-markdown { color: #e8eaf0 !important; }
.gradio-container .prose a { color: #65C23A !important; }
.gradio-container .prose h3 { color: #e8eaf0 !important; }
.gradio-container hr { border-color: #252932 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f12; }
::-webkit-scrollbar-thumb { background: #252932; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #65C23A; }

.qp-pulse-dot {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #65C23A; box-shadow: 0 0 6px #65C23A;
  animation: qp-pulse 2s ease-in-out infinite;
}
@keyframes qp-pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 6px #65C23A; }
  50%       { opacity: 0.5; box-shadow: 0 0 2px #65C23A; }
}
"""

_HERO_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<div style="text-align:center;padding:48px 24px 36px;background:linear-gradient(160deg,#0d0f12 0%,#111620 100%);min-height:220px;">
  <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(101,194,58,0.10);border:1px solid rgba(101,194,58,0.25);border-radius:20px;padding:5px 14px;margin-bottom:22px;">
    <span class="qp-pulse-dot"></span>
    <span style="font-size:0.72rem;font-weight:700;color:#65C23A;letter-spacing:0.12em;text-transform:uppercase;font-family:'JetBrains Mono',monospace;">live · multi-source</span>
  </div>
  <h1 style="margin:0 0 10px;font-size:clamp(2rem,5vw,2.8rem);font-weight:700;color:#e8eaf0;letter-spacing:-0.02em;font-family:'Inter',sans-serif;">QuickPulse</h1>
  <p style="margin:0 auto 6px;max-width:520px;font-size:1rem;color:#65C23A;font-weight:500;font-family:'Inter',sans-serif;">Now do it across the live web.</p>
  <p style="margin:0 auto;max-width:560px;font-size:0.9rem;color:#9aa0ad;line-height:1.6;font-family:'Inter',sans-serif;">Fetch, summarize and cluster many live sources simultaneously — with sentiment, topic analytics, and CSV export.</p>
</div>
"""


# ── CHART HELPERS ─────────────────────────────────────────────────────────────

def plot_topic_frequency(result):
    df = result["dataframe"]
    topic_counts = df["cluster_label"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    fig = px.bar(
        topic_counts, x="Topic", y="Count",
        title="Topic Frequency", color="Count",
        color_continuous_scale=[[0, "#2a5e14"], [0.5, "#65C23A"], [1, "#a3e87a"]],
    )
    fig.update_layout(
        showlegend=False, height=350,
        paper_bgcolor="#13161b", plot_bgcolor="#13161b",
        font=dict(family="Inter, sans-serif", color="#9aa0ad", size=12),
        title_font=dict(color="#e8eaf0", size=14, family="Inter, sans-serif"),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#1a1e26", linecolor="#252932", tickfont=dict(color="#9aa0ad", size=10)),
        yaxis=dict(gridcolor="#1a1e26", linecolor="#252932", tickfont=dict(color="#9aa0ad", size=10)),
        margin=dict(l=16, r=16, t=44, b=16),
    )
    fig.update_traces(marker_line_width=0)
    return fig


def plot_sentiment_trends(result):
    df = result["dataframe"]
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    color_map = {
        "Positive": "#65C23A", "positive": "#65C23A",
        "Neutral":  "#4a7fa5", "neutral":  "#4a7fa5",
        "Negative": "#c0392b", "negative": "#c0392b",
    }
    fig = px.pie(
        sentiment_counts, names="Sentiment", values="Count",
        title="Sentiment Distribution", color="Sentiment",
        color_discrete_map=color_map, hole=0.45,
    )
    fig.update_traces(
        textinfo="label+percent",
        textfont=dict(family="Inter, sans-serif", color="#e8eaf0", size=11),
        marker=dict(line=dict(color="#0d0f12", width=2)),
    )
    fig.update_layout(
        height=350, paper_bgcolor="#13161b",
        font=dict(family="Inter, sans-serif", color="#9aa0ad", size=12),
        title_font=dict(color="#e8eaf0", size=14, family="Inter, sans-serif"),
        legend=dict(bgcolor="#1a1e26", bordercolor="#252932", borderwidth=1, font=dict(color="#9aa0ad", size=11)),
        margin=dict(l=16, r=16, t=44, b=16),
    )
    return fig


# ── BACKEND ───────────────────────────────────────────────────────────────────

def render_top_clusters_table(result, top_n=5):
    df = result["dataframe"]
    cluster_counts = df["cluster_label"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Articles"]
    return cluster_counts.head(top_n)

def _empty_result(sentiment_filters):
    return sentiment_filters, "", "", "", "", "", None, None, None, None, gr.update(visible=False)

def fetch_and_process_latest_news(sentiment_filters):
    articles = gather_news.fetch_newsapi_top_headlines()
    return process_and_display_articles(articles, sentiment_filters, "Top Headlines")

def fetch_and_process_topic_news(topic, sentiment_filters):
    articles = gather_news.fetch_newsapi_everything(topic)
    return process_and_display_articles(articles, sentiment_filters, topic or "Topic")

def process_and_display_articles(articles, sentiment_filters, topic_label):
    if not articles:
        return _empty_result(sentiment_filters)

    articles = sorted(articles, key=lambda x: x.get("publishedAt", ""), reverse=True)
    extracted_articles = extract_summarize_and_analyze_articles(articles)
    deduped_articles = deduplicate_articles(extracted_articles)
    if not deduped_articles:
        return _empty_result(sentiment_filters)

    df = pd.DataFrame(deduped_articles)
    result = cluster_news.cluster_and_label_articles(df, content_column="content", summary_column="summary")
    if result is None:
        return _empty_result(sentiment_filters)

    cluster_md_blocks = display_clusters_as_columns_grouped_by_sentiment(result, sentiment_filters)
    csv_file, _ = save_clustered_articles(result["dataframe"], topic_label)
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
            "score": score,
        })
    return extracted_articles

def deduplicate_articles(articles):
    seen_urls, seen_title_source, seen_title_summary = set(), set(), set()
    deduped = []
    for art in articles:
        url = art.get("url")
        title = art.get("title", "").strip().lower()
        source = art.get("source", "").strip().lower()
        summary = art.get("summary", "").strip().lower()
        key_ts = (title, source)
        key_tsu = (title, summary)
        if url and url in seen_urls: continue
        if key_ts in seen_title_source: continue
        if key_tsu in seen_title_summary: continue
        deduped.append(art)
        if url: seen_urls.add(url)
        seen_title_source.add(key_ts)
        seen_title_summary.add(key_tsu)
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

    markdown_blocks = []
    for cluster_label, articles in df.groupby("cluster_label"):
        cluster_md = (
            "<div style='border:1px solid #252932;border-radius:12px;"
            "margin-bottom:20px;padding:20px;background:#13161b;font-family:Inter,sans-serif;'>"
        )
        cluster_md += (
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>"
            f"<span style='background:rgba(101,194,58,0.12);border:1px solid rgba(101,194,58,0.3);"
            f"border-radius:6px;padding:3px 10px;font-size:0.68rem;font-weight:700;color:#65C23A;"
            f"letter-spacing:0.1em;text-transform:uppercase;font-family:JetBrains Mono,monospace;'>CLUSTER</span>"
            f"<span style='font-size:1rem;font-weight:600;color:#e8eaf0;'>{cluster_label}</span>"
            f"</div>"
        )

        lda_topics = articles["lda_topics"].iloc[0] if "lda_topics" in articles.columns else ""
        if lda_topics:
            cluster_md += (f"<p style='margin:0 0 6px;font-size:0.82rem;'>"
                f"<span style='color:#9aa0ad;font-weight:600;'>Main Themes: </span>"
                f"<span style='color:#a3e87a;'>{lda_topics}</span></p>")

        primary = cluster_primary_topics.get(cluster_label, [])
        if primary:
            cluster_md += (f"<p style='margin:0 0 6px;font-size:0.82rem;'>"
                f"<span style='color:#9aa0ad;font-weight:600;'>Primary Topics: </span>"
                f"<span style='color:#65C23A;'>{', '.join(primary)}</span></p>")

        related = cluster_related_topics.get(cluster_label, [])
        if related:
            cluster_md += (f"<p style='margin:0 0 12px;font-size:0.82rem;'>"
                f"<span style='color:#9aa0ad;font-weight:600;'>Related Topics: </span>"
                f"<span style='color:#5a6270;'>{', '.join(related)}</span></p>")

        cluster_md += (f"<p style='margin:0 0 14px;font-size:0.8rem;color:#9aa0ad;'>"
            f"<span style='font-weight:600;color:#e8eaf0;'>{len(articles)}</span> articles</p>")

        sentiment_cfg = {
            "Positive": {"bg": "#0d1f0a", "border": "#3a7d1e", "dot": "#65C23A"},
            "Neutral":  {"bg": "#0e1520", "border": "#2a5298", "dot": "#4a7fa5"},
            "Negative": {"bg": "#1f0d0d", "border": "#8b1a1a", "dot": "#c0392b"},
        }

        for sentiment, cfg in sentiment_cfg.items():
            bucket = articles[articles["sentiment"] == sentiment]
            if bucket.empty:
                continue
            cluster_md += (
                f"<div style='background:{cfg['bg']};border-left:3px solid {cfg['border']};"
                f"border-radius:0 8px 8px 0;margin-bottom:12px;padding:12px 14px;'>"
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>"
                f"<span style='width:7px;height:7px;border-radius:50%;background:{cfg['dot']};display:inline-block;'></span>"
                f"<span style='font-size:0.78rem;font-weight:700;color:#e8eaf0;letter-spacing:0.04em;text-transform:uppercase;'>"
                f"{sentiment} &nbsp;<span style='color:{cfg['dot']};'>({len(bucket)})</span></span></div>"
            )
            for _, art in bucket.iterrows():
                score_badge = ""
                try:
                    sv = float(art.get("score", 0))
                    score_badge = (
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
                        f"color:{cfg['dot']};background:rgba(0,0,0,0.3);border:1px solid {cfg['border']};"
                        f"border-radius:4px;padding:1px 7px;margin-left:8px;'>{sv:.2f}</span>"
                    )
                except (ValueError, TypeError):
                    pass
                cluster_md += (
                    f"<div style='margin:0 0 10px;padding:10px 12px;background:#13161b;"
                    f"border:1px solid #252932;border-radius:8px;'>"
                    f"<p style='margin:0 0 6px;font-size:0.88rem;font-weight:600;color:#e8eaf0;'>"
                    f"📰 {art['title']}{score_badge}</p>"
                    f"<p style='margin:0 0 4px;font-size:0.78rem;color:#9aa0ad;'>"
                    f"<span style='color:#5a6270;'>Source:</span> {art['source']}</p>"
                    f"<details style='margin:6px 0;'>"
                    f"<summary style='cursor:pointer;font-size:0.78rem;font-weight:600;color:#65C23A;list-style:none;'>▶ Summary</summary>"
                    f"<p style='margin:8px 0 0 12px;font-size:0.82rem;color:#9aa0ad;line-height:1.55;'>{art['summary']}</p>"
                    f"</details>"
                    f"<a href='{art['url']}' target='_blank' style='font-size:0.78rem;color:#65C23A;"
                    f"text-decoration:none;font-weight:500;'>Read full article →</a></div>"
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
    if topic and topic.strip():
        return fetch_and_process_topic_news(topic, sentiment_filters)

    extracted_articles = []
    if urls:
        url_list = [u.strip() for u in urls.split("\n") if u.strip()]
        extracted_articles.extend(extract_summarize_and_analyze_content_from_urls(url_list))

    if not extracted_articles:
        return _empty_result(sentiment_filters)

    deduped = deduplicate_articles(extracted_articles)
    df = pd.DataFrame(deduped)
    result = cluster_news.cluster_and_label_articles(df, content_column="content", summary_column="summary")
    if result is None:
        return _empty_result(sentiment_filters)

    cluster_md_blocks = display_clusters_as_columns_grouped_by_sentiment(result, sentiment_filters)
    csv_file, _ = save_clustered_articles(result["dataframe"], topic or "batch_upload")
    topic_fig = plot_topic_frequency(result)
    sentiment_fig = plot_sentiment_trends(result)
    top_clusters_table = render_top_clusters_table(result)
    return sentiment_filters, *cluster_md_blocks, csv_file, topic_fig, sentiment_fig, top_clusters_table, gr.update(visible=True)

def clear_interface():
    return (
        "", ["Positive", "Neutral", "Negative"], "",
        "", "", "", "", "",
        None, None, None, None,
        gr.update(visible=False),
    )


# ── GRADIO UI ─────────────────────────────────────────────────────────────────

_dark_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#0d0f12",
    body_background_fill_dark="#0d0f12",
    block_background_fill="#13161b",
    block_background_fill_dark="#13161b",
    block_border_color="#252932",
    block_border_color_dark="#252932",
    input_background_fill="#1a1e26",
    input_background_fill_dark="#1a1e26",
    input_border_color="#252932",
    input_border_color_dark="#252932",
    button_primary_background_fill="#65C23A",
    button_primary_background_fill_dark="#65C23A",
    button_primary_text_color="#0d0f12",
    button_primary_text_color_dark="#0d0f12",
    body_text_color="#e8eaf0",
    body_text_color_dark="#e8eaf0",
    body_text_color_subdued="#9aa0ad",
    body_text_color_subdued_dark="#9aa0ad",
)

with gr.Blocks(theme=_dark_theme, css=_DARK_CSS) as demo:

    gr.HTML(_HERO_HTML)

    with gr.Row():
        with gr.Column(scale=2):
            topic_input = gr.Textbox(label="Enter Topic", placeholder="e.g. climate change")
            sentiment_filter = gr.CheckboxGroup(
                choices=["Positive", "Neutral", "Negative"],
                value=["Positive", "Neutral", "Negative"],
                label="Sentiment Filter",
            )
            with gr.Accordion("🔗 Enter Multiple URLs", open=False):
                urls_input = gr.Textbox(label="Enter URLs (newline separated)", lines=4)
            with gr.Row():
                submit_button = gr.Button("Generate Digest", variant="primary", scale=1)
                latest_news_button = gr.Button("Fetch Top News", variant="secondary", scale=1)
                clear_button = gr.Button("Clear", variant="secondary", scale=1)
            csv_output = gr.File(label="📁 Download Clustered Digest CSV")
        with gr.Column(scale=3):
            with gr.Row():
                topic_fig = gr.Plot(label="Topic Frequency")
                sentiment_fig = gr.Plot(label="Sentiment Trends")
            top_clusters_table = gr.Dataframe(label="Top Clusters")

    gr.Markdown("<hr style='border:none;border-top:1px solid #252932;margin:32px 0;'>")

    clustered_digest_section = gr.Group(visible=False)
    with clustered_digest_section:
        gr.Markdown(
            "<h3 style='font-family:Inter,sans-serif;font-size:0.72rem;font-weight:700;"
            "color:#65C23A;letter-spacing:0.12em;text-transform:uppercase;margin:0 0 20px;'>"
            "Clustered News Digest</h3>"
        )
        with gr.Row():
            column_0 = gr.Markdown()
            column_1 = gr.Markdown()
            column_2 = gr.Markdown()
            column_3 = gr.Markdown()
            column_4 = gr.Markdown()

    _outputs = [
        sentiment_filter,
        column_0, column_1, column_2, column_3, column_4,
        csv_output,
        topic_fig, sentiment_fig, top_clusters_table,
        clustered_digest_section,
    ]

    submit_button.click(fn=update_ui_with_columns, inputs=[topic_input, urls_input, sentiment_filter], outputs=_outputs)
    latest_news_button.click(fn=fetch_and_process_latest_news, inputs=[sentiment_filter], outputs=_outputs)
    clear_button.click(
        fn=clear_interface, inputs=[],
        outputs=[topic_input, sentiment_filter, urls_input,
                 column_0, column_1, column_2, column_3, column_4,
                 csv_output, topic_fig, sentiment_fig, top_clusters_table,
                 clustered_digest_section],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
