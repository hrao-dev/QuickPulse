# app.py
# QuickPulse — AI-Powered News Intelligence Dashboard
#
# Pipeline (in execution order):
#   1. gather_news       — fetch articles from NewsAPI (title + snippet only)
#   2. cluster_news      — semantic embedding classifier (MiniLM prototypes)
#   3. ner_trends        — spaCy NER aggregated across all articles
#   4. divergence        — cross-source cosine variance per topic bucket
#   5. briefing          — Groq LLM synthesis per topic (cached 60 min)
#   6. Gradio UI         — render topic cards, entity panels, charts
#
# New additions vs original:
#   - cluster_news now uses sentence embeddings instead of keyword scoring
#   - ner_trends.py: new module, entity frequency across full news cycle
#   - divergence.py: new module, consensus/contested signal per topic
#   - Topic cards now show divergence badge + entity chips from NER
#   - Entity Trends panel (orgs, people, places) added to analytics row
#   - All legacy imports (summarizer, analyze_sentiment, extract_news) kept
#     as stubs so nothing breaks if they're still present

import json
import time
import traceback
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import gather_news
import cluster_news
import briefing as briefing_module
import ner_trends
import divergence as divergence_module

# ── Legacy stub imports (kept for backward compatibility) ─────────────────────
try:
    import summarizer
except ImportError:
    summarizer = None

try:
    import analyze_sentiment
except ImportError:
    analyze_sentiment = None

try:
    import extract_news
except ImportError:
    extract_news = None

# ── Constants ─────────────────────────────────────────────────────────────────

SENTIMENT_COLORS = {
    "Mostly Positive": ("#e8f5e9", "#43a047"),
    "Mixed":           ("#fff8e1", "#f9a825"),
    "Mostly Negative": ("#ffebee", "#c62828"),
}

DIVERGENCE_BADGES = {
    "Consensus":          ("✓ Consensus",   "#e8f5e9", "#2e7d32"),
    "Mixed":              ("~ Mixed",        "#fff8e1", "#f57f17"),
    "Contested":          ("⚡ Contested",   "#ffebee", "#b71c1c"),
    "Insufficient data":  ("— n/a",          "#f5f5f5", "#9e9e9e"),
}

TOPIC_EMOJIS = briefing_module.TOPIC_EMOJIS


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """
    Three-pass deduplication:
      1. Exact URL match
      2. (title, source) pair
      3. (title, snippet) pair — catches syndicated rewrites
    """
    seen_urls: set = set()
    seen_title_source: set = set()
    seen_title_snippet: set = set()
    deduped = []

    for art in articles:
        url    = art.get("url", "")
        title  = art.get("title", "").strip().lower()
        source = art.get("source", "").strip().lower()
        snippet = art.get("snippet", "").strip().lower()

        k_ts  = (title, source)
        k_tsn = (title, snippet)

        if url and url in seen_urls:
            continue
        if k_ts in seen_title_source:
            continue
        if k_tsn in seen_title_snippet:
            continue

        deduped.append(art)
        if url:
            seen_urls.add(url)
        seen_title_source.add(k_ts)
        seen_title_snippet.add(k_tsn)

    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(topic: str | None = None) -> dict:
    """
    Full pipeline from fetch → classify → NER → divergence → briefing.

    Returns a result dict:
    {
        "briefing":    { ...briefing_module output... },
        "ner":         { "ORG": [...], "PERSON": [...], "GPE": [...] },
        "divergence":  { topic: { score, label, ... }, ... },
        "topic_buckets": { topic: [articles] },
        "all_articles":  [articles],
    }
    """
    # 1. Fetch
    raw_articles = gather_news.fetch_articles(topic=topic)
    if not raw_articles:
        return {}

    # Sort newest-first
    raw_articles = sorted(
        raw_articles,
        key=lambda x: x.get("publishedAt", ""),
        reverse=True,
    )

    # 2. Deduplicate (pre-classification, on raw articles)
    articles = deduplicate_articles(raw_articles)
    if not articles:
        return {}

    # 3. Semantic embedding classification
    #    classify_articles() attaches 'topic', 'topic_score', 'topic_scores',
    #    and '_embedding' to each article dict in-place.
    articles = cluster_news.classify_articles(articles)

    # 4. Group into topic buckets
    topic_buckets = cluster_news.group_by_topic(articles)

    # 5. NER — runs over ALL articles regardless of topic bucket
    ner_result = ner_trends.extract_trending_entities(articles, top_n=8)

    # 6. Divergence — per topic bucket, reuses _embedding already on each article
    divergence_result = divergence_module.annotate_topic_buckets(topic_buckets)

    # 7. Groq briefing synthesis (cache-aware)
    #    Pass force=False so the 60-min cache is respected
    briefing_result = briefing_module.generate_briefing(
        topic_buckets=topic_buckets,
        force=False,
    )

    return {
        "briefing":      briefing_result,
        "ner":           ner_result,
        "divergence":    divergence_result,
        "topic_buckets": topic_buckets,
        "all_articles":  articles,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_topic_frequency(topic_buckets: dict[str, list]) -> go.Figure:
    if not topic_buckets:
        return go.Figure()

    topics = list(topic_buckets.keys())
    counts = [len(topic_buckets[t]) for t in topics]
    short_labels = [t.split("&")[0].strip() for t in topics]

    fig = px.bar(
        x=short_labels,
        y=counts,
        labels={"x": "Topic", "y": "Articles"},
        title="Articles per topic",
        color=short_labels,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_sentiment_distribution(briefing_result: dict) -> go.Figure:
    if not briefing_result or "topics" not in briefing_result:
        return go.Figure()

    counts = {"Mostly Positive": 0, "Mixed": 0, "Mostly Negative": 0}
    for topic_data in briefing_result["topics"].values():
        s = topic_data.get("sentiment", "Mixed")
        counts[s] = counts.get(s, 0) + 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = ["#81c784", "#ffd54f", "#e57373"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo="label+percent",
        hole=0.35,
    )])
    fig.update_layout(
        title="Sentiment across topics",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def plot_divergence_chart(divergence_result: dict) -> go.Figure:
    """
    Horizontal bar chart showing divergence score per topic.
    Color encodes Consensus / Mixed / Contested.
    """
    if not divergence_result:
        return go.Figure()

    label_map = {"Consensus": "#81c784", "Mixed": "#ffd54f", "Contested": "#e57373"}
    topics, scores, colors, labels = [], [], [], []

    for topic, data in divergence_result.items():
        if data.get("label", "Insufficient data") == "Insufficient data":
            continue
        short = topic.split("&")[0].strip()
        topics.append(short)
        scores.append(round(data["score"], 3))
        lbl = data["label"]
        labels.append(lbl)
        colors.append(label_map.get(lbl, "#bdbdbd"))

    if not topics:
        return go.Figure()

    fig = go.Figure(go.Bar(
        x=scores,
        y=topics,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
    ))
    fig.update_layout(
        title="Source divergence by topic",
        xaxis=dict(range=[0, 1], title="Divergence score"),
        height=300,
        margin=dict(l=20, r=80, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_top_topics_table(topic_buckets: dict, divergence_result: dict) -> pd.DataFrame:
    rows = []
    for topic, articles in sorted(
        topic_buckets.items(), key=lambda x: len(x[1]), reverse=True
    ):
        div = divergence_result.get(topic, {})
        rows.append({
            "Topic":      topic,
            "Articles":   len(articles),
            "Divergence": div.get("label", "—"),
            "Score":      div.get("score", 0.0),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HTML RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _divergence_badge_html(label: str) -> str:
    text, bg, color = DIVERGENCE_BADGES.get(label, DIVERGENCE_BADGES["Insufficient data"])
    return (
        f"<span style='"
        f"background:{bg}; color:{color}; border:1px solid {color}; "
        f"border-radius:12px; padding:2px 10px; font-size:0.82em; "
        f"font-weight:600; margin-left:8px;'>{text}</span>"
    )


def _keyword_chips_html(keywords: list[str]) -> str:
    if not keywords:
        return ""
    chips = " ".join(
        f"<span style='background:#e3f2fd; color:#0d47a1; border-radius:10px; "
        f"padding:2px 9px; font-size:0.80em; margin:2px; display:inline-block;'>"
        f"{kw}</span>"
        for kw in keywords
    )
    return f"<div style='margin:6px 0 4px 0;'>{chips}</div>"


def _entity_chips_html(entities: list[str]) -> str:
    if not entities:
        return ""
    chips = " ".join(
        f"<span style='background:#f3e5f5; color:#6a1b9a; border-radius:10px; "
        f"padding:2px 9px; font-size:0.80em; margin:2px; display:inline-block;'>"
        f"{e}</span>"
        for e in entities
    )
    return f"<div style='margin:4px 0;'><b style='font-size:0.85em;'>Key entities:</b> {chips}</div>"


def render_topic_cards(
    briefing_result: dict,
    divergence_result: dict,
    sentiment_filters: list[str],
) -> list[str]:
    """
    Render up to 6 topic cards as HTML strings for the Gradio Markdown columns.
    Each card contains:
      - Topic header + divergence badge
      - 3-sentence Groq briefing
      - Sentiment label (color-coded)
      - Named entity chips (from Groq)
      - TF-IDF keyword chips
      - Outlier article titles (if contested)
      - Article links
    """
    if not briefing_result or "topics" not in briefing_result:
        return [""] * 6

    cards = []
    topics_data = briefing_result["topics"]

    for topic, data in topics_data.items():
        sentiment = data.get("sentiment", "Mixed")

        # Apply sentiment filter — map Groq labels to filter values
        sentiment_map = {
            "Mostly Positive": "Positive",
            "Mixed":           "Neutral",
            "Mostly Negative": "Negative",
        }
        mapped = sentiment_map.get(sentiment, "Neutral")
        if sentiment_filters and mapped not in sentiment_filters:
            cards.append("")
            continue

        emoji      = data.get("emoji", "📰")
        briefing   = data.get("briefing", "")
        entities   = data.get("entities", [])
        top_story  = data.get("top_story", "")
        keywords   = data.get("keywords", [])
        volume     = data.get("volume", 0)
        articles   = data.get("articles", [])

        div_data   = divergence_result.get(topic, {})
        div_label  = div_data.get("label", "Insufficient data")
        div_badge  = _divergence_badge_html(div_label)
        outliers   = div_data.get("outlier_titles", [])

        sent_bg, sent_border = SENTIMENT_COLORS.get(sentiment, ("#fff", "#aaa"))

        # ── Card shell ──
        card = (
            f"<div style='border:1.5px solid #e0e0e0; border-radius:12px; "
            f"margin-bottom:16px; padding:16px; background:#fafafa;'>"
        )

        # Header row: emoji + topic name + divergence badge
        card += (
            f"<div style='display:flex; align-items:center; margin-bottom:10px;'>"
            f"<span style='font-size:1.4em; margin-right:8px;'>{emoji}</span>"
            f"<span style='font-size:1.05em; font-weight:700; color:#1a237e;'>{topic}</span>"
            f"{div_badge}"
            f"<span style='margin-left:auto; font-size:0.80em; color:#757575;'>"
            f"{volume} articles</span>"
            f"</div>"
        )

        # Sentiment strip
        card += (
            f"<div style='background:{sent_bg}; border-left:5px solid {sent_border}; "
            f"border-radius:4px; padding:6px 10px; margin-bottom:10px; "
            f"font-size:0.85em; font-weight:600; color:{sent_border};'>"
            f"{sentiment}"
            f"</div>"
        )

        # Briefing text
        if briefing:
            card += (
                f"<p style='font-size:0.93em; color:#212121; line-height:1.6; "
                f"margin:0 0 10px 0;'>{briefing}</p>"
            )

        # Top story callout
        if top_story:
            card += (
                f"<div style='background:#e8eaf6; border-radius:6px; "
                f"padding:6px 10px; margin-bottom:10px; font-size:0.85em; color:#283593;'>"
                f"<b>Top story:</b> {top_story}"
                f"</div>"
            )

        # Entity chips (from Groq synthesis)
        if entities:
            card += _entity_chips_html(entities)

        # TF-IDF keyword chips
        if keywords:
            card += _keyword_chips_html(keywords)

        # Contested outlier notice
        if div_label == "Contested" and outliers:
            card += (
                f"<div style='background:#fff3e0; border-left:4px solid #e65100; "
                f"border-radius:4px; padding:6px 10px; margin:8px 0; font-size:0.82em;'>"
                f"<b>Most divergent angles:</b><br>"
                + "<br>".join(f"&nbsp;&nbsp;• {t}" for t in outliers if t) +
                f"</div>"
            )

        # Article links
        if articles:
            card += "<div style='margin-top:10px;'>"
            for art in articles:
                art_title = art.get("title", "Untitled")
                art_url   = art.get("url", "#")
                art_src   = art.get("source", "")
                card += (
                    f"<div style='border-top:1px solid #eeeeee; padding:6px 0;'>"
                    f"<a href='{art_url}' target='_blank' "
                    f"style='color:#1565c0; font-size:0.88em; font-weight:500; "
                    f"text-decoration:none;'>{art_title}</a>"
                    f"<span style='font-size:0.78em; color:#9e9e9e; margin-left:6px;'>"
                    f"{art_src}</span>"
                    f"</div>"
                )
            card += "</div>"

        card += "</div>"
        cards.append(card)

    # Always return exactly 6 slots
    while len(cards) < 6:
        cards.append("")
    return cards[:6]


def render_entity_panel(ner_result: dict) -> str:
    """
    Render trending entities (ORG, PERSON, GPE) as a single HTML block
    with three side-by-side columns.
    """
    if not ner_result:
        return "<p style='color:#9e9e9e;'>No entity data available.</p>"

    label_config = {
        "ORG":    ("Organizations", "#e3f2fd", "#1565c0"),
        "PERSON": ("People",        "#f3e5f5", "#6a1b9a"),
        "GPE":    ("Places",        "#e8f5e9", "#2e7d32"),
    }

    html = "<div style='display:flex; gap:12px; flex-wrap:wrap;'>"

    for label_key, (label_name, bg, color) in label_config.items():
        entities = ner_result.get(label_key, [])
        if not entities:
            continue

        html += (
            f"<div style='flex:1; min-width:160px; background:{bg}; "
            f"border-radius:10px; padding:12px;'>"
            f"<div style='font-weight:700; color:{color}; "
            f"font-size:0.88em; margin-bottom:8px;'>{label_name}</div>"
        )

        for ent in entities[:7]:
            name  = ent.get("text", "")
            count = ent.get("count", 0)
            arts  = ent.get("articles", [])
            # Tooltip-style: show first article title on hover via title attr
            tooltip = arts[0]["title"] if arts else ""
            bar_w = min(100, count * 12)  # scale bar width to count
            html += (
                f"<div style='margin-bottom:6px;' title='{tooltip}'>"
                f"<div style='display:flex; justify-content:space-between; "
                f"font-size:0.82em; margin-bottom:2px;'>"
                f"<span style='color:#212121; font-weight:500;'>{name}</span>"
                f"<span style='color:{color};'>{count}</span>"
                f"</div>"
                f"<div style='background:rgba(0,0,0,0.08); border-radius:3px; height:4px;'>"
                f"<div style='background:{color}; width:{bar_w}%; height:4px; "
                f"border-radius:3px;'></div>"
                f"</div>"
                f"</div>"
            )

        html += "</div>"

    html += "</div>"
    return html


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN ORCHESTRATOR (called by Gradio buttons)
# ══════════════════════════════════════════════════════════════════════════════

def _empty_outputs():
    """Return the correct number of empty outputs matching all Gradio outputs."""
    return (
        ["Positive", "Neutral", "Negative"],   # sentiment_filter
        "", "", "", "", "", "",                 # 6 topic card columns
        None,                                  # topic_freq_fig
        None,                                  # sentiment_fig
        None,                                  # divergence_fig
        pd.DataFrame(),                        # top_topics_table
        "<p style='color:#9e9e9e;'>No data.</p>",  # entity_panel
        gr.update(visible=False),              # results_section
    )


def process_and_render(topic: str, sentiment_filters: list[str]) -> tuple:
    """
    Main handler for both the Generate Digest and Fetch Top News buttons.
    Returns a tuple matching all Gradio output components.
    """
    try:
        result = run_pipeline(topic=topic.strip() if topic else None)
    except Exception:
        traceback.print_exc()
        return _empty_outputs()

    if not result:
        return _empty_outputs()

    briefing_result  = result.get("briefing", {})
    ner_result       = result.get("ner", {})
    divergence_result = result.get("divergence", {})
    topic_buckets    = result.get("topic_buckets", {})

    # ── Render topic cards (6 slots)
    cards = render_topic_cards(briefing_result, divergence_result, sentiment_filters)

    # ── Charts
    topic_freq_fig   = plot_topic_frequency(topic_buckets)
    sentiment_fig    = plot_sentiment_distribution(briefing_result)
    divergence_fig   = plot_divergence_chart(divergence_result)

    # ── Top topics table
    top_table = build_top_topics_table(topic_buckets, divergence_result)

    # ── Entity panel HTML
    entity_html = render_entity_panel(ner_result)

    return (
        sentiment_filters,
        cards[0], cards[1], cards[2], cards[3], cards[4], cards[5],
        topic_freq_fig,
        sentiment_fig,
        divergence_fig,
        top_table,
        entity_html,
        gr.update(visible=True),
    )


def generate_digest(topic: str, sentiment_filters: list[str]) -> tuple:
    return process_and_render(topic, sentiment_filters)


def fetch_top_news(sentiment_filters: list[str]) -> tuple:
    return process_and_render("", sentiment_filters)


def refilter(
    topic: str,
    sentiment_filters: list[str],
    *_cached,
) -> tuple:
    """
    Re-render topic cards with new sentiment filters WITHOUT re-fetching.
    Reads from briefing cache directly.
    """
    cached = briefing_module._load_cache()
    if not cached:
        # Nothing cached yet — run the full pipeline
        return process_and_render(topic, sentiment_filters)

    # We need divergence from the last run — try loading it from the cache
    # (divergence is stored alongside briefing in cache.json if we put it there,
    #  otherwise fall back to empty)
    divergence_result = cached.get("divergence", {})

    cards = render_topic_cards(cached, divergence_result, sentiment_filters)

    return (
        sentiment_filters,
        cards[0], cards[1], cards[2], cards[3], cards[4], cards[5],
        gr.update(),   # keep existing charts
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(visible=True),
    )


def clear_all() -> tuple:
    return (
        "",                                    # topic_input
        ["Positive", "Neutral", "Negative"],   # sentiment_filter
        "", "", "", "", "", "",                 # 6 topic columns
        None,                                  # topic_freq_fig
        None,                                  # sentiment_fig
        None,                                  # divergence_fig
        pd.DataFrame(),                        # top_topics_table
        "",                                    # entity_panel
        gr.update(visible=False),              # results_section
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
.qp-header { text-align: center; margin-bottom: 4px; }
.entity-panel { padding: 12px; }
.gr-markdown p { margin: 4px 0; }
"""

with gr.Blocks(theme=gr.themes.Base(), css=CSS) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.Markdown(
        "<div class='qp-header'>"
        "<h1>QuickPulse</h1>"
        "<h3 style='color:#1976d2; margin:0;'>"
        "AI-Powered News Intelligence — Semantic Topics · Sentiment · Entity Trends · Source Divergence"
        "</h3>"
        "<p style='color:#616161; margin:6px 0 0 0;'>"
        "Fetches live headlines, classifies by meaning, detects how much sources agree, "
        "and synthesises each topic into a 3-sentence briefing via Groq."
        "</p>"
        "</div>"
    )

    # ── Controls row ────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            topic_input = gr.Textbox(
                label="Topic (leave blank for top headlines)",
                placeholder="e.g. artificial intelligence, climate policy …",
            )
            sentiment_filter = gr.CheckboxGroup(
                choices=["Positive", "Neutral", "Negative"],
                value=["Positive", "Neutral", "Negative"],
                label="Sentiment filter",
            )
            with gr.Row():
                btn_generate = gr.Button("Generate digest", variant="primary", scale=2)
                btn_top_news = gr.Button("Top headlines", scale=2)
                btn_clear    = gr.Button("Clear", scale=1)

        # ── Analytics column (always visible) ───────────────────────────────
        with gr.Column(scale=3):
            with gr.Row():
                topic_freq_fig  = gr.Plot(label="Topic frequency")
                sentiment_fig   = gr.Plot(label="Sentiment")
                divergence_fig  = gr.Plot(label="Source divergence")
            top_topics_table = gr.Dataframe(
                label="Topic summary",
                headers=["Topic", "Articles", "Divergence", "Score"],
            )

    gr.Markdown("---")

    # ── Entity trends panel ─────────────────────────────────────────────────
    gr.Markdown("### Trending entities")
    entity_panel = gr.HTML(
        value="<p style='color:#9e9e9e;'>Run a digest to see trending organizations, "
              "people, and places across today's news cycle.</p>",
        label="Entity trends",
    )

    gr.Markdown("---")

    # ── Topic cards (hidden until results arrive) ───────────────────────────
    results_section = gr.Group(visible=False)
    with results_section:
        gr.Markdown("### Topic briefings")
        with gr.Row():
            col0 = gr.Markdown()
            col1 = gr.Markdown()
            col2 = gr.Markdown()
        with gr.Row():
            col3 = gr.Markdown()
            col4 = gr.Markdown()
            col5 = gr.Markdown()

    # ── Last updated note ───────────────────────────────────────────────────
    gr.Markdown(
        "<p style='text-align:center; color:#9e9e9e; font-size:0.82em; margin-top:12px;'>"
        "Briefings are cached for 60 minutes. "
        "Charts update on every run. Divergence scores use cosine variance across source embeddings."
        "</p>"
    )

    # ── Shared output list (must match return tuple order exactly) ───────────
    _OUTPUTS = [
        sentiment_filter,
        col0, col1, col2, col3, col4, col5,
        topic_freq_fig,
        sentiment_fig,
        divergence_fig,
        top_topics_table,
        entity_panel,
        results_section,
    ]

    # ── Button wiring ────────────────────────────────────────────────────────
    btn_generate.click(
        fn=generate_digest,
        inputs=[topic_input, sentiment_filter],
        outputs=_OUTPUTS,
    )

    btn_top_news.click(
        fn=fetch_top_news,
        inputs=[sentiment_filter],
        outputs=_OUTPUTS,
    )

    # Sentiment filter change re-renders cards from cache (no refetch)
    sentiment_filter.change(
        fn=refilter,
        inputs=[topic_input, sentiment_filter],
        outputs=_OUTPUTS,
    )

    btn_clear.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            topic_input,
            sentiment_filter,
            col0, col1, col2, col3, col4, col5,
            topic_freq_fig,
            sentiment_fig,
            divergence_fig,
            top_topics_table,
            entity_panel,
            results_section,
        ],
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
