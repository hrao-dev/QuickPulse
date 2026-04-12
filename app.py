# app.py
# QuickPulse — Streamlit briefing UI
# Replaces the original Gradio interface.
#
# Layout:
#   - Hero header with live pulse indicator
#   - Sidebar: topic filter + refresh controls
#   - Main: 6 topic briefing cards (2-column grid)
#     Each card shows: 3-sentence briefing, sentiment badge, entity chips,
#     keyword chips, article volume, top 3 source links
#   - Charts tab: topic volume bar chart + sentiment donut

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import briefing as briefing_module
import cluster_news
import gather_news

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuickPulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens (QuickPulse dark theme) ─────────────────────────────────────

ACCENT = "#65C23A"
BG_BASE = "#0d0f12"
BG_CARD = "#13161b"
BG_ELEVATED = "#1a1e26"
BORDER = "#252932"
TXT_PRIMARY = "#e8eaf0"
TXT_SECONDARY = "#9aa0ad"

SENTIMENT_COLORS = {
    "Mostly Positive": {"bg": "#0d1f0a", "border": "#3a7d1e", "text": "#65C23A", "label": "⬆ Mostly Positive"},
    "Mixed":           {"bg": "#0e1520", "border": "#2a5298", "text": "#4a7fa5", "label": "↔ Mixed"},
    "Mostly Negative": {"bg": "#1f0d0d", "border": "#8b1a1a", "text": "#c0392b", "label": "⬇ Mostly Negative"},
}

# ── Global CSS ─────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stApp {{
        background-color: {BG_BASE} !important;
        color: {TXT_PRIMARY} !important;
        font-family: 'Inter', system-ui, sans-serif !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {BG_CARD} !important;
        border-right: 1px solid {BORDER} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {TXT_PRIMARY} !important; }}

    /* Buttons */
    .stButton > button {{
        background-color: {ACCENT} !important;
        color: #0d0f12 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: opacity 0.15s !important;
    }}
    .stButton > button:hover {{ opacity: 0.88 !important; }}

    /* Secondary button variant via data-secondary hack */
    .secondary-btn > button {{
        background-color: transparent !important;
        color: {TXT_SECONDARY} !important;
        border: 1px solid {BORDER} !important;
    }}
    .secondary-btn > button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
        opacity: 1 !important;
    }}

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background-color: {BG_ELEVATED} !important;
        color: {TXT_PRIMARY} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent !important;
        border-bottom: 1px solid {BORDER} !important;
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {TXT_SECONDARY} !important;
        border-radius: 6px 6px 0 0 !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom: 2px solid {ACCENT} !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG_BASE}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

    /* Hide Streamlit branding */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* Pulse animation */
    @keyframes qp-pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}
    .pulse-dot {{
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        background: {ACCENT};
        animation: qp-pulse 2s ease-in-out infinite;
        margin-right: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div style="text-align:center;padding:44px 24px 28px;border-bottom:1px solid {BORDER};margin-bottom:28px;">
      <div style="display:inline-flex;align-items:center;gap:8px;
                  background:rgba(101,194,58,0.10);border:1px solid rgba(101,194,58,0.25);
                  border-radius:20px;padding:5px 14px;margin-bottom:18px;">
        <span class="pulse-dot"></span>
        <span style="font-size:0.7rem;font-weight:700;color:{ACCENT};letter-spacing:0.12em;
                     text-transform:uppercase;font-family:'JetBrains Mono',monospace;">
          live · multi-source
        </span>
      </div>
      <h1 style="margin:0 0 8px;font-size:clamp(2rem,5vw,2.6rem);font-weight:700;
                 color:{TXT_PRIMARY};letter-spacing:-0.02em;">QuickPulse</h1>
      <p style="margin:0 auto 4px;font-size:1rem;color:{ACCENT};font-weight:500;">
        Today's news, distilled by topic.
      </p>
      <p style="margin:0 auto;max-width:520px;font-size:0.88rem;color:{TXT_SECONDARY};line-height:1.6;">
        Multi-source headlines clustered into 6 topics — with synthesised briefings,
        sentiment signals, and named entity extraction.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:{TXT_SECONDARY};margin-bottom:12px;'>Controls</p>",
        unsafe_allow_html=True,
    )

    topic_input = st.text_input(
        "Custom topic (optional)",
        placeholder="e.g. quantum computing",
        help="Leave blank to fetch top headlines across all categories.",
    )

    selected_topics = st.multiselect(
        "Show topics",
        options=cluster_news.TOPIC_LABELS,
        default=cluster_news.TOPIC_LABELS,
        help="Filter which topic cards are displayed.",
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("⚡ Generate", use_container_width=True)
    with col2:
        force_btn = st.button("↺ Force refresh", use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<p style='font-size:0.72rem;color:{TXT_SECONDARY};line-height:1.6;'>"
        f"Briefings are cached for 60 min.<br>"
        f"Force refresh regenerates immediately.</p>",
        unsafe_allow_html=True,
    )

# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(topic: str | None = None, force: bool = False) -> dict:
    """Fetch → classify → synthesise → return briefing dict."""
    with st.spinner("Fetching headlines..."):
        articles = gather_news.fetch_articles(topic=topic or None, max_articles=50)

    if not articles:
        st.error("No articles fetched. Check your api_key environment variable.")
        return {}

    with st.spinner(f"Classifying {len(articles)} articles into topic buckets..."):
        articles = cluster_news.classify_articles(articles)

    topic_buckets = cluster_news.group_by_topic(articles)

    with st.spinner("Synthesising briefings via Groq (6 topics)..."):
        result = briefing_module.generate_briefing(topic_buckets, force=force)

    return result


# ── Session state — load on first visit or on button press ────────────────────

if "briefing" not in st.session_state:
    st.session_state.briefing = None

if run_btn or force_btn:
    st.session_state.briefing = run_pipeline(
        topic=topic_input.strip() or None,
        force=force_btn,
    )

# Attempt to load from cache on first visit (no button pressed yet)
if st.session_state.briefing is None:
    cached = briefing_module._load_cache()
    if cached:
        st.session_state.briefing = cached

briefing_data = st.session_state.briefing

# ── Last-updated badge ─────────────────────────────────────────────────────────

if briefing_data:
    last_updated = briefing_module.get_last_updated(briefing_data)
    st.markdown(
        f"<p style='font-size:0.75rem;color:{TXT_SECONDARY};text-align:right;"
        f"margin-bottom:16px;'>Last updated: <span style='color:{ACCENT};'>"
        f"{last_updated}</span></p>",
        unsafe_allow_html=True,
    )

# ── Helper: render one topic card ─────────────────────────────────────────────

def _sentiment_badge(sentiment: str) -> str:
    cfg = SENTIMENT_COLORS.get(sentiment, SENTIMENT_COLORS["Mixed"])
    return (
        f"<span style='background:{cfg['bg']};border:1px solid {cfg['border']};"
        f"border-radius:20px;padding:3px 10px;font-size:0.72rem;font-weight:700;"
        f"color:{cfg['text']};white-space:nowrap;'>{cfg['label']}</span>"
    )


def _chip(label: str, color: str = ACCENT, bg: str = "rgba(101,194,58,0.10)") -> str:
    return (
        f"<span style='background:{bg};border:1px solid {color}33;"
        f"border-radius:20px;padding:2px 10px;font-size:0.72rem;"
        f"color:{color};margin:2px 2px 0 0;display:inline-block;'>{label}</span>"
    )


def render_topic_card(topic: str, data: dict) -> None:
    sentiment = data.get("sentiment", "Mixed")
    cfg = SENTIMENT_COLORS.get(sentiment, SENTIMENT_COLORS["Mixed"])

    entity_chips = "".join(
        _chip(e, color="#9aa0ad", bg=BG_ELEVATED)
        for e in data.get("entities", [])
    )
    keyword_chips = "".join(
        _chip(k) for k in data.get("keywords", [])
    )
    source_links = "".join(
        f"<div style='margin:4px 0;'>"
        f"<a href='{a['url']}' target='_blank' style='color:{ACCENT};"
        f"font-size:0.8rem;text-decoration:none;font-weight:500;'>"
        f"→ {a['title'][:80]}{'…' if len(a['title']) > 80 else ''}</a>"
        f"<span style='color:{TXT_SECONDARY};font-size:0.72rem;margin-left:8px;'>"
        f"{a['source']}</span></div>"
        for a in data.get("articles", [])[:3]
    )

    volume = data.get("volume", 0)
    emoji = data.get("emoji", "📰")

    st.markdown(
        f"""
        <div style="background:{BG_CARD};border:1px solid {BORDER};
                    border-radius:12px;padding:20px 22px;height:100%;
                    margin-bottom:20px;">

          <!-- Header row -->
          <div style="display:flex;justify-content:space-between;
                      align-items:flex-start;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="font-size:1.3rem;">{emoji}</span>
              <span style="font-size:1rem;font-weight:600;color:{TXT_PRIMARY};">{topic}</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
              {_sentiment_badge(sentiment)}
              <span style="font-size:0.75rem;color:{TXT_SECONDARY};
                           background:{BG_ELEVATED};border-radius:20px;
                           padding:3px 10px;border:1px solid {BORDER};">
                {volume} articles
              </span>
            </div>
          </div>

          <!-- Briefing -->
          <p style="font-size:0.9rem;color:{TXT_PRIMARY};line-height:1.65;
                    margin:0 0 14px;border-left:3px solid {cfg['border']};
                    padding-left:12px;">{data.get('briefing', '')}</p>

          <!-- Entities -->
          {'<div style="margin-bottom:10px;"><span style="font-size:0.7rem;font-weight:700;color:' + TXT_SECONDARY + ';letter-spacing:0.08em;text-transform:uppercase;margin-right:8px;">Key entities</span>' + entity_chips + '</div>' if entity_chips else ''}

          <!-- Keywords -->
          {'<div style="margin-bottom:14px;"><span style="font-size:0.7rem;font-weight:700;color:' + TXT_SECONDARY + ';letter-spacing:0.08em;text-transform:uppercase;margin-right:8px;">Topics</span>' + keyword_chips + '</div>' if keyword_chips else ''}

          <!-- Divider -->
          <div style="border-top:1px solid {BORDER};margin:12px 0;"></div>

          <!-- Source links -->
          <div style="font-size:0.78rem;">
            <span style="font-size:0.7rem;font-weight:700;color:{TXT_SECONDARY};
                         letter-spacing:0.08em;text-transform:uppercase;
                         display:block;margin-bottom:6px;">Top stories</span>
            {source_links}
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Tabs: Briefing cards | Charts ──────────────────────────────────────────────

tab_briefing, tab_charts = st.tabs(["📰 Briefing Cards", "📊 Insights"])

with tab_briefing:
    if not briefing_data or not briefing_data.get("topics"):
        st.markdown(
            f"""
            <div style="text-align:center;padding:64px 24px;color:{TXT_SECONDARY};">
              <p style="font-size:2rem;margin-bottom:12px;">⚡</p>
              <p style="font-size:1rem;font-weight:600;color:{TXT_PRIMARY};margin-bottom:8px;">
                No briefing yet
              </p>
              <p style="font-size:0.88rem;">
                Hit <span style="color:{ACCENT};font-weight:600;">Generate</span>
                in the sidebar to fetch today's news briefing.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        topics_to_show = {
            t: d
            for t, d in briefing_data["topics"].items()
            if t in selected_topics
        }

        if not topics_to_show:
            st.info("No topics match your current filter. Adjust the sidebar selections.")
        else:
            # 2-column grid
            topic_items = list(topics_to_show.items())
            for i in range(0, len(topic_items), 2):
                col_a, col_b = st.columns(2, gap="medium")
                with col_a:
                    render_topic_card(*topic_items[i])
                if i + 1 < len(topic_items):
                    with col_b:
                        render_topic_card(*topic_items[i + 1])

with tab_charts:
    if not briefing_data or not briefing_data.get("topics"):
        st.info("Generate a briefing first to see charts.")
    else:
        topics = briefing_data["topics"]

        # ── Volume bar chart ──────────────────────────────────────────────────
        vol_data = pd.DataFrame(
            [
                {"Topic": t, "Articles": d["volume"], "Emoji": d["emoji"]}
                for t, d in topics.items()
                if t in selected_topics
            ]
        ).sort_values("Articles", ascending=False)

        fig_vol = px.bar(
            vol_data,
            x="Topic",
            y="Articles",
            title="Article volume by topic",
            color="Articles",
            color_continuous_scale=[[0, "#2a5e14"], [0.5, "#65C23A"], [1, "#a3e87a"]],
            text="Articles",
        )
        fig_vol.update_traces(
            textposition="outside",
            marker_line_width=0,
            textfont=dict(color=TXT_PRIMARY, size=11),
        )
        fig_vol.update_layout(
            height=340,
            paper_bgcolor=BG_CARD,
            plot_bgcolor=BG_CARD,
            font=dict(family="Inter, sans-serif", color=TXT_SECONDARY, size=12),
            title_font=dict(color=TXT_PRIMARY, size=14),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor=BG_ELEVATED, linecolor=BORDER, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=BG_ELEVATED, linecolor=BORDER, tickfont=dict(size=10)),
            margin=dict(l=16, r=16, t=44, b=60),
        )
        # Rotate x-axis labels for readability
        fig_vol.update_xaxes(tickangle=-20)

        # ── Sentiment donut ───────────────────────────────────────────────────
        sent_counts: dict[str, int] = {}
        for t, d in topics.items():
            if t in selected_topics:
                s = d.get("sentiment", "Mixed")
                sent_counts[s] = sent_counts.get(s, 0) + 1

        sent_df = pd.DataFrame(
            [{"Sentiment": k, "Count": v} for k, v in sent_counts.items()]
        )

        sent_color_map = {
            "Mostly Positive": "#65C23A",
            "Mixed":           "#4a7fa5",
            "Mostly Negative": "#c0392b",
        }

        fig_sent = px.pie(
            sent_df,
            names="Sentiment",
            values="Count",
            title="Sentiment distribution across topics",
            color="Sentiment",
            color_discrete_map=sent_color_map,
            hole=0.48,
        )
        fig_sent.update_traces(
            textinfo="label+percent",
            textfont=dict(family="Inter, sans-serif", color=TXT_PRIMARY, size=11),
            marker=dict(line=dict(color=BG_BASE, width=2)),
        )
        fig_sent.update_layout(
            height=340,
            paper_bgcolor=BG_CARD,
            font=dict(family="Inter, sans-serif", color=TXT_SECONDARY, size=12),
            title_font=dict(color=TXT_PRIMARY, size=14),
            legend=dict(
                bgcolor=BG_ELEVATED,
                bordercolor=BORDER,
                borderwidth=1,
                font=dict(color=TXT_SECONDARY, size=11),
            ),
            margin=dict(l=16, r=16, t=44, b=16),
        )

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.plotly_chart(fig_vol, use_container_width=True)
        with c2:
            st.plotly_chart(fig_sent, use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────────
        st.markdown(
            f"<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;"
            f"text-transform:uppercase;color:{TXT_SECONDARY};margin:24px 0 8px;'>Topic summary</p>",
            unsafe_allow_html=True,
        )
        summary_rows = [
            {
                "Topic": f"{d['emoji']} {t}",
                "Articles": d["volume"],
                "Sentiment": d["sentiment"],
                "Top entities": ", ".join(d.get("entities", [])[:3]),
            }
            for t, d in topics.items()
            if t in selected_topics
        ]
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )


# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div style="text-align:center;padding:32px 0 16px;margin-top:40px;
                border-top:1px solid {BORDER};">
      <p style="font-size:0.75rem;color:{TXT_SECONDARY};">
        QuickPulse · Part of the
        <a href="https://huggingface.co/harao-ml" target="_blank"
           style="color:{ACCENT};text-decoration:none;">harao-ml</a>
        NLP portfolio ·
        <a href="https://huggingface.co/spaces/harao-ml/SumUp" target="_blank"
           style="color:{ACCENT};text-decoration:none;">SumUp</a>
        &nbsp;·&nbsp;
        <a href="https://huggingface.co/spaces/harao-ml/DocQuest" target="_blank"
           style="color:{ACCENT};text-decoration:none;">DocQuest</a>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
