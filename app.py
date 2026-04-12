# app.py
# QuickPulse — Streamlit briefing UI
# Fixes in this version:
#   1. cluster_news.py now uses fast keyword rules — no model loading,
#      classification of 50 articles takes <100ms instead of 5-10 min.
#   2. Sidebar "Show topics" multiselect now correctly filters cards.
#      Root cause was st.session_state key collision with the widget key —
#      fixed by using a dedicated session key and an on_change callback.
#   3. Pipeline wrapped in st.cache_data (TTL=60min) so re-renders don't
#      re-run the fetch+classify+Groq steps.

import time

import pandas as pd
import plotly.express as px
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

# ── Design tokens ─────────────────────────────────────────────────────────────

ACCENT        = "#65C23A"
BG_BASE       = "#0d0f12"
BG_CARD       = "#13161b"
BG_ELEVATED   = "#1a1e26"
BORDER        = "#252932"
TXT_PRIMARY   = "#e8eaf0"
TXT_SECONDARY = "#9aa0ad"

SENTIMENT_CFG = {
    "Mostly Positive": {"bg": "#0d1f0a", "border": "#3a7d1e", "text": "#65C23A",  "label": "⬆ Mostly Positive"},
    "Mixed":           {"bg": "#0e1520", "border": "#2a5298", "text": "#4a7fa5",  "label": "↔ Mixed"},
    "Mostly Negative": {"bg": "#1f0d0d", "border": "#8b1a1a", "text": "#c0392b",  "label": "⬇ Mostly Negative"},
}

# ── Global CSS ─────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"], .stApp {{
    background-color: {BG_BASE} !important;
    color: {TXT_PRIMARY} !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}}
[data-testid="stSidebar"] {{
    background-color: {BG_CARD} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {TXT_PRIMARY} !important; }}

/* Primary button */
.stButton > button {{
    background-color: {ACCENT} !important;
    color: #0d0f12 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}}
.stButton > button:hover {{ opacity: 0.85 !important; }}

/* Inputs */
.stTextInput > div > div > input {{
    background-color: {BG_ELEVATED} !important;
    color: {TXT_PRIMARY} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
/* Multiselect */
[data-baseweb="select"] > div {{
    background-color: {BG_ELEVATED} !important;
    border-color: {BORDER} !important;
    border-radius: 8px !important;
}}
[data-baseweb="tag"] {{
    background-color: rgba(101,194,58,0.15) !important;
    color: {ACCENT} !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid {BORDER} !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {TXT_SECONDARY} !important;
    font-weight: 500 !important;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{ background: {BG_CARD} !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG_BASE}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

#MainMenu, footer, header {{ visibility: hidden; }}

@keyframes qp-pulse {{
    0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }}
}}
.pulse-dot {{
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; background: {ACCENT};
    animation: qp-pulse 2s ease-in-out infinite;
    margin-right: 6px; vertical-align: middle;
}}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="text-align:center;padding:40px 24px 24px;border-bottom:1px solid {BORDER};margin-bottom:24px;">
  <div style="display:inline-flex;align-items:center;gap:8px;
              background:rgba(101,194,58,0.10);border:1px solid rgba(101,194,58,0.25);
              border-radius:20px;padding:4px 14px;margin-bottom:16px;">
    <span class="pulse-dot"></span>
    <span style="font-size:0.7rem;font-weight:700;color:{ACCENT};letter-spacing:0.12em;
                 text-transform:uppercase;font-family:'JetBrains Mono',monospace;">
      live · multi-source
    </span>
  </div>
  <h1 style="margin:0 0 6px;font-size:clamp(1.8rem,4vw,2.5rem);font-weight:700;
             color:{TXT_PRIMARY};letter-spacing:-0.02em;">QuickPulse</h1>
  <p style="margin:0 auto 4px;font-size:0.95rem;color:{ACCENT};font-weight:500;">
    Today's news, distilled by topic.
  </p>
  <p style="margin:0 auto;max-width:500px;font-size:0.85rem;color:{TXT_SECONDARY};line-height:1.6;">
    50 live headlines → 6 topic buckets → synthesised briefings with sentiment &amp; entities.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────

if "briefing_data" not in st.session_state:
    st.session_state.briefing_data = None
if "selected_topics" not in st.session_state:
    st.session_state.selected_topics = list(cluster_news.TOPIC_LABELS)

# ── Pipeline (cached) ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_pipeline(topic_key: str) -> dict:
    """
    Fetch → classify (keyword rules, <100ms) → Groq synthesis.
    Cached for 1 hour by topic_key. Re-runs only when key changes or TTL expires.
    """
    topic = topic_key if topic_key != "__top__" else None
    articles = gather_news.fetch_articles(topic=topic, max_articles=50)
    if not articles:
        return {}
    articles = cluster_news.classify_articles(articles)   # <100ms, no model
    topic_buckets = cluster_news.group_by_topic(articles)
    return briefing_module.generate_briefing(topic_buckets, force=False)


def run_pipeline(topic: str | None, force: bool = False) -> dict:
    topic_key = topic.strip() if topic and topic.strip() else "__top__"
    if force:
        # Clear cache for this key so it re-runs
        _cached_pipeline.clear()
    return _cached_pipeline(topic_key)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"<p style='font-size:0.68rem;font-weight:700;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:{TXT_SECONDARY};margin-bottom:10px;'>"
        f"Controls</p>",
        unsafe_allow_html=True,
    )

    topic_input = st.text_input(
        "Custom topic (optional)",
        placeholder="e.g. quantum computing",
        key="topic_input_widget",
        help="Leave blank to fetch top headlines.",
    )

    # --- Topic filter (the bug fix) ---
    # We store the selection in st.session_state.selected_topics explicitly
    # so it survives reruns without fighting the widget key.
    def _on_topic_change():
        st.session_state.selected_topics = st.session_state._topic_filter_widget

    st.multiselect(
        "Show topics",
        options=cluster_news.TOPIC_LABELS,
        default=st.session_state.selected_topics,
        key="_topic_filter_widget",
        on_change=_on_topic_change,
        help="Deselect topics to hide their cards.",
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("⚡ Generate", use_container_width=True)
    with col2:
        force_btn = st.button("↺ Refresh", use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<p style='font-size:0.72rem;color:{TXT_SECONDARY};line-height:1.6;'>"
        f"Classification: keyword rules (~instant)<br>"
        f"Synthesis: Groq API (~20s first run)<br>"
        f"Cache TTL: 60 min</p>",
        unsafe_allow_html=True,
    )

# ── Trigger pipeline ───────────────────────────────────────────────────────────

if run_btn or force_btn:
    topic_val = st.session_state.get("topic_input_widget", "").strip() or None
    with st.spinner("Fetching & classifying headlines... (this is fast now ⚡)"):
        data = run_pipeline(topic=topic_val, force=force_btn)
    if data:
        st.session_state.briefing_data = data
    else:
        st.error("No articles fetched — check your api_key and GROQ_API_KEY secrets.")

# Auto-load from cache on first visit
if st.session_state.briefing_data is None:
    cached = briefing_module._load_cache()
    if cached:
        st.session_state.briefing_data = cached

briefing_data = st.session_state.briefing_data

# ── Last-updated badge ─────────────────────────────────────────────────────────

if briefing_data:
    lu = briefing_module.get_last_updated(briefing_data)
    st.markdown(
        f"<p style='font-size:0.74rem;color:{TXT_SECONDARY};text-align:right;"
        f"margin:-8px 0 16px;'>Last updated: "
        f"<span style='color:{ACCENT};'>{lu}</span></p>",
        unsafe_allow_html=True,
    )

# ── Card helpers ───────────────────────────────────────────────────────────────

def _badge(text: str, color: str, bg: str, border: str) -> str:
    return (
        f"<span style='background:{bg};border:1px solid {border};"
        f"border-radius:20px;padding:3px 10px;font-size:0.71rem;"
        f"font-weight:700;color:{color};white-space:nowrap;'>{text}</span>"
    )

def _chip(label: str, color: str = ACCENT, bg: str = "rgba(101,194,58,0.10)") -> str:
    return (
        f"<span style='background:{bg};border:1px solid {color}40;"
        f"border-radius:20px;padding:2px 9px;font-size:0.71rem;"
        f"color:{color};margin:2px 2px 0 0;display:inline-block;'>{label}</span>"
    )

def render_card(topic: str, data: dict) -> None:
    sentiment = data.get("sentiment", "Mixed")
    scfg = SENTIMENT_CFG.get(sentiment, SENTIMENT_CFG["Mixed"])

    entity_chips = "".join(
        _chip(e, color="#aab0bc", bg=BG_ELEVATED) for e in data.get("entities", [])
    )
    kw_chips = "".join(_chip(k) for k in data.get("keywords", []))

    source_links = "".join(
        f"<div style='margin:4px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
        f"<a href='{a['url']}' target='_blank' style='color:{ACCENT};"
        f"font-size:0.8rem;text-decoration:none;font-weight:500;'>"
        f"→ {a['title'][:75]}{'…' if len(a['title'])>75 else ''}</a>"
        f"<span style='color:{TXT_SECONDARY};font-size:0.7rem;margin-left:8px;'>"
        f"{a['source']}</span></div>"
        for a in data.get("articles", [])[:3]
    )

    sent_badge  = _badge(scfg["label"], scfg["text"], scfg["bg"], scfg["border"])
    vol_badge   = _badge(f"{data.get('volume',0)} articles", TXT_SECONDARY, BG_ELEVATED, BORDER)

    st.markdown(f"""
    <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;
                padding:20px 22px;margin-bottom:20px;">

      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  margin-bottom:14px;flex-wrap:wrap;gap:8px;">
        <span style="font-size:1rem;font-weight:600;color:{TXT_PRIMARY};">
          {data.get('emoji','📰')} {topic}
        </span>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          {sent_badge} {vol_badge}
        </div>
      </div>

      <p style="font-size:0.88rem;color:{TXT_PRIMARY};line-height:1.65;
                margin:0 0 14px;border-left:3px solid {scfg['border']};
                padding-left:12px;">{data.get('briefing','')}</p>

      {"<div style='margin-bottom:10px;'><span style='font-size:0.68rem;font-weight:700;color:" + TXT_SECONDARY + ";letter-spacing:0.08em;text-transform:uppercase;margin-right:8px;'>Entities</span>" + entity_chips + "</div>" if entity_chips else ""}
      {"<div style='margin-bottom:14px;'><span style='font-size:0.68rem;font-weight:700;color:" + TXT_SECONDARY + ";letter-spacing:0.08em;text-transform:uppercase;margin-right:8px;'>Keywords</span>" + kw_chips + "</div>" if kw_chips else ""}

      <div style="border-top:1px solid {BORDER};margin:12px 0;"></div>
      <span style="font-size:0.68rem;font-weight:700;color:{TXT_SECONDARY};
                   letter-spacing:0.08em;text-transform:uppercase;display:block;margin-bottom:6px;">
        Top stories
      </span>
      {source_links}
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_cards, tab_charts = st.tabs(["📰 Briefing Cards", "📊 Insights"])

with tab_cards:
    if not briefing_data or not briefing_data.get("topics"):
        st.markdown(f"""
        <div style="text-align:center;padding:60px 24px;color:{TXT_SECONDARY};">
          <p style="font-size:2rem;margin-bottom:10px;">⚡</p>
          <p style="font-size:1rem;font-weight:600;color:{TXT_PRIMARY};margin-bottom:6px;">No briefing yet</p>
          <p style="font-size:0.85rem;">Hit <span style="color:{ACCENT};font-weight:700;">Generate</span>
          in the sidebar. First run takes ~20s (Groq calls). After that it's cached.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        selected = st.session_state.selected_topics
        filtered = {
            t: d for t, d in briefing_data["topics"].items()
            if t in selected
        }

        if not filtered:
            st.info("No topics selected. Use the sidebar filter to choose topics.")
        else:
            items = list(filtered.items())
            for i in range(0, len(items), 2):
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    render_card(*items[i])
                if i + 1 < len(items):
                    with c2:
                        render_card(*items[i + 1])

with tab_charts:
    if not briefing_data or not briefing_data.get("topics"):
        st.info("Generate a briefing first to see charts.")
    else:
        selected = st.session_state.selected_topics
        topics   = {t: d for t, d in briefing_data["topics"].items() if t in selected}

        if not topics:
            st.info("No topics selected.")
        else:
            # Volume bar
            vol_df = pd.DataFrame([
                {"Topic": f"{d['emoji']} {t}", "Articles": d["volume"]}
                for t, d in topics.items()
            ]).sort_values("Articles", ascending=False)

            fig_vol = px.bar(
                vol_df, x="Topic", y="Articles",
                title="Article volume by topic",
                color="Articles",
                color_continuous_scale=[[0,"#2a5e14"],[0.5,"#65C23A"],[1,"#a3e87a"]],
                text="Articles",
            )
            fig_vol.update_traces(textposition="outside", marker_line_width=0,
                                  textfont=dict(color=TXT_PRIMARY, size=11))
            fig_vol.update_layout(
                height=320, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                font=dict(family="Inter,sans-serif", color=TXT_SECONDARY, size=11),
                title_font=dict(color=TXT_PRIMARY, size=13),
                coloraxis_showscale=False,
                xaxis=dict(gridcolor=BG_ELEVATED, linecolor=BORDER, tickangle=-15),
                yaxis=dict(gridcolor=BG_ELEVATED, linecolor=BORDER),
                margin=dict(l=16, r=16, t=44, b=60),
            )

            # Sentiment donut
            sent_counts: dict[str, int] = {}
            for d in topics.values():
                s = d.get("sentiment", "Mixed")
                sent_counts[s] = sent_counts.get(s, 0) + 1

            fig_sent = px.pie(
                pd.DataFrame([{"Sentiment": k, "Count": v} for k, v in sent_counts.items()]),
                names="Sentiment", values="Count",
                title="Sentiment distribution",
                color="Sentiment",
                color_discrete_map={"Mostly Positive":"#65C23A","Mixed":"#4a7fa5","Mostly Negative":"#c0392b"},
                hole=0.48,
            )
            fig_sent.update_traces(
                textinfo="label+percent",
                textfont=dict(color=TXT_PRIMARY, size=11),
                marker=dict(line=dict(color=BG_BASE, width=2)),
            )
            fig_sent.update_layout(
                height=320, paper_bgcolor=BG_CARD,
                font=dict(family="Inter,sans-serif", color=TXT_SECONDARY, size=11),
                title_font=dict(color=TXT_PRIMARY, size=13),
                legend=dict(bgcolor=BG_ELEVATED, bordercolor=BORDER, borderwidth=1,
                            font=dict(color=TXT_SECONDARY, size=11)),
                margin=dict(l=16, r=16, t=44, b=16),
            )

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.plotly_chart(fig_vol,  use_container_width=True)
            with c2:
                st.plotly_chart(fig_sent, use_container_width=True)

            # Summary table
            st.markdown(
                f"<p style='font-size:0.68rem;font-weight:700;letter-spacing:0.1em;"
                f"text-transform:uppercase;color:{TXT_SECONDARY};margin:20px 0 8px;'>"
                f"Topic summary</p>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame([{
                    "Topic":        f"{d['emoji']} {t}",
                    "Articles":     d["volume"],
                    "Sentiment":    d["sentiment"],
                    "Top entities": ", ".join(d.get("entities", [])[:3]),
                } for t, d in topics.items()]),
                use_container_width=True,
                hide_index=True,
            )

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="text-align:center;padding:28px 0 12px;margin-top:32px;border-top:1px solid {BORDER};">
  <p style="font-size:0.74rem;color:{TXT_SECONDARY};">
    QuickPulse · Part of the
    <a href="https://huggingface.co/harao-ml" target="_blank"
       style="color:{ACCENT};text-decoration:none;">harao-ml</a> NLP portfolio ·
    <a href="https://huggingface.co/spaces/harao-ml/SumUp" target="_blank"
       style="color:{ACCENT};text-decoration:none;">SumUp</a>
    &nbsp;·&nbsp;
    <a href="https://huggingface.co/spaces/harao-ml/DocQuest" target="_blank"
       style="color:{ACCENT};text-decoration:none;">DocQuest</a>
  </p>
</div>
""", unsafe_allow_html=True)
