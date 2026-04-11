# app.py
# QuickPulse — Streamlit UI
#
# Architecture:
#   - APScheduler runs pipeline.run() every 30 min in a background thread
#   - The render path ONLY reads from cache/articles.json — zero blocking API calls
#   - Users can trigger an on-demand refresh via the sidebar button
#   - A /healthz route is served by a tiny Thread so UptimeRobot keeps the Space warm

import json
import threading
import time
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler

import pipeline

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── brand colours (mirrors original Gradio theme) ────────────────────────────
ACCENT    = "#65C23A"
BG_BASE   = "#0d0f12"
BG_CARD   = "#13161b"
BG_ELV    = "#1a1e26"
BORDER    = "#252932"
TXT_PRI   = "#e8eaf0"
TXT_SEC   = "#9aa0ad"

SENTIMENT_CFG = {
    "Positive": {"dot": "#65C23A", "bg": "#0d1f0a", "border": "#3a7d1e"},
    "Neutral":  {"dot": "#4a7fa5", "bg": "#0e1520", "border": "#2a5298"},
    "Negative": {"dot": "#c0392b", "bg": "#1f0d0d", "border": "#8b1a1a"},
}

# ── health-check server (keeps HF Space warm) ─────────────────────────────────
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, *_):
        pass  # silence access logs

def _start_health_server():
    try:
        srv = HTTPServer(("0.0.0.0", 8080), _HealthHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        logger.info("Health server listening on :8080")
    except Exception as e:
        logger.warning(f"Health server failed to start: {e}")

# ── background scheduler ─────────────────────────────────────────────────────
_scheduler_started = False

def _start_scheduler():
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True

    def _refresh():
        logger.info("Scheduled refresh: top headlines")
        try:
            pipeline.run()
        except Exception as e:
            logger.error(f"Scheduled refresh failed: {e}")

    # Warm the cache immediately at startup in a daemon thread
    threading.Thread(target=_refresh, daemon=True).start()

    scheduler = BackgroundScheduler()
    scheduler.add_job(_refresh, "interval", minutes=30, id="auto_refresh")
    scheduler.start()
    logger.info("APScheduler started — refresh every 30 min")

# Run once per interpreter process (Streamlit re-runs the script on every
# interaction, so guard with a module-level flag)
_start_health_server()
_start_scheduler()


# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuickPulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: {BG_BASE} !important;
    color: {TXT_PRI} !important;
}}
.block-container {{ padding: 2rem 2.5rem 4rem; max-width: 1400px; }}

/* sidebar */
[data-testid="stSidebar"] {{
    background-color: {BG_CARD} !important;
    border-right: 1px solid {BORDER};
}}

/* inputs */
input, textarea, [data-baseweb="input"] input {{
    background-color: {BG_ELV} !important;
    color: {TXT_PRI} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}

/* buttons */
.stButton > button {{
    background-color: {ACCENT} !important;
    color: {BG_BASE} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.15s;
}}
.stButton > button:hover {{ opacity: 0.85; }}
.stButton > button[kind="secondary"] {{
    background-color: transparent !important;
    color: {TXT_SEC} !important;
    border: 1px solid {BORDER} !important;
}}

/* multiselect tags */
[data-baseweb="tag"] {{
    background-color: rgba(101,194,58,0.15) !important;
    border: 1px solid rgba(101,194,58,0.4) !important;
    color: {ACCENT} !important;
    border-radius: 6px !important;
}}

/* metric cards */
[data-testid="metric-container"] {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1rem 1.2rem;
}}

/* expander */
details summary {{
    cursor: pointer;
    color: {ACCENT} !important;
    font-weight: 600;
    font-size: 0.8rem;
}}

/* scrollbar */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG_BASE}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

/* pulse dot animation */
@keyframes qp-pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 6px {ACCENT}; }}
    50%       {{ opacity: 0.5; box-shadow: 0 0 2px {ACCENT}; }}
}}
.qp-pulse-dot {{
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: {ACCENT};
    box-shadow: 0 0 6px {ACCENT};
    animation: qp-pulse 2s ease-in-out infinite;
    margin-right: 6px;
}}
</style>
""", unsafe_allow_html=True)


# ── hero header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:40px 24px 28px;background:linear-gradient(160deg,{BG_BASE} 0%,#111620 100%);border-radius:14px;margin-bottom:28px;">
  <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(101,194,58,0.10);border:1px solid rgba(101,194,58,0.25);border-radius:20px;padding:5px 14px;margin-bottom:18px;">
    <span class="qp-pulse-dot"></span>
    <span style="font-size:0.72rem;font-weight:700;color:{ACCENT};letter-spacing:0.12em;text-transform:uppercase;font-family:'JetBrains Mono',monospace;">live · multi-source</span>
  </div>
  <h1 style="margin:0 0 8px;font-size:clamp(2rem,5vw,2.8rem);font-weight:700;color:{TXT_PRI};letter-spacing:-0.02em;">QuickPulse</h1>
  <p style="margin:0 auto 4px;max-width:520px;font-size:1rem;color:{ACCENT};font-weight:500;">Fetch, cluster and summarise live news — instantly.</p>
  <p style="margin:0 auto;max-width:560px;font-size:0.88rem;color:{TXT_SEC};line-height:1.6;">Multi-source · Sentiment analysis · HDBSCAN topic clustering · CSV export</p>
</div>
""", unsafe_allow_html=True)


# ── sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<p style='color:{ACCENT};font-weight:700;font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px;'>Controls</p>", unsafe_allow_html=True)

    mode = st.radio("Mode", ["Top Headlines", "Search by Topic", "Custom URLs"], index=0)

    topic_input = ""
    urls_input  = ""

    if mode == "Search by Topic":
        topic_input = st.text_input("Topic", placeholder="e.g. climate change")

    if mode == "Custom URLs":
        urls_input = st.text_area("URLs (one per line)", height=140)

    sentiment_filters = st.multiselect(
        "Sentiment filter",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )

    st.markdown("---")

    run_btn     = st.button("Generate Digest", use_container_width=True)
    refresh_btn = st.button("↻  Refresh cache now", use_container_width=True)
    clear_btn   = st.button("Clear", use_container_width=True)

    st.markdown("---")
    st.markdown(f"<p style='color:{TXT_SEC};font-size:0.75rem;'>Cache auto-refreshes every 30 min.<br>Use <em>Refresh cache now</em> for latest news.</p>", unsafe_allow_html=True)


# ── session-state bootstrap ───────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "running" not in st.session_state:
    st.session_state.running = False


# ── action handlers ───────────────────────────────────────────────────────────
def _run_pipeline(topic=None, urls=None):
    with st.spinner("Fetching and processing articles… this takes ~60 s on first run while the model warms up."):
        url_list = [u.strip() for u in urls.splitlines() if u.strip()] if urls else None
        result = pipeline.run(topic=topic or None, urls=url_list)
    if result:
        st.session_state.result = result
        st.success(f"Done — {result.get('article_count', len(result.get('articles', [])))} articles, {result.get('number_of_clusters', 0)} clusters.")
    else:
        st.warning("No articles returned. Check your API key or try a different topic.")

if clear_btn:
    st.session_state.result = None
    st.rerun()

if refresh_btn:
    _run_pipeline()

if run_btn:
    if mode == "Search by Topic" and not topic_input.strip():
        st.sidebar.error("Please enter a topic.")
    elif mode == "Custom URLs" and not urls_input.strip():
        st.sidebar.error("Please enter at least one URL.")
    else:
        _run_pipeline(
            topic=topic_input if mode == "Search by Topic" else None,
            urls=urls_input  if mode == "Custom URLs"    else None,
        )


# ── determine what to render ──────────────────────────────────────────────────
result = st.session_state.result

# Fall back to cache if no in-session result
if result is None:
    cached = pipeline.load_cache()
    if cached:
        result = cached

# ── empty state ───────────────────────────────────────────────────────────────
if not result or not result.get("articles"):
    st.markdown(f"""
    <div style="text-align:center;padding:60px 24px;color:{TXT_SEC};">
        <p style="font-size:2rem;margin-bottom:12px;">📰</p>
        <p style="font-size:1rem;font-weight:600;color:{TXT_PRI};">No digest yet</p>
        <p style="font-size:0.88rem;">Click <strong>Generate Digest</strong> or <strong>Refresh cache now</strong> in the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── build dataframe from result ───────────────────────────────────────────────
df = pd.DataFrame(result["articles"])
df["sentiment"] = df["sentiment"].str.capitalize()

meta = result.get("meta", {})
refreshed_at = meta.get("refreshed_at", "")
if refreshed_at:
    try:
        ts = datetime.fromisoformat(refreshed_at)
        st.caption(f"Last refreshed: {ts.strftime('%d %b %Y %H:%M UTC')}  ·  {meta.get('article_count', len(df))} articles  ·  topic: {meta.get('topic', '—')}")
    except Exception:
        pass

# Apply sentiment filter
if sentiment_filters:
    df_filtered = df[df["sentiment"].isin(sentiment_filters)]
else:
    df_filtered = df.copy()


# ── metrics row ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Articles", len(df))
m2.metric("Clusters", result.get("number_of_clusters", df["cluster_label"].nunique() if "cluster_label" in df.columns else 0))
m3.metric("Positive", int((df["sentiment"] == "Positive").sum()))
m4.metric("Negative", int((df["sentiment"] == "Negative").sum()))

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ── charts ────────────────────────────────────────────────────────────────────
def _dark_layout(fig, title=""):
    fig.update_layout(
        height=320,
        title=dict(text=title, font=dict(color=TXT_PRI, size=13, family="Inter")),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(family="Inter, sans-serif", color=TXT_SEC, size=11),
        margin=dict(l=16, r=16, t=40, b=16),
        coloraxis_showscale=False,
        legend=dict(bgcolor=BG_ELV, bordercolor=BORDER, borderwidth=1, font=dict(color=TXT_SEC, size=11)),
        xaxis=dict(gridcolor=BG_ELV, linecolor=BORDER, tickfont=dict(color=TXT_SEC, size=10)),
        yaxis=dict(gridcolor=BG_ELV, linecolor=BORDER, tickfont=dict(color=TXT_SEC, size=10)),
    )
    return fig

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if "cluster_label" in df.columns:
        tc = df["cluster_label"].value_counts().reset_index()
        tc.columns = ["Topic", "Count"]
        fig = px.bar(tc, x="Topic", y="Count",
                     color="Count",
                     color_continuous_scale=[[0,"#2a5e14"],[0.5,ACCENT],[1,"#a3e87a"]])
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(_dark_layout(fig, "Topic frequency"), use_container_width=True)

with chart_col2:
    sc = df["sentiment"].value_counts().reset_index()
    sc.columns = ["Sentiment", "Count"]
    color_map = {"Positive": ACCENT, "Neutral": "#4a7fa5", "Negative": "#c0392b"}
    fig2 = px.pie(sc, names="Sentiment", values="Count",
                  color="Sentiment", color_discrete_map=color_map, hole=0.45)
    fig2.update_traces(
        textinfo="label+percent",
        textfont=dict(family="Inter, sans-serif", color=TXT_PRI, size=11),
        marker=dict(line=dict(color=BG_BASE, width=2)),
    )
    st.plotly_chart(_dark_layout(fig2, "Sentiment distribution"), use_container_width=True)

# Top clusters table
if "cluster_label" in df.columns:
    with st.expander("Top clusters by article count", expanded=False):
        top = df["cluster_label"].value_counts().head(8).reset_index()
        top.columns = ["Cluster", "Articles"]
        st.dataframe(top, use_container_width=True, hide_index=True)

st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:28px 0;'>", unsafe_allow_html=True)


# ── CSV export ────────────────────────────────────────────────────────────────
csv_bytes = df.to_csv(index=False).encode("utf-8")
topic_slug = (meta.get("topic") or "digest").replace(" ", "_")
st.download_button(
    label="📁 Download CSV",
    data=csv_bytes,
    file_name=f"{topic_slug}_quickpulse.csv",
    mime="text/csv",
)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# ── clustered digest ──────────────────────────────────────────────────────────
st.markdown(f"<p style='color:{ACCENT};font-weight:700;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:20px;'>Clustered News Digest</p>", unsafe_allow_html=True)

cluster_primary   = result.get("cluster_primary_topics", {})
cluster_related   = result.get("cluster_related_topics", {})

if "cluster_label" not in df_filtered.columns or df_filtered.empty:
    st.info("No articles match the selected sentiment filters.")
    st.stop()

clusters = df_filtered.groupby("cluster_label")

for cluster_label, articles in clusters:
    cluster_id_str = str(articles["cluster_id"].iloc[0]) if "cluster_id" in articles.columns else cluster_label
    primary = cluster_primary.get(cluster_id_str, cluster_primary.get(cluster_label, []))
    related = cluster_related.get(cluster_id_str, cluster_related.get(cluster_label, []))
    lda = articles["lda_topics"].iloc[0] if "lda_topics" in articles.columns else ""

    # Cluster card header
    header_parts = []
    if lda:
        header_parts.append(f"<span style='color:#a3e87a;font-size:0.8rem;'>Themes: {lda}</span>")
    if primary:
        header_parts.append(f"<span style='color:{ACCENT};font-size:0.8rem;'>Primary: {', '.join(primary)}</span>")
    if related:
        header_parts.append(f"<span style='color:#5a6270;font-size:0.8rem;'>Related: {', '.join(related)}</span>")

    header_html = "<br>".join(header_parts) if header_parts else ""

    st.markdown(f"""
    <div style="border:1px solid {BORDER};border-radius:12px;margin-bottom:20px;padding:20px;background:{BG_CARD};">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <span style="background:rgba(101,194,58,0.12);border:1px solid rgba(101,194,58,0.3);border-radius:6px;padding:3px 10px;font-size:0.68rem;font-weight:700;color:{ACCENT};letter-spacing:0.1em;text-transform:uppercase;font-family:'JetBrains Mono',monospace;">CLUSTER</span>
        <span style="font-size:1rem;font-weight:600;color:{TXT_PRI};">{cluster_label}</span>
        <span style="font-size:0.78rem;color:{TXT_SEC};margin-left:auto;">{len(articles)} articles</span>
      </div>
      {header_html}
    </div>
    """, unsafe_allow_html=True)

    # Articles grouped by sentiment
    for sentiment, cfg in SENTIMENT_CFG.items():
        sent_articles = articles[articles["sentiment"] == sentiment]
        if sent_articles.empty:
            continue

        st.markdown(f"""
        <div style="background:{cfg['bg']};border-left:3px solid {cfg['border']};border-radius:0 8px 8px 0;margin-bottom:10px;padding:10px 14px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="width:7px;height:7px;border-radius:50%;background:{cfg['dot']};display:inline-block;"></span>
            <span style="font-size:0.78rem;font-weight:700;color:{TXT_PRI};letter-spacing:0.04em;text-transform:uppercase;">
              {sentiment} <span style="color:{cfg['dot']};margin-left:4px;">({len(sent_articles)})</span>
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        for _, art in sent_articles.iterrows():
            score_val = art.get("score", None)
            score_str = f"{float(score_val):.2f}" if score_val is not None else ""

            with st.expander(f"📰 {art['title']}", expanded=False):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"<span style='color:{TXT_SEC};font-size:0.78rem;'>Source:</span> <span style='color:{TXT_PRI};font-size:0.78rem;'>{art['source']}</span>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:{TXT_SEC};font-size:0.82rem;line-height:1.55;margin-top:8px;'>{art['summary']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<a href='{art['url']}' target='_blank' style='color:{ACCENT};font-size:0.78rem;font-weight:500;text-decoration:none;'>Read full article →</a>", unsafe_allow_html=True)
                with cols[1]:
                    if score_str:
                        st.markdown(f"<div style='text-align:right;font-family:JetBrains Mono,monospace;font-size:0.8rem;color:{cfg['dot']};background:rgba(0,0,0,0.3);border:1px solid {cfg['border']};border-radius:6px;padding:4px 8px;display:inline-block;'>{score_str}</div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:right;font-size:0.72rem;color:{TXT_SEC};margin-top:6px;'>{art.get('publishedAt','')[:10]}</p>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
