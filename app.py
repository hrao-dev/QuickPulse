# app.py  —  QuickPulse · Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
# Key fixes vs previous version:
#   1. Pipeline status is written to cache/status.json so the UI can show
#      "running / last error / last success" even across page reloads.
#   2. Scheduler guard uses a PID file instead of a module-level bool so it
#      survives Streamlit's multi-worker restarts.
#   3. Button clicks call pipeline.run() synchronously with a live st.status()
#      log so the user sees exactly what's happening instead of a blank spinner.
#   4. All Streamlit widget colours are overridden with targeted CSS selectors
#      that reach the shadow DOM — fixes dark-on-dark text in sidebar.
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler

import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
ACCENT   = "#65C23A"
BG_BASE  = "#0d0f12"
BG_CARD  = "#13161b"
BG_ELV   = "#1a1e26"
BORDER   = "#252932"
TXT_PRI  = "#e8eaf0"
TXT_SEC  = "#9aa0ad"

SENTIMENT_CFG = {
    "Positive": {"dot": "#65C23A", "bg": "#0d1f0a", "border": "#3a7d1e"},
    "Neutral":  {"dot": "#4a7fa5", "bg": "#0e1520", "border": "#2a5298"},
    "Negative": {"dot": "#c0392b", "bg": "#1f0d0d", "border": "#8b1a1a"},
}

STATUS_FILE = Path("cache/pipeline_status.json")
LOCK_FILE   = Path("cache/scheduler.lock")
Path("cache").mkdir(exist_ok=True)


# ── pipeline status helpers ───────────────────────────────────────────────────
def _write_status(state: str, message: str = ""):
    STATUS_FILE.write_text(json.dumps({
        "state": state,
        "message": message,
        "ts": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")

def _read_status() -> dict:
    try:
        if STATUS_FILE.exists():
            return json.loads(STATUS_FILE.read_text())
    except Exception:
        pass
    return {"state": "idle", "message": "", "ts": ""}


# ── health-check server (keep HF Space warm) ──────────────────────────────────
class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *_): pass

_health_started = False
def _start_health_server():
    global _health_started
    if _health_started:
        return
    _health_started = True
    try:
        srv = HTTPServer(("0.0.0.0", 8080), _H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        logger.info("Health server :8080 started")
    except Exception as e:
        logger.warning(f"Health server: {e}")


# ── scheduler (PID-file guard) ────────────────────────────────────────────────
def _start_scheduler():
    pid = str(os.getpid())
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text().strip() == pid:
            return
        LOCK_FILE.write_text(pid)
    except Exception:
        pass

    def _job():
        logger.info("Scheduler: auto-refresh starting")
        _write_status("running", "Auto-refresh in progress…")
        try:
            result = pipeline.run()
            if result:
                n = result.get("meta", {}).get("article_count", len(result.get("articles", [])))
                nc = result.get("number_of_clusters", 0)
                _write_status("done", f"{n} articles · {nc} clusters")
            else:
                _write_status("error", "Pipeline returned empty — check api_key secret in HF Space settings")
        except Exception as e:
            _write_status("error", str(e))
            logger.error(f"Scheduler failed: {e}")

    threading.Thread(target=_job, daemon=True).start()
    sched = BackgroundScheduler()
    sched.add_job(_job, "interval", minutes=30)
    sched.start()
    logger.info("APScheduler started — refresh every 30 min")

_start_health_server()
_start_scheduler()


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuickPulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body {{ background-color:{BG_BASE} !important; }}
.stApp, .stApp > header {{ background-color:{BG_BASE} !important; }}
.block-container {{ padding:2rem 2.5rem 4rem; max-width:1400px; font-family:'Inter',sans-serif; }}

/* ── ALL text visible on dark bg ── */
p, span, div, label, li, h1, h2, h3, h4, h5,
.stMarkdown, [data-testid="stMarkdownContainer"] p {{
    color:{TXT_PRI} !important;
}}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background-color:{BG_CARD} !important;
    border-right:1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color:{TXT_PRI} !important; }}
[data-testid="stSidebar"] .stRadio label span p {{ color:{TXT_PRI} !important; font-size:0.88rem !important; }}
[data-testid="stSidebar"] .stRadio > label {{ color:{TXT_SEC} !important; font-size:0.72rem !important; font-weight:700 !important; text-transform:uppercase !important; letter-spacing:0.07em !important; }}
[data-testid="stSidebar"] [data-baseweb="radio"] label {{ color:{TXT_PRI} !important; font-size:0.88rem !important; font-weight:400 !important; }}
[data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"] {{ border-color:{BORDER} !important; }}
[data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"][aria-checked="true"] {{ background:{ACCENT} !important; border-color:{ACCENT} !important; }}
[data-testid="stSidebar"] .stMultiSelect > label {{ color:{TXT_SEC} !important; font-size:0.72rem !important; font-weight:700 !important; }}
[data-testid="stSidebar"] .stTextInput > label {{ color:{TXT_SEC} !important; font-size:0.72rem !important; font-weight:700 !important; text-transform:uppercase !important; letter-spacing:0.07em !important; }}
[data-testid="stSidebar"] .stTextArea > label {{ color:{TXT_SEC} !important; font-size:0.72rem !important; font-weight:700 !important; }}

/* ── inputs ── */
.stTextInput input, .stTextArea textarea {{
    background:{BG_ELV} !important; color:{TXT_PRI} !important;
    border:1px solid {BORDER} !important; border-radius:8px !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color:{ACCENT} !important;
    box-shadow:0 0 0 2px rgba(101,194,58,0.18) !important;
    outline:none !important;
}}

/* ── multiselect ── */
[data-baseweb="select"] > div {{
    background:{BG_ELV} !important; border:1px solid {BORDER} !important;
    border-radius:8px !important;
}}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="select"] span {{ color:{TXT_PRI} !important; }}
[data-baseweb="tag"] {{
    background:rgba(101,194,58,0.15) !important;
    border:1px solid rgba(101,194,58,0.35) !important;
    color:{ACCENT} !important; border-radius:6px !important;
}}
[data-baseweb="menu"] {{
    background:{BG_CARD} !important; border:1px solid {BORDER} !important;
}}
[data-baseweb="menu"] li {{ color:{TXT_PRI} !important; }}
[data-baseweb="menu"] li:hover {{ background:{BG_ELV} !important; }}

/* ── buttons ── */
.stButton > button {{
    background:{ACCENT} !important; color:{BG_BASE} !important;
    border:none !important; border-radius:8px !important;
    font-weight:700 !important; font-size:0.84rem !important;
    padding:0.45rem 1rem !important; width:100%;
    transition:opacity 0.15s;
}}
.stButton > button:hover {{ opacity:0.85 !important; }}
.stDownloadButton > button {{
    background:transparent !important; color:{ACCENT} !important;
    border:1px solid {ACCENT} !important; border-radius:8px !important;
    font-weight:600 !important;
}}

/* ── metrics ── */
[data-testid="metric-container"] {{
    background:{BG_CARD}; border:1px solid {BORDER};
    border-radius:12px; padding:1rem 1.2rem;
}}
[data-testid="metric-container"] label {{
    color:{TXT_SEC} !important; font-size:0.7rem !important;
    font-weight:700 !important; letter-spacing:0.08em !important; text-transform:uppercase !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color:{TXT_PRI} !important; font-size:1.55rem !important; font-weight:700 !important;
}}

/* ── expanders ── */
[data-testid="stExpander"] {{
    background:{BG_CARD} !important; border:1px solid {BORDER} !important;
    border-radius:10px !important; margin-bottom:6px !important;
}}
[data-testid="stExpander"] summary {{ color:{TXT_PRI} !important; font-weight:500 !important; font-size:0.88rem !important; }}
[data-testid="stExpander"] summary:hover {{ color:{ACCENT} !important; }}
[data-testid="stExpander"] svg {{ fill:{TXT_SEC} !important; }}

/* ── dataframe ── */
[data-testid="stDataFrame"] th {{
    background:{BG_ELV} !important; color:{ACCENT} !important;
    font-size:0.7rem !important; font-weight:700 !important;
    text-transform:uppercase !important; letter-spacing:0.06em !important;
}}
[data-testid="stDataFrame"] td {{
    color:{TXT_SEC} !important; font-size:0.83rem !important;
}}

/* ── status indicator ── */
[data-testid="stStatusWidget"] {{ border-radius:10px !important; }}

/* ── captions ── */
.stCaption, [data-testid="stCaptionContainer"] p {{ color:{TXT_SEC} !important; font-size:0.75rem !important; }}

/* ── alerts ── */
[data-testid="stAlert"] {{ border-radius:10px !important; }}
[data-testid="stAlert"] p {{ color:{TXT_PRI} !important; }}

hr {{ border-color:{BORDER} !important; opacity:1 !important; }}

::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{BG_BASE}; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px; }}

@keyframes qp-pulse {{
    0%,100% {{ opacity:1; box-shadow:0 0 6px {ACCENT}; }}
    50%      {{ opacity:0.5; box-shadow:0 0 2px {ACCENT}; }}
}}
.qp-dot {{
    display:inline-block; width:7px; height:7px; border-radius:50%;
    background:{ACCENT}; box-shadow:0 0 6px {ACCENT};
    animation:qp-pulse 2s ease-in-out infinite; vertical-align:middle;
}}
</style>
""", unsafe_allow_html=True)


# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:44px 24px 32px;
            background:linear-gradient(160deg,{BG_BASE} 0%,#0e1420 100%);
            border-radius:16px;margin-bottom:32px;border:1px solid {BORDER};">
  <div style="display:inline-flex;align-items:center;gap:8px;
              background:rgba(101,194,58,0.09);border:1px solid rgba(101,194,58,0.22);
              border-radius:20px;padding:5px 14px;margin-bottom:20px;">
    <span class="qp-dot"></span>
    <span style="font-size:0.7rem;font-weight:700;color:{ACCENT};letter-spacing:0.14em;
                 text-transform:uppercase;font-family:'JetBrains Mono',monospace;">live · multi-source</span>
  </div>
  <h1 style="margin:0 0 8px;font-size:clamp(1.8rem,4vw,2.6rem);font-weight:700;
             color:{TXT_PRI};letter-spacing:-0.02em;font-family:'Inter',sans-serif;">QuickPulse</h1>
  <p style="margin:0 auto 6px;max-width:500px;font-size:0.95rem;color:{ACCENT};font-weight:500;">
    Fetch · Summarise · Cluster — live news, instantly.</p>
  <p style="margin:0 auto;max-width:560px;font-size:0.84rem;color:{TXT_SEC};line-height:1.65;">
    FLAN-T5 summarisation &nbsp;·&nbsp; HDBSCAN topic clustering &nbsp;·&nbsp;
    Zero-shot sentiment &nbsp;·&nbsp; CSV export</p>
</div>
""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;
                padding-bottom:14px;border-bottom:1px solid {BORDER};">
      <span class="qp-dot"></span>
      <span style="font-size:0.72rem;font-weight:700;color:{ACCENT};letter-spacing:0.12em;
                   text-transform:uppercase;font-family:'JetBrains Mono',monospace;">QuickPulse</span>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Source mode",
        ["Top Headlines", "Search by Topic", "Custom URLs"],
        index=0,
    )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    topic_input = ""
    urls_input  = ""

    if mode == "Search by Topic":
        topic_input = st.text_input("Topic", placeholder="e.g. climate change, AI, economy")

    if mode == "Custom URLs":
        urls_input = st.text_area(
            "URLs — one per line", height=110,
            placeholder="https://example.com/article\nhttps://..."
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    sentiment_filters = st.multiselect(
        "Sentiment filter",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    run_btn     = st.button("⚡  Generate Digest",        use_container_width=True)
    refresh_btn = st.button("↻   Refresh Top Headlines",  use_container_width=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    clear_btn   = st.button("✕   Clear results",          use_container_width=True)

    # ── pipeline status card ──
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    status  = _read_status()
    state   = status.get("state", "idle")
    msg     = status.get("message", "")
    ts_raw  = status.get("ts", "")
    ts_str  = ""
    if ts_raw:
        try:
            ts_str = datetime.fromisoformat(ts_raw).strftime("%H:%M UTC")
        except Exception:
            ts_str = ts_raw[:16]

    c_map = {"running": "#ba7517", "done": ACCENT, "error": "#c23a3a", "idle": TXT_SEC}
    l_map = {"running": "⏳  Running…", "done": "✓  Up to date", "error": "⚠  Error", "idle": "—  Not started yet"}
    st.markdown(f"""
    <div style="background:{BG_ELV};border:1px solid {BORDER};border-radius:10px;padding:11px 13px;">
      <p style="margin:0 0 4px;font-size:0.68rem;font-weight:700;color:{TXT_SEC};
                text-transform:uppercase;letter-spacing:0.08em;">Pipeline status</p>
      <p style="margin:0;font-size:0.84rem;font-weight:600;color:{c_map.get(state, TXT_SEC)};">
        {l_map.get(state, '—')}</p>
      {f'<p style="margin:3px 0 0;font-size:0.72rem;color:{TXT_SEC};">{ts_str}</p>' if ts_str else ''}
      {f'<p style="margin:4px 0 0;font-size:0.72rem;color:{c_map.get(state,TXT_SEC)};word-break:break-word;">{msg}</p>' if msg else ''}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TXT_SEC};font-size:0.72rem;line-height:1.65;'>"
        f"Cache refreshes automatically every 30 min.<br>"
        f"Set <code style='color:{ACCENT};background:rgba(101,194,58,0.1);"
        f"padding:1px 4px;border-radius:3px;'>api_key</code> in your HF Space secrets.</p>",
        unsafe_allow_html=True,
    )


# ── session state ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None


# ── run pipeline with live feedback ──────────────────────────────────────────
def _run_pipeline(topic=None, urls=None):
    url_list = [u.strip() for u in urls.splitlines() if u.strip()] if urls else None
    _write_status("running", "Pipeline started…")

    with st.status("Running QuickPulse pipeline…", expanded=True) as s:
        try:
            import gather_news
            import extract_news as en

            st.write("📡  Fetching article metadata from NewsAPI…")
            if url_list:
                raw = en.extract_news_articles(url_list)
                st.write(f"✓  Extracted {len(raw)} articles from provided URLs")
            elif topic and topic.strip():
                raw = gather_news.fetch_newsapi_everything(topic)
                st.write(f"✓  Fetched {len(raw)} articles for topic: **{topic}**")
            else:
                raw = gather_news.fetch_newsapi_top_headlines()
                st.write(f"✓  Fetched {len(raw)} top-headline articles")

            if not raw:
                s.update(label="No articles found", state="error")
                _write_status(
                    "error",
                    "NewsAPI returned 0 articles. Verify api_key is set in HF Space secrets."
                )
                st.error(
                    "**NewsAPI returned 0 articles.**\n\n"
                    "Make sure `api_key` is added as a secret in your HF Space settings "
                    "(Settings → Variables and Secrets)."
                )
                return

            st.write(
                f"🧠  Summarising {len(raw)} articles with FLAN-T5…  "
                f"*(model downloads on first run — allow ~90 s)*"
            )
            st.write("📊  Clustering with HDBSCAN + LDA…")

            result = pipeline.run(topic=topic, urls=url_list)

            if not result or not result.get("articles"):
                s.update(label="Pipeline returned no results", state="error")
                _write_status("error", "Pipeline returned empty result — check HF Space logs")
                st.error(
                    "Pipeline completed but returned no results. "
                    "Open the HF Space Logs tab for details."
                )
                return

            n  = result.get("meta", {}).get("article_count", len(result["articles"]))
            nc = result.get("number_of_clusters", 0)
            s.update(label=f"Done — {n} articles · {nc} clusters", state="complete")
            _write_status("done", f"{n} articles · {nc} clusters")
            st.session_state.result = result

        except Exception as e:
            s.update(label=f"Error: {e}", state="error")
            _write_status("error", str(e))
            st.error(f"Pipeline error: {e}")
            logger.exception("Pipeline error in UI")


# ── button actions ────────────────────────────────────────────────────────────
if clear_btn:
    st.session_state.result = None
    st.rerun()

if refresh_btn:
    _run_pipeline()

if run_btn:
    if mode == "Search by Topic" and not topic_input.strip():
        st.sidebar.error("Enter a topic first.")
    elif mode == "Custom URLs" and not urls_input.strip():
        st.sidebar.error("Enter at least one URL.")
    else:
        _run_pipeline(
            topic=topic_input if mode == "Search by Topic" else None,
            urls=urls_input   if mode == "Custom URLs"    else None,
        )


# ── result source ─────────────────────────────────────────────────────────────
result = st.session_state.result or pipeline.load_cache()

if not result or not result.get("articles"):
    st.markdown(f"""
    <div style="text-align:center;padding:72px 24px;border:1px dashed {BORDER};
                border-radius:16px;margin-top:16px;">
      <p style="font-size:2.4rem;margin-bottom:12px;">📰</p>
      <p style="font-size:1.05rem;font-weight:600;color:{TXT_PRI};margin-bottom:8px;">No digest yet</p>
      <p style="font-size:0.88rem;color:{TXT_SEC};max-width:440px;margin:0 auto;line-height:1.7;">
        Click <strong style="color:{ACCENT};">⚡ Generate Digest</strong> in the sidebar.<br><br>
        If the pipeline status shows an error, check that your
        <code style="color:{ACCENT};background:rgba(101,194,58,0.1);padding:1px 5px;
        border-radius:3px;">api_key</code>
        secret is set under <em>Settings → Variables and Secrets</em> in your HF Space.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── build df ──────────────────────────────────────────────────────────────────
df   = pd.DataFrame(result["articles"])
df["sentiment"] = df["sentiment"].str.capitalize()
meta = result.get("meta", {})

ts_raw = meta.get("refreshed_at", "")
if ts_raw:
    try:
        ts = datetime.fromisoformat(ts_raw).strftime("%d %b %Y · %H:%M UTC")
    except Exception:
        ts = ts_raw[:16]
    st.caption(
        f"Last refreshed {ts}  ·  "
        f"{meta.get('article_count', len(df))} articles  ·  "
        f"topic: {meta.get('topic', '—')}"
    )

df_filtered = df[df["sentiment"].isin(sentiment_filters)] if sentiment_filters else df.copy()


# ── metrics ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Articles",  len(df))
m2.metric("Clusters",  result.get(
    "number_of_clusters",
    df["cluster_label"].nunique() if "cluster_label" in df.columns else 0,
))
m3.metric("Positive",  int((df["sentiment"] == "Positive").sum()))
m4.metric("Negative",  int((df["sentiment"] == "Negative").sum()))
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


# ── charts ────────────────────────────────────────────────────────────────────
def _dark(fig, title=""):
    fig.update_layout(
        height=300, margin=dict(l=16, r=16, t=40, b=16),
        title=dict(text=title, font=dict(color=TXT_PRI, size=13, family="Inter")),
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        font=dict(family="Inter,sans-serif", color=TXT_SEC, size=11),
        coloraxis_showscale=False,
        legend=dict(bgcolor=BG_ELV, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TXT_SEC, size=11)),
        xaxis=dict(gridcolor=BG_ELV, linecolor=BORDER, tickfont=dict(color=TXT_SEC, size=10)),
        yaxis=dict(gridcolor=BG_ELV, linecolor=BORDER, tickfont=dict(color=TXT_SEC, size=10)),
    )
    return fig

c1, c2 = st.columns(2)
with c1:
    if "cluster_label" in df.columns:
        tc = df["cluster_label"].value_counts().reset_index()
        tc.columns = ["Topic", "Count"]
        fig = px.bar(tc, x="Topic", y="Count", color="Count",
                     color_continuous_scale=[[0,"#2a5e14"],[0.5,ACCENT],[1,"#a3e87a"]])
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(_dark(fig, "Topic frequency"), use_container_width=True)

with c2:
    sc = df["sentiment"].value_counts().reset_index()
    sc.columns = ["Sentiment", "Count"]
    fig2 = px.pie(sc, names="Sentiment", values="Count",
                  color="Sentiment",
                  color_discrete_map={"Positive": ACCENT, "Neutral": "#4a7fa5", "Negative": "#c0392b"},
                  hole=0.45)
    fig2.update_traces(
        textinfo="label+percent",
        textfont=dict(family="Inter,sans-serif", color=TXT_PRI, size=11),
        marker=dict(line=dict(color=BG_BASE, width=2)),
    )
    st.plotly_chart(_dark(fig2, "Sentiment distribution"), use_container_width=True)

if "cluster_label" in df.columns:
    with st.expander("Top clusters by article count"):
        top = df["cluster_label"].value_counts().head(8).reset_index()
        top.columns = ["Cluster", "Articles"]
        st.dataframe(top, use_container_width=True, hide_index=True)

st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:24px 0;'>", unsafe_allow_html=True)


# ── CSV download ──────────────────────────────────────────────────────────────
slug = (meta.get("topic") or "digest").replace(" ", "_")
st.download_button(
    "📁  Download full digest as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{slug}_quickpulse.csv",
    mime="text/csv",
)
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# ── digest section header ─────────────────────────────────────────────────────
st.markdown(f"""
<p style="color:{ACCENT};font-weight:700;font-size:0.72rem;letter-spacing:0.12em;
          text-transform:uppercase;margin-bottom:20px;font-family:'JetBrains Mono',monospace;">
  ▸  Clustered News Digest
</p>
""", unsafe_allow_html=True)

cluster_primary = result.get("cluster_primary_topics", {})
cluster_related = result.get("cluster_related_topics", {})

if "cluster_label" not in df_filtered.columns or df_filtered.empty:
    st.info("No articles match the selected sentiment filters.")
    st.stop()


# ── cluster cards ─────────────────────────────────────────────────────────────
for cluster_label, arts in df_filtered.groupby("cluster_label"):
    cid  = str(arts["cluster_id"].iloc[0]) if "cluster_id" in arts.columns else str(cluster_label)
    prim = cluster_primary.get(cid, cluster_primary.get(str(cluster_label), []))
    rel  = cluster_related.get(cid, cluster_related.get(str(cluster_label), []))
    lda  = arts["lda_topics"].iloc[0] if "lda_topics" in arts.columns else ""

    badges = []
    if lda:
        badges.append(f"<span style='color:#a3e87a;'>Themes: {lda}</span>")
    if prim:
        badges.append(f"<span style='color:{ACCENT};'>Primary: {', '.join(prim)}</span>")
    if rel:
        badges.append(f"<span style='color:#5a6270;'>Related: {', '.join(rel)}</span>")
    badge_html = "&nbsp; · &nbsp;".join(badges) if badges else ""

    st.markdown(f"""
    <div style="border:1px solid {BORDER};border-radius:14px;margin-bottom:24px;
                padding:18px 20px 10px;background:{BG_CARD};">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="background:rgba(101,194,58,0.12);border:1px solid rgba(101,194,58,0.28);
                     border-radius:6px;padding:3px 10px;font-size:0.65rem;font-weight:700;
                     color:{ACCENT};letter-spacing:0.1em;text-transform:uppercase;
                     font-family:'JetBrains Mono',monospace;">cluster</span>
        <span style="font-size:0.96rem;font-weight:600;color:{TXT_PRI};">{cluster_label}</span>
        <span style="font-size:0.74rem;color:{TXT_SEC};margin-left:auto;">{len(arts)} articles</span>
      </div>
      <p style="font-size:0.77rem;color:{TXT_SEC};margin:0 0 2px;line-height:1.7;">{badge_html}</p>
    </div>
    """, unsafe_allow_html=True)

    for sentiment, cfg in SENTIMENT_CFG.items():
        grp = arts[arts["sentiment"] == sentiment]
        if grp.empty:
            continue

        st.markdown(f"""
        <div style="background:{cfg['bg']};border-left:3px solid {cfg['border']};
                    border-radius:0 8px 8px 0;margin-bottom:8px;padding:9px 14px;">
          <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
                       background:{cfg['dot']};vertical-align:middle;margin-right:8px;"></span>
          <span style="font-size:0.75rem;font-weight:700;color:{TXT_PRI};
                       letter-spacing:0.05em;text-transform:uppercase;">
            {sentiment}&nbsp;
            <span style="color:{cfg['dot']};font-weight:400;">({len(grp)})</span>
          </span>
        </div>
        """, unsafe_allow_html=True)

        for _, art in grp.iterrows():
            score_val = art.get("score")
            score_str = f"{float(score_val):.2f}" if score_val is not None else ""

            with st.expander(f"📰  {art['title']}"):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(
                        f"<p style='font-size:0.75rem;color:{TXT_SEC};margin:0 0 10px;'>"
                        f"<strong style='color:#5a6270;'>Source</strong> {art['source']}"
                        f"&emsp;<strong style='color:#5a6270;'>Published</strong> "
                        f"{str(art.get('publishedAt', ''))[:10]}</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<p style='color:{TXT_SEC};font-size:0.83rem;line-height:1.65;"
                        f"border-left:2px solid {BORDER};padding-left:12px;margin:0 0 12px;'>"
                        f"{art['summary']}</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<a href='{art['url']}' target='_blank' "
                        f"style='color:{ACCENT};font-size:0.78rem;font-weight:600;"
                        f"text-decoration:none;'>Read full article →</a>",
                        unsafe_allow_html=True,
                    )
                with right:
                    if score_str:
                        st.markdown(
                            f"<div style='text-align:center;font-family:JetBrains Mono,monospace;"
                            f"font-size:1.05rem;font-weight:700;color:{cfg['dot']};"
                            f"background:rgba(0,0,0,0.22);border:1px solid {cfg['border']};"
                            f"border-radius:8px;padding:10px 6px;margin-top:2px;'>"
                            f"{score_str}<br>"
                            f"<span style='font-size:0.6rem;color:{TXT_SEC};font-weight:400;"
                            f"font-family:Inter,sans-serif;letter-spacing:0.04em;'>score</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
