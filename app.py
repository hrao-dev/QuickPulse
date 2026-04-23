"""
QuickPulse — Streamlit Edition (Redesigned)
Editorial dark aesthetic. No HTML leaking into markdown.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go

import gather_news
import cluster_news
import summarizer
import analyze_sentiment
import extract_news

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# bg0   #07080d   page background
# bg1   #0d0f17   card / panel
# bg2   #13161f   elevated surface / input
# rim   #1c2030   borders
# acc   #6C8EFF   electric periwinkle — primary accent
# cya   #3DD9C5   teal — secondary / links
# pos   #4ADE80   green — positive
# neg   #FB7185   rose  — negative
# neu   #94A3B8   slate — neutral
# hi    #EEF0F8   primary text
# lo    #4A5568   muted text
# ─────────────────────────────────────────────────────────────────────────────

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Spline+Sans+Mono:wght@400;500;600&display=swap');

:root {
  --bg0: #07080d;
  --bg1: #0d0f17;
  --bg2: #13161f;
  --rim: #1c2030;
  --acc: #6C8EFF;
  --cya: #3DD9C5;
  --pos: #4ADE80;
  --neg: #FB7185;
  --neu: #94A3B8;
  --hi:  #EEF0F8;
  --lo:  #4A5568;
  --font: 'Outfit', sans-serif;
  --mono: 'Spline Sans Mono', monospace;
}

html, body, [class*="st-"], .stApp {
  font-family: var(--font) !important;
  background: var(--bg0) !important;
  color: var(--hi) !important;
}
.main .block-container {
  background: var(--bg0) !important;
  padding: 0 2rem 3rem !important;
  max-width: 1320px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--rim) !important;
}
[data-testid="stSidebar"] > div:first-child {
  background: var(--bg1) !important;
  padding: 2rem 1.4rem !important;
}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: var(--bg2) !important;
  border: 1px solid var(--rim) !important;
  border-radius: 10px !important;
  color: var(--hi) !important;
  font-family: var(--font) !important;
  font-size: 0.92rem !important;
  padding: 0.6rem 0.9rem !important;
  transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--acc) !important;
  box-shadow: 0 0 0 3px rgba(108,142,255,0.12) !important;
}

/* Labels */
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stMultiSelect"] label {
  font-family: var(--mono) !important;
  font-size: 0.65rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--lo) !important;
}

/* Multiselect */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
  background: rgba(108,142,255,0.15) !important;
  border: 1px solid rgba(108,142,255,0.35) !important;
  border-radius: 6px !important;
  color: var(--acc) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
}
[data-testid="stMultiSelect"] > div {
  background: var(--bg2) !important;
  border: 1px solid var(--rim) !important;
  border-radius: 10px !important;
}

/* Buttons */
.stButton > button {
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  border-radius: 10px !important;
  padding: 0.55rem 1.2rem !important;
  transition: all 0.18s ease !important;
  width: 100% !important;
}
.stButton > button[data-testid="baseButton-primary"] {
  background: var(--acc) !important;
  color: #07080d !important;
  border: none !important;
}
.stButton > button[data-testid="baseButton-primary"]:hover {
  background: #8AAAFF !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 20px rgba(108,142,255,0.25) !important;
}
.stButton > button[data-testid="baseButton-secondary"] {
  background: transparent !important;
  color: var(--neu) !important;
  border: 1px solid var(--rim) !important;
}
.stButton > button[data-testid="baseButton-secondary"]:hover {
  border-color: var(--acc) !important;
  color: var(--acc) !important;
  background: rgba(108,142,255,0.06) !important;
}

/* Expander */
details summary {
  font-family: var(--font) !important;
  font-size: 0.85rem !important;
  color: var(--neu) !important;
  background: var(--bg1) !important;
  border: 1px solid var(--rim) !important;
  border-radius: 10px !important;
}
details > div {
  background: var(--bg1) !important;
  border: 1px solid var(--rim) !important;
  border-top: none !important;
  border-radius: 0 0 10px 10px !important;
}

/* Metric cards */
[data-testid="metric-container"] {
  background: var(--bg1) !important;
  border: 1px solid var(--rim) !important;
  border-radius: 14px !important;
  padding: 1.25rem 1.4rem !important;
  position: relative;
  overflow: hidden;
}
[data-testid="metric-container"]::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--acc), var(--cya));
}
[data-testid="stMetricLabel"] p {
  font-family: var(--mono) !important;
  font-size: 0.63rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--lo) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--font) !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--hi) !important;
}

/* Charts */
.js-plotly-plot, .plot-container {
  border-radius: 14px !important;
  border: 1px solid var(--rim) !important;
  overflow: hidden !important;
  background: var(--bg1) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
  border: 1px solid var(--rim) !important;
  border-radius: 14px !important;
  overflow: hidden;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg0); }
::-webkit-scrollbar-thumb { background: var(--rim); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--acc); }

/* Misc */
.stSpinner > div { border-top-color: var(--acc) !important; }
.stAlert { background: var(--bg1) !important; border: 1px solid var(--rim) !important; border-radius: 10px !important; }
hr { border: none !important; border-top: 1px solid var(--rim) !important; margin: 2rem 0 !important; }
[data-testid="stHeader"] { background: transparent !important; }
</style>
"""

HERO_HTML = """
<div style="
  background:linear-gradient(160deg,#07080d 0%,#0d0c1c 60%,#07080d 100%);
  padding:3rem 2.5rem 2.5rem;
  margin-bottom:2rem;
  border-bottom:1px solid #1c2030;
  position:relative;overflow:hidden;
">
  <div style="position:absolute;top:-80px;left:50%;transform:translateX(-50%);
    width:600px;height:240px;
    background:radial-gradient(ellipse,rgba(108,142,255,0.12) 0%,transparent 68%);
    pointer-events:none;"></div>
  <div style="position:absolute;bottom:-40px;right:10%;
    width:300px;height:200px;
    background:radial-gradient(ellipse,rgba(61,217,197,0.07) 0%,transparent 70%);
    pointer-events:none;"></div>

  <div style="margin-bottom:1.2rem;">
    <span style="
      display:inline-flex;align-items:center;gap:7px;
      background:rgba(108,142,255,0.1);border:1px solid rgba(108,142,255,0.25);
      border-radius:100px;padding:4px 14px;">
      <span style="width:6px;height:6px;border-radius:50%;background:#6C8EFF;
        display:inline-block;box-shadow:0 0 8px #6C8EFF;
        animation:hpulse 2.2s ease-in-out infinite;"></span>
      <span style="font-family:'Spline Sans Mono',monospace;font-size:0.62rem;
        font-weight:500;color:#6C8EFF;letter-spacing:0.15em;text-transform:uppercase;">
        live · multi-source</span>
    </span>
  </div>

  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:0.75rem;">
    <h1 style="font-family:'Outfit',sans-serif;font-weight:800;
      font-size:clamp(2.4rem,5vw,3.4rem);letter-spacing:-0.04em;
      margin:0;color:#EEF0F8;">QuickPulse</h1>
    <span style="font-family:'Spline Sans Mono',monospace;font-size:0.72rem;
      color:#1c2030;padding-bottom:0.4rem;">v2</span>
  </div>

  <p style="font-family:'Outfit',sans-serif;font-size:1rem;color:#94A3B8;
    max-width:520px;margin:0;line-height:1.65;">
    Fetch, cluster &amp; analyze live news —
    <span style="color:#3DD9C5;">sentiment-scored</span>,
    <span style="color:#6C8EFF;">topic-clustered</span>, AI-summarized.
  </p>

  <style>
    @keyframes hpulse {
      0%,100% { opacity:1; box-shadow:0 0 8px #6C8EFF; }
      50%      { opacity:0.45; box-shadow:0 0 2px #6C8EFF; }
    }
  </style>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

_BASE_LAYOUT = dict(
    paper_bgcolor="#0d0f17",
    plot_bgcolor="#07080d",
    font=dict(family="Outfit, sans-serif", color="#4A5568", size=12),
    title_font=dict(family="Outfit, sans-serif", color="#EEF0F8", size=14),
    margin=dict(l=16, r=16, t=48, b=16),
    xaxis=dict(gridcolor="#13161f", linecolor="#1c2030", tickfont=dict(color="#4A5568", size=10)),
    yaxis=dict(gridcolor="#13161f", linecolor="#1c2030", tickfont=dict(color="#4A5568", size=10)),
)


def plot_topic_bar(result):
    df = result["dataframe"]
    tc = df["cluster_label"].value_counts().reset_index()
    tc.columns = ["Topic", "Count"]
    tc["Label"] = tc["Topic"].apply(lambda x: x[:34] + "…" if len(x) > 34 else x)
    fig = go.Figure(go.Bar(
        y=tc["Label"], x=tc["Count"], orientation="h",
        marker=dict(
            color=tc["Count"],
            colorscale=[[0, "#1a2050"], [0.5, "#6C8EFF"], [1, "#3DD9C5"]],
            showscale=False, line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>%{x} articles<extra></extra>",
    ))
    fig.update_layout(
        title="Topic Clusters",
        height=max(240, len(tc) * 38 + 80),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_sentiment_donut(result):
    df = result["dataframe"]
    sc = df["sentiment"].str.capitalize().value_counts().reset_index()
    sc.columns = ["Sentiment", "Count"]
    cmap = {"Positive": "#4ADE80", "Neutral": "#94A3B8", "Negative": "#FB7185"}
    colors = [cmap.get(s, "#6C8EFF") for s in sc["Sentiment"]]
    total = int(sc["Count"].sum())
    fig = go.Figure(go.Pie(
        labels=sc["Sentiment"], values=sc["Count"], hole=0.62,
        marker=dict(colors=colors, line=dict(color="#07080d", width=3)),
        textinfo="label+percent",
        textfont=dict(family="Outfit, sans-serif", color="#EEF0F8", size=11),
        hovertemplate="<b>%{label}</b><br>%{value} articles<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{total}</b><br>articles",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family="Outfit, sans-serif", color="#EEF0F8", size=13),
    )
    fig.update_layout(
        title="Sentiment",
        height=300,
        legend=dict(bgcolor="#0d0f17", bordercolor="#1c2030", borderwidth=1,
                    font=dict(color="#94A3B8", size=11)),
        **{k: v for k, v in _BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DIGEST — full self-contained HTML rendered via components.html
# This bypasses Streamlit's markdown sanitizer entirely, so no raw HTML leaks.
# ─────────────────────────────────────────────────────────────────────────────

def _esc(s):
    """Minimal HTML escape for user-derived strings."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_digest_html(result, sentiment_filters=None):
    df = result["dataframe"].copy()
    cluster_primary = result.get("cluster_primary_topics", {})
    cluster_related = result.get("cluster_related_topics", {})
    df["sentiment"] = df["sentiment"].str.capitalize()

    if sentiment_filters:
        df = df[df["sentiment"].isin(sentiment_filters)]
    if df.empty:
        body = "<p style='color:#4A5568;padding:2rem;'>No articles match the selected filters.</p>"
        return _wrap_html(body), 120

    SENT_CFG = {
        "Positive": {"bar": "#4ADE80", "bg": "rgba(74,222,128,0.06)",  "tbg": "rgba(74,222,128,0.12)",  "tbr": "rgba(74,222,128,0.3)"},
        "Neutral":  {"bar": "#94A3B8", "bg": "rgba(148,163,184,0.04)", "tbg": "rgba(148,163,184,0.1)",  "tbr": "rgba(148,163,184,0.25)"},
        "Negative": {"bar": "#FB7185", "bg": "rgba(251,113,133,0.06)", "tbg": "rgba(251,113,133,0.12)", "tbr": "rgba(251,113,133,0.3)"},
    }

    def pill(text, color, bg, border):
        return (f'<span style="display:inline-block;background:{bg};border:1px solid {border};'
                f'border-radius:6px;padding:2px 9px;font-size:0.63rem;font-weight:500;'
                f'color:{color};letter-spacing:0.03em;margin:2px 3px 2px 0;">{_esc(text)}</span>')

    cards = []
    for cluster_label, arts in df.groupby("cluster_label"):
        lda     = arts["lda_topics"].iloc[0] if "lda_topics" in arts else ""
        primary = cluster_primary.get(cluster_label, [])
        related = cluster_related.get(cluster_label, [])

        primary_pills = "".join(pill(t, "#6C8EFF", "rgba(108,142,255,0.12)", "rgba(108,142,255,0.3)") for t in primary)
        related_pills = "".join(pill(t, "#4A5568", "rgba(74,85,104,0.1)",    "rgba(74,85,104,0.2)")   for t in related)
        all_pills     = primary_pills + related_pills

        sent_sections = []
        for sent_label, cfg in SENT_CFG.items():
            sent_arts = arts[arts["sentiment"] == sent_label]
            if sent_arts.empty:
                continue

            article_cards = []
            for _, art in sent_arts.iterrows():
                score_val = art.get("score")
                score_badge = ""
                try:
                    sv = float(score_val)
                    score_badge = (
                        f'<span style="flex-shrink:0;font-family:Spline Sans Mono,monospace;'
                        f'font-size:0.63rem;font-weight:600;color:{cfg["bar"]};'
                        f'background:{cfg["tbg"]};border:1px solid {cfg["tbr"]};'
                        f'border-radius:6px;padding:2px 8px;white-space:nowrap;">{sv:.2f}</span>'
                    )
                except (ValueError, TypeError, AttributeError):
                    pass

                article_cards.append(f"""
<div class="acard" style="background:#07080d;border:1px solid #1c2030;border-radius:10px;
  padding:14px 16px;margin-bottom:10px;">
  <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;">
    <p style="font-family:Outfit,sans-serif;font-size:0.88rem;font-weight:600;
      color:#EEF0F8;line-height:1.4;margin:0;flex:1;">{_esc(art.get("title","Untitled"))}</p>
    {score_badge}
  </div>
  <p style="font-family:Outfit,sans-serif;font-size:0.73rem;color:#4A5568;margin:0 0 10px;">
    {_esc(art.get("source","Unknown"))}
  </p>
  <details style="margin-bottom:10px;">
    <summary style="font-family:Outfit,sans-serif;font-size:0.78rem;font-weight:500;
      color:#6C8EFF;cursor:pointer;list-style:none;
      display:inline-flex;align-items:center;gap:5px;">
      <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
        <path d="M2 3.5L5 6.5L8 3.5" stroke="#6C8EFF" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
      Summary
    </summary>
    <p style="font-family:Outfit,sans-serif;font-size:0.82rem;color:#94A3B8;
      line-height:1.65;margin:10px 0 0;padding:10px 12px;
      background:#0d0f17;border-left:2px solid {cfg["bar"]};
      border-radius:0 6px 6px 0;">{_esc(art.get("summary",""))}</p>
  </details>
  <a href="{_esc(art.get("url","#"))}" target="_blank"
    style="font-family:Outfit,sans-serif;font-size:0.75rem;font-weight:600;
    color:#3DD9C5;text-decoration:none;display:inline-flex;align-items:center;gap:4px;">
    Read article
    <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
      <path d="M2 8L8 2M8 2H4M8 2V6" stroke="#3DD9C5" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </a>
</div>""")

            sent_sections.append(f"""
<div style="background:{cfg["bg"]};border-left:2px solid {cfg["bar"]};
  border-radius:0 10px 10px 0;padding:14px 14px 4px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
    <span style="width:6px;height:6px;border-radius:50%;background:{cfg["bar"]};
      box-shadow:0 0 6px {cfg["bar"]};display:inline-block;flex-shrink:0;"></span>
    <span style="font-family:Spline Sans Mono,monospace;font-size:0.63rem;font-weight:600;
      color:{cfg["bar"]};letter-spacing:0.1em;text-transform:uppercase;">{sent_label}</span>
    <span style="margin-left:auto;font-family:Outfit,sans-serif;font-size:0.72rem;color:#4A5568;">
      {len(sent_arts)} article{"s" if len(sent_arts) != 1 else ""}
    </span>
  </div>
  {"".join(article_cards)}
</div>""")

        cards.append(f"""
<div style="background:#0d0f17;border:1px solid #1c2030;border-radius:14px;
  padding:22px 24px;margin-bottom:16px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,#6C8EFF,#3DD9C5);opacity:0.55;"></div>
  <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:14px;flex-wrap:wrap;">
    <span style="font-family:Spline Sans Mono,monospace;font-size:0.58rem;font-weight:600;
      color:#6C8EFF;letter-spacing:0.14em;text-transform:uppercase;
      background:rgba(108,142,255,0.1);border:1px solid rgba(108,142,255,0.25);
      border-radius:6px;padding:3px 9px;white-space:nowrap;margin-top:2px;">cluster</span>
    <span style="font-family:Outfit,sans-serif;font-size:1rem;font-weight:700;
      color:#EEF0F8;line-height:1.3;flex:1;">{_esc(cluster_label)}</span>
    <span style="font-family:Outfit,sans-serif;font-size:0.75rem;color:#4A5568;white-space:nowrap;">
      {len(arts)} articles</span>
  </div>
  {f'<div style="margin-bottom:14px;line-height:1.9;">{all_pills}</div>' if all_pills else ""}
  {f'<p style="font-family:Outfit,sans-serif;font-size:0.75rem;color:#3DD9C5;margin:0 0 14px;opacity:0.7;">{_esc(lda)}</p>' if lda else ""}
  {"".join(sent_sections)}
</div>""")

    n_clusters = df["cluster_label"].nunique()
    n_articles = len(df)
    est_height = max(400, n_clusters * 340 + n_articles * 130)
    return _wrap_html("\n".join(cards)), est_height


def _wrap_html(body):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Spline+Sans+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#07080d;font-family:Outfit,sans-serif;color:#EEF0F8;padding:4px 2px 16px;}}
  ::-webkit-scrollbar{{width:4px;}}
  ::-webkit-scrollbar-track{{background:#07080d;}}
  ::-webkit-scrollbar-thumb{{background:#1c2030;border-radius:2px;}}
  details>summary{{list-style:none;}}
  details>summary::-webkit-details-marker{{display:none;}}
  .acard{{transition:border-color 0.2s;}}
  .acard:hover{{border-color:#6C8EFF !important;}}
</style>
</head>
<body>{body}</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND
# ─────────────────────────────────────────────────────────────────────────────

def extract_summarize_and_analyze_articles(articles):
    extracted = []
    for article in articles:
        content = article.get("text") or article.get("content")
        if not content:
            continue
        summary   = summarizer.generate_summary(content)
        sentiment, score = analyze_sentiment.analyze_summary(summary)
        extracted.append({
            "title":       article.get("title", "No title"),
            "url":         article.get("url"),
            "source":      article.get("source", "Unknown"),
            "author":      article.get("author", "Unknown"),
            "publishedAt": article.get("publishedAt", "Unknown"),
            "content":     content,
            "summary":     summary,
            "sentiment":   sentiment,
            "score":       score,
        })
    return extracted


def deduplicate_articles(articles):
    seen_urls, seen_ts, seen_tsumm = set(), set(), set()
    deduped = []
    for art in articles:
        url   = art.get("url")
        title = art.get("title", "").strip().lower()
        src   = art.get("source", "").strip().lower()
        summ  = art.get("summary", "").strip().lower()
        if url and url in seen_urls:         continue
        if (title, src)  in seen_ts:         continue
        if (title, summ) in seen_tsumm:      continue
        deduped.append(art)
        if url: seen_urls.add(url)
        seen_ts.add((title, src))
        seen_tsumm.add((title, summ))
    return deduped


def run_pipeline(articles, sentiment_filters):
    if not articles:
        return None
    articles  = sorted(articles, key=lambda x: x.get("publishedAt", ""), reverse=True)
    extracted = extract_summarize_and_analyze_articles(articles)
    deduped   = deduplicate_articles(extracted)
    if not deduped:
        return None
    df = pd.DataFrame(deduped)
    return cluster_news.cluster_and_label_articles(df, content_column="content", summary_column="summary")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuickPulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(PAGE_CSS,  unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)

# Session state
for key in ("result", "active_filters"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<p style="font-family:'Spline Sans Mono',monospace;font-size:0.62rem;
      font-weight:500;color:#4A5568;letter-spacing:0.12em;text-transform:uppercase;
      margin-bottom:1.5rem;">Controls</p>""", unsafe_allow_html=True)

    topic_input = st.text_input("Topic", placeholder="e.g. quantum computing",
                                 help="Leave blank for top headlines.")

    sentiment_filters = st.multiselect(
        "Sentiment Filter",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )

    with st.expander("Batch URL input"):
        urls_input = st.text_area("URLs — one per line", height=100,
                                   placeholder="https://…\nhttps://…")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    run_btn       = st.button("⚡  Generate Digest", use_container_width=True, type="primary")
    headlines_btn = st.button("📡  Top Headlines",   use_container_width=True, type="secondary")
    clear_btn     = st.button("✕  Clear",            use_container_width=True, type="secondary")

    st.markdown("""<div style="margin-top:2.5rem;padding-top:1.5rem;border-top:1px solid #1c2030;">
      <p style="font-family:'Spline Sans Mono',monospace;font-size:0.58rem;
        color:#1c2030;line-height:2;letter-spacing:0.04em;">
        POWERED BY<br>NewsAPI · HDBSCAN<br>FlanT5 · BART-MNLI<br>sentence-transformers
      </p></div>""", unsafe_allow_html=True)

# ── Actions ───────────────────────────────────────────────────────────────────
if clear_btn:
    st.session_state.result         = None
    st.session_state.active_filters = None
    st.rerun()

articles = []

if run_btn:
    if topic_input and topic_input.strip():
        with st.spinner("Fetching articles…"):
            articles = gather_news.fetch_newsapi_everything(topic_input)
    elif urls_input and urls_input.strip():
        url_list = [u.strip() for u in urls_input.splitlines() if u.strip()]
        with st.spinner(f"Extracting {len(url_list)} URLs…"):
            articles = extract_news.extract_news_articles(url_list)
    else:
        st.warning("Enter a topic or paste URLs to get started.")
    if articles:
        with st.spinner("Summarizing & clustering…"):
            st.session_state.result         = run_pipeline(articles, sentiment_filters)
            st.session_state.active_filters = sentiment_filters

if headlines_btn:
    with st.spinner("Fetching top headlines…"):
        articles = gather_news.fetch_newsapi_top_headlines()
    if articles:
        with st.spinner("Summarizing & clustering…"):
            st.session_state.result         = run_pipeline(articles, sentiment_filters)
            st.session_state.active_filters = sentiment_filters

# ── Results ───────────────────────────────────────────────────────────────────
result  = st.session_state.result
filters = st.session_state.active_filters or sentiment_filters

if result is not None:
    df_res    = result["dataframe"]
    n_art     = len(df_res)
    n_clust   = result["number_of_clusters"]
    pos_pct   = round(100 * (df_res["sentiment"].str.capitalize() == "Positive").sum() / max(n_art, 1))
    neg_pct   = round(100 * (df_res["sentiment"].str.capitalize() == "Negative").sum() / max(n_art, 1))

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Articles",  n_art)
    k2.metric("Clusters",  n_clust)
    k3.metric("Positive",  f"{pos_pct}%")
    k4.metric("Negative",  f"{neg_pct}%")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Charts — 3:2 split so bar has room for labels
    col_bar, col_donut = st.columns([3, 2])
    with col_bar:
        st.plotly_chart(plot_topic_bar(result),      use_container_width=True, config={"displayModeBar": False})
    with col_donut:
        st.plotly_chart(plot_sentiment_donut(result), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr>", unsafe_allow_html=True)

    # Section label
    st.markdown("""<p style="font-family:'Spline Sans Mono',monospace;font-size:0.65rem;
      font-weight:600;color:#6C8EFF;letter-spacing:0.14em;text-transform:uppercase;
      margin-bottom:1rem;">Clustered News Digest</p>""", unsafe_allow_html=True)

    # Digest — rendered in an iframe-like component to avoid markdown sanitizer
    digest_html, est_height = build_digest_html(result, filters)
    components.html(digest_html, height=est_height, scrolling=True)

else:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
      padding:5rem 2rem;border:1px dashed #1c2030;border-radius:16px;
      background:#0d0f17;margin-top:0.5rem;text-align:center;">
      <div style="width:56px;height:56px;border-radius:14px;
        background:rgba(108,142,255,0.1);border:1px solid rgba(108,142,255,0.2);
        display:flex;align-items:center;justify-content:center;
        font-size:1.6rem;margin-bottom:1.25rem;">⚡</div>
      <p style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;
        color:#EEF0F8;margin-bottom:0.5rem;">Ready to pulse</p>
      <p style="font-family:'Outfit',sans-serif;font-size:0.88rem;color:#4A5568;
        max-width:380px;line-height:1.65;">
        Type a topic in the sidebar, paste URLs, or hit
        <span style="color:#94A3B8;font-weight:500;">Top Headlines</span>
        to analyze the latest news.
      </p>
    </div>""", unsafe_allow_html=True)
