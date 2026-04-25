"""
QuickPulse
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
# DESIGN TOKENS  (matched 1-to-1 from mockup)
# bg0   #07090f   page / deepest background
# bg1   #0a0d17   sidebar / cluster header row
# bg2   #0c0f1a   input surfaces / hover state
# rim   #131825   primary border
# rim2  #1a2030   secondary border (inputs, pills)
# acc   #00e5a0   electric green — primary accent / positive
# pur   #9d7dff   purple — cluster tag 2
# amb   #ffb347   amber  — cluster tag 3
# blu   #6ca0ff   blue   — cluster tag 4
# neg   #ff6b6b   red    — negative sentiment
# lo    #6b7d99   muted text
# dim   #2e3a50   very muted / labels
# hi    #dde4f0   primary text
# mid   #b0bcd4   secondary text
# ─────────────────────────────────────────────────────────────────────────────

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg0:  #07090f;
  --bg1:  #0a0d17;
  --bg2:  #0c0f1a;
  --rim:  #131825;
  --rim2: #1a2030;
  --acc:  #00e5a0;
  --pur:  #9d7dff;
  --amb:  #ffb347;
  --blu:  #6ca0ff;
  --neg:  #ff6b6b;
  --lo:   #6b7d99;
  --dim:  #2e3a50;
  --hi:   #dde4f0;
  --mid:  #b0bcd4;
  --font: 'Inter', sans-serif;
  --mono: 'JetBrains Mono', monospace;
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

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg0) !important;
  border-right: 1px solid var(--rim) !important;
}
[data-testid="stSidebar"] > div:first-child {
  background: var(--bg0) !important;
  padding: 24px 18px !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: var(--bg2) !important;
  border: 1px solid #2e3a50 !important;
  border-radius: 8px !important;
  color: var(--hi) !important;
  font-family: var(--font) !important;
  font-size: 0.85rem !important;
  padding: 8px 11px !important;
  transition: border-color 0.15s;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: rgba(0,229,160,0.4) !important;
  box-shadow: none !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
  color: var(--dim) !important;
}

/* ── Labels ── */
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stMultiSelect"] label {
  font-family: var(--mono) !important;
  font-size: 0.6rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--dim) !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
  border-radius: 6px !important;
  font-family: var(--mono) !important;
  font-size: 0.68rem !important;
  font-weight: 500 !important;
}
[data-testid="stMultiSelect"] > div {
  background: var(--bg2) !important;
  border: 1px solid var(--rim2) !important;
  border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  border-radius: 8px !important;
  padding: 11px 12px !important;
  transition: all 0.15s ease !important;
  width: 100% !important;
  letter-spacing: 0.01em !important;
}
.stButton > button[data-testid="baseButton-primary"] {
  background: var(--acc) !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em !important;
}
.stButton > button[data-testid="baseButton-primary"]:hover {
  background: #00ffb3 !important;
  color: #ffffff !important;
}
.stButton > button[data-testid="baseButton-primary"] p,
.stButton > button[data-testid="baseButton-primary"] span,
.stButton > button[data-testid="baseButton-primary"] div {
  color: #ffffff !important;
}
.stButton > button[data-testid="baseButton-secondary"] {
  background: transparent !important;
  color: var(--lo) !important;
  border: 1px solid var(--rim2) !important;
  text-align: left !important;
}
.stButton:first-of-type > button[data-testid="baseButton-secondary"] {
  color: #ffffff !important;
  border-color: #2e3a50 !important;
}
.stButton:first-of-type > button[data-testid="baseButton-secondary"] p,
.stButton:first-of-type > button[data-testid="baseButton-secondary"] span {
  color: #ffffff !important;
}
.stButton > button[data-testid="baseButton-secondary"]:hover {
  border-color: var(--dim) !important;
  color: var(--mid) !important;
  background: rgba(255,255,255,0.02) !important;
}

/* ── Sentiment filter pills — JS in components.html handles dynamic colors ── */
/* Base pill shape applied as fallback via CSS */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:first-of-type
  .stButton > button {
  border-radius: 20px !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  padding: 4px 10px !important;
  min-height: 0 !important;
  height: auto !important;
  line-height: 1.5 !important;
}

/* ── Expander ── */
details summary {
  font-family: var(--font) !important;
  font-size: 0.82rem !important;
  color: var(--lo) !important;
  background: var(--bg2) !important;
  border: 1px solid var(--rim2) !important;
  border-radius: 8px !important;
}
details > div {
  background: var(--bg1) !important;
  border: 1px solid var(--rim2) !important;
  border-top: none !important;
  border-radius: 0 0 8px 8px !important;
}

/* ── Charts ── */
.js-plotly-plot, .plot-container {
  border-radius: 0 !important;
  border: none !important;
  overflow: hidden !important;
  background: var(--bg0) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg0); }
::-webkit-scrollbar-thumb { background: var(--rim2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--dim); }

/* ── Misc ── */
.stSpinner > div { border-top-color: var(--acc) !important; }
.stAlert {
  background: var(--bg1) !important;
  border: 1px solid var(--rim2) !important;
  border-radius: 8px !important;
}
hr {
  border: none !important;
  border-top: 1px solid var(--rim) !important;
  margin: 0 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
div[data-testid="column"] { padding: 0 !important; }
</style>
"""

HERO_HTML = """
<div style="background:#07090f;padding:28px 0 20px;border-bottom:1px solid #131825;margin-bottom:0;">
  <div style="margin-bottom:10px;">
    <span style="display:inline-flex;align-items:center;gap:6px;
      background:rgba(0,229,160,0.07);border:1px solid rgba(0,229,160,0.18);
      border-radius:100px;padding:3px 10px;">
      <span style="width:5px;height:5px;border-radius:50%;background:#00e5a0;
        display:inline-block;animation:lp 2s ease-in-out infinite;"></span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
        font-weight:500;color:#00e5a0;letter-spacing:0.12em;text-transform:uppercase;">
        live · multi-source</span>
    </span>
  </div>
  <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:8px;">
    <h1 style="font-family:'Inter',sans-serif;font-weight:700;
      font-size:clamp(1.8rem,3.5vw,2.5rem);letter-spacing:-0.05em;margin:0;color:#fff;">
      Quick<em style="font-style:normal;color:#00e5a0;">Pulse</em></h1>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
      color:#1a2030;padding-bottom:0.2rem;">v2</span>
  </div>
  <p style="font-family:'Inter',sans-serif;font-size:0.88rem;color:#6b7d99;
    max-width:460px;margin:0;line-height:1.6;">
    Fetch, cluster &amp; analyze live news —
    <span style="color:#00e5a0;">sentiment-scored</span>,
    <span style="color:#9d7dff;">topic-clustered</span>,
    AI-summarized.
  </p>
  <style>
    @keyframes lp { 0%,100%{opacity:1} 50%{opacity:0.3} }
  </style>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_COLORS = ["#00e5a0", "#9d7dff", "#ffb347", "#6ca0ff", "#ff6b6b"]

_BASE_LAYOUT = dict(
    paper_bgcolor="#07090f",
    plot_bgcolor="#07090f",
    font=dict(family="Inter, sans-serif", color="#2e3a50", size=11),
    margin=dict(l=12, r=12, t=44, b=12),
    xaxis=dict(
        gridcolor="#0c0f1a",
        linecolor="#131825",
        tickfont=dict(color="#2e3a50", size=10),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#0c0f1a",
        linecolor="#131825",
        tickfont=dict(color="#6b7d99", size=10),
        zeroline=False,
    ),
)


def plot_topic_bar(result):
    df = result["dataframe"]
    tc = df["cluster_label"].value_counts().reset_index()
    tc.columns = ["Topic", "Count"]
    tc["Label"] = tc["Topic"].apply(lambda x: x[:36] + "…" if len(x) > 36 else x)
    bar_colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(tc))]
    fig = go.Figure(go.Bar(
        y=tc["Label"],
        x=tc["Count"],
        orientation="h",
        marker=dict(color=bar_colors, opacity=0.82, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>%{x} articles<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="<span style='font-family:JetBrains Mono,monospace;font-size:9px;"
                 "letter-spacing:0.1em;text-transform:uppercase;color:#2e3a50;'>Topic clusters</span>",
            x=0, pad=dict(l=0),
        ),
        height=max(220, len(tc) * 36 + 60),
        showlegend=False,
        bargap=0.38,
        **_BASE_LAYOUT,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showgrid=False)
    return fig


def plot_sentiment_donut(result):
    df = result["dataframe"]
    sc = df["sentiment"].str.capitalize().value_counts().reset_index()
    sc.columns = ["Sentiment", "Count"]
    cmap = {"Positive": "#00e5a0", "Neutral": "#2e3a50", "Negative": "#ff6b6b"}
    colors = [cmap.get(s, "#6b7d99") for s in sc["Sentiment"]]
    total = int(sc["Count"].sum())
    fig = go.Figure(go.Pie(
        labels=sc["Sentiment"],
        values=sc["Count"],
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#07090f", width=3)),
        textinfo="label+percent",
        textfont=dict(family="Inter, sans-serif", color="#dde4f0", size=10),
        hovertemplate="<b>%{label}</b><br>%{value} articles<extra></extra>",
        opacity=0.88,
    ))
    fig.add_annotation(
        text=f"<b>{total}</b><br>articles",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family="Inter, sans-serif", color="#ffffff", size=14),
    )
    fig.update_layout(
        title=dict(
            text="<span style='font-family:JetBrains Mono,monospace;font-size:9px;"
                 "letter-spacing:0.1em;text-transform:uppercase;color:#2e3a50;'>Sentiment</span>",
            x=0, pad=dict(l=0),
        ),
        height=260,
        legend=dict(
            bgcolor="#07090f",
            bordercolor="#131825",
            borderwidth=1,
            font=dict(color="#6b7d99", size=10, family="Inter, sans-serif"),
        ),
        **{k: v for k, v in _BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DIGEST HTML
# ─────────────────────────────────────────────────────────────────────────────

def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


_CLUSTER_TAGS = [
    {"color": "#00e5a0", "bg": "rgba(0,229,160,0.08)",   "border": "rgba(0,229,160,0.18)"},
    {"color": "#9d7dff", "bg": "rgba(122,90,255,0.10)",  "border": "rgba(157,125,255,0.20)"},
    {"color": "#ffb347", "bg": "rgba(255,179,71,0.08)",  "border": "rgba(255,179,71,0.18)"},
    {"color": "#6ca0ff", "bg": "rgba(100,160,255,0.08)", "border": "rgba(108,160,255,0.20)"},
    {"color": "#ff6b6b", "bg": "rgba(255,107,107,0.08)", "border": "rgba(255,107,107,0.18)"},
]

SENT_CFG = {
    "Positive": {
        "color": "#00e5a0",
        "sbg":   "rgba(0,229,160,0.08)",
        "sbd":   "rgba(0,229,160,0.16)",
        "lbdr":  "#00e5a0",
        "lbg":   "#0c0f1a",
    },
    "Neutral": {
        "color": "#6b7d99",
        "sbg":   "rgba(107,125,153,0.08)",
        "sbd":   "rgba(107,125,153,0.18)",
        "lbdr":  "#6b7d99",
        "lbg":   "#0c0f1a",
    },
    "Negative": {
        "color": "#ff6b6b",
        "sbg":   "rgba(255,107,107,0.10)",
        "sbd":   "rgba(255,107,107,0.18)",
        "lbdr":  "#ff6b6b",
        "lbg":   "#0c0f1a",
    },
}


def build_digest_html(result, sentiment_filters=None):
    df = result["dataframe"].copy()
    cluster_primary = result.get("cluster_primary_topics", {})
    cluster_related = result.get("cluster_related_topics", {})
    df["sentiment"] = df["sentiment"].str.capitalize()

    if sentiment_filters:
        df = df[df["sentiment"].isin(sentiment_filters)]
    if df.empty:
        body = ("<p style='color:#2e3a50;padding:2rem;font-family:Inter,sans-serif;'>"
                "No articles match the selected filters.</p>")
        return _wrap_html(body), 120

    cards = []
    for idx, (cluster_label, arts) in enumerate(df.groupby("cluster_label")):
        tag = _CLUSTER_TAGS[idx % len(_CLUSTER_TAGS)]
        primary = cluster_primary.get(cluster_label, [])
        related = cluster_related.get(cluster_label, [])

        def pill(text, c, bg, bd):
            return (f'<span style="display:inline-block;background:{bg};border:1px solid {bd};'
                    f'border-radius:5px;padding:2px 8px;font-size:0.58rem;font-weight:600;'
                    f'color:{c};letter-spacing:0.04em;margin:2px 3px 2px 0;'
                    f'font-family:JetBrains Mono,monospace;">{_esc(text)}</span>')

        primary_pills = "".join(
            pill(t, tag["color"], tag["bg"], tag["border"]) for t in primary)
        related_pills = "".join(
            pill(t, "#2e3a50", "rgba(46,58,80,0.1)", "rgba(46,58,80,0.2)") for t in related)
        all_pills = primary_pills + related_pills

        sent_sections = []
        for sent_label, cfg in SENT_CFG.items():
            sent_arts = arts[arts["sentiment"] == sent_label]
            if sent_arts.empty:
                continue

            article_rows = []
            for _, art in sent_arts.iterrows():
                score_val = art.get("score")
                score_badge = ""
                try:
                    sv = float(score_val)
                    score_badge = (
                        f'<span style="flex-shrink:0;font-family:JetBrains Mono,monospace;'
                        f'font-size:0.6rem;font-weight:700;color:{cfg["color"]};'
                        f'background:{cfg["sbg"]};border:1px solid {cfg["sbd"]};'
                        f'border-radius:5px;padding:2px 7px;white-space:nowrap;">{sv:.2f}</span>'
                    )
                except (ValueError, TypeError, AttributeError):
                    pass

                article_rows.append(f"""
<div class="arow" style="display:flex;align-items:flex-start;gap:10px;
  padding:9px 18px;border-top:1px solid #0d1120;">
  <div style="flex:1;min-width:0;">
    <p style="font-family:Inter,sans-serif;font-size:0.75rem;color:#dde4f0;
      line-height:1.5;margin:0 0 3px;">{_esc(art.get("title","Untitled"))}</p>
    <p style="font-family:Inter,sans-serif;font-size:0.65rem;color:#6b7d99;margin:0 0 5px;">
      {_esc(art.get("source","Unknown"))}</p>
    <details style="margin-bottom:5px;">
      <summary style="font-family:Inter,sans-serif;font-size:0.65rem;font-weight:500;
        color:{cfg["color"]};cursor:pointer;list-style:none;
        display:inline-flex;align-items:center;gap:4px;
        background:none;border:none;border-radius:0;padding:0;">
        <svg width="9" height="9" viewBox="0 0 10 10" fill="none">
          <path d="M2 3.5L5 6.5L8 3.5" stroke="{cfg["color"]}" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        Summary
      </summary>
      <p style="font-family:Inter,sans-serif;font-size:0.72rem;color:#b0bcd4;
        line-height:1.6;margin:7px 0 0;padding:8px 10px;
        background:{cfg["lbg"]};border-left:2px solid {cfg["lbdr"]};
        border-radius:0 5px 5px 0;">{_esc(art.get("summary",""))}</p>
    </details>
    <a href="{_esc(art.get("url","#"))}" target="_blank"
      style="font-family:Inter,sans-serif;font-size:0.65rem;font-weight:500;
      color:#6b7d99;text-decoration:none;display:inline-flex;align-items:center;gap:3px;">
      Read article
      <svg width="9" height="9" viewBox="0 0 10 10" fill="none">
        <path d="M2 8L8 2M8 2H4M8 2V6" stroke="#6b7d99" stroke-width="1.5"
          stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </a>
  </div>
  {score_badge}
</div>""")

            sent_sections.append(f"""
<div style="border-top:1px solid #0f1420;">
  <div style="padding:5px 18px;display:flex;align-items:center;gap:7px;">
    <span style="font-size:0.58rem;font-weight:700;padding:2px 7px;border-radius:10px;
      text-transform:uppercase;letter-spacing:0.06em;font-family:JetBrains Mono,monospace;
      background:{cfg["sbg"]};color:{cfg["color"]};border:1px solid {cfg["sbd"]};">
      {sent_label}</span>
    <span style="font-size:0.65rem;color:#2e3a50;font-family:Inter,sans-serif;">
      {len(sent_arts)} article{"s" if len(sent_arts)!=1 else ""}</span>
  </div>
  {"".join(article_rows)}
</div>""")

        cards.append(f"""
<div style="border:1px solid #1a2332;border-top:2px solid {tag["color"]};
  border-radius:8px;margin-bottom:14px;overflow:hidden;">
  <div style="display:flex;align-items:center;gap:9px;padding:11px 18px;background:#0a0d17;">
    <span style="font-size:0.58rem;font-weight:700;letter-spacing:0.07em;
      text-transform:uppercase;padding:3px 8px;border-radius:5px;
      font-family:JetBrains Mono,monospace;
      background:{tag["bg"]};color:{tag["color"]};border:1px solid {tag["border"]};">Cluster</span>
    <span style="font-size:0.8rem;font-weight:600;color:#dde4f0;letter-spacing:-0.02em;
      font-family:Inter,sans-serif;">{_esc(cluster_label)}</span>
    <span style="margin-left:auto;font-size:0.65rem;color:#2e3a50;white-space:nowrap;
      font-family:Inter,sans-serif;">{len(arts)} articles</span>
  </div>
  {f'<div style="padding:6px 18px 8px;line-height:1.9;">{all_pills}</div>' if all_pills else ""}
  {"".join(sent_sections)}
</div>""")

    n_clusters = df["cluster_label"].nunique()
    n_articles = len(df)
    est_height = max(400, n_clusters * 320 + n_articles * 110)
    return _wrap_html("\n".join(cards)), est_height


def _wrap_html(body):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#07090f;font-family:Inter,sans-serif;color:#dde4f0;padding:0 0 16px}}
  ::-webkit-scrollbar{{width:3px}}
  ::-webkit-scrollbar-track{{background:#07090f}}
  ::-webkit-scrollbar-thumb{{background:#131825;border-radius:2px}}
  details>summary{{list-style:none}}
  details>summary::-webkit-details-marker{{display:none}}
  .arow:hover{{background:#0c0f1a !important}}
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
        summary = summarizer.generate_summary(content)
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
        if url and url in seen_urls:     continue
        if (title, src)  in seen_ts:    continue
        if (title, summ) in seen_tsumm: continue
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

    if sentiment_filters:
        deduped = [a for a in deduped if a.get("sentiment", "").capitalize() in sentiment_filters]
    if not deduped:
        return None

    df = pd.DataFrame(deduped)
    return cluster_news.cluster_and_label_articles(
        df, content_column="content", summary_column="summary")


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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:22px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <div style="width:36px;height:36px;border-radius:9px;flex-shrink:0;
          background:rgba(0,229,160,0.12);border:1px solid rgba(0,229,160,0.25);
          display:flex;align-items:center;justify-content:center;
          font-family:'Inter',sans-serif;font-size:0.82rem;font-weight:700;
          color:#00e5a0;letter-spacing:-0.02em;">QP</div>
        <div style="font-family:'Inter',sans-serif;font-size:1.05rem;font-weight:700;
          letter-spacing:-0.04em;color:#fff;">
          Quick<em style="font-style:normal;color:#00e5a0;">Pulse</em>
        </div>
      </div>
      <span style="display:inline-flex;align-items:center;gap:5px;
        font-family:'JetBrains Mono',monospace;font-size:0.56rem;font-weight:500;
        color:#00e5a0;background:rgba(0,229,160,0.07);
        border:1px solid rgba(0,229,160,0.18);border-radius:20px;padding:3px 9px;">
        <span style="width:5px;height:5px;border-radius:50%;background:#00e5a0;
          display:inline-block;"></span>
        live · multi-source
      </span>
    </div>""", unsafe_allow_html=True)

    if "topic_input" not in st.session_state:
        st.session_state.topic_input = ""

    topic_input = st.text_input(
        "Topic",
        placeholder="e.g. quantum computing",
        help="Leave blank for top headlines.",
        key="topic_input",
    )

    st.markdown("""
    <p style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:700;
      letter-spacing:0.12em;text-transform:uppercase;color:#6b7d99;
      margin:14px 0 8px;">Filter</p>
    """, unsafe_allow_html=True)

    if "sentiment_filters" not in st.session_state:
        st.session_state.sentiment_filters = ["Positive", "Neutral", "Negative"]

    pos_active = "Positive" in st.session_state.sentiment_filters
    neu_active = "Neutral"  in st.session_state.sentiment_filters
    neg_active = "Negative" in st.session_state.sentiment_filters

    st.markdown(f"""
    <style>
    /* Force the columns container to wrap into two rows */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="baseButton-secondary"]) {{
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        padding: 0 !important;
        background: transparent !important;
    }}
    /* All three columns auto-width */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="baseButton-secondary"])
        > [data-testid="column"] {{
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
        padding: 0 !important;
    }}
    /* Shared pill style */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="baseButton-secondary"])
        .stButton > button {{
        border-radius: 20px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.6rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        padding: 5px 16px !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1.4 !important;
        width: auto !important;
        min-width: 0 !important;
        white-space: nowrap !important;
        transition: all 0.15s ease !important;
    }}
    /* Positive */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="baseButton-secondary"])
        > [data-testid="column"]:nth-child(1) .stButton > button {{
        background: {"rgba(0,229,160,0.15)"   if pos_active else "transparent"} !important;
        border: 1px solid {"#00e5a0"          if pos_active else "#2e3a50"} !important;
        box-shadow: {"0 0 0 1px #00e5a0"      if pos_active else "none"} !important;
        color: {"#00e5a0"                     if pos_active else "#3d4f6a"} !important;
        opacity: {"1"                         if pos_active else "0.5"} !important;
    }}
    /* Neutral */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="baseButton-secondary"])
        > [data-testid="column"]:nth-child(2) .stButton > button {{
        background: {"rgba(107,125,153,0.15)" if neu_active else "transparent"} !important;
        border: 1px solid {"#6b7d99"          if neu_active else "#2e3a50"} !important;
        box-shadow: {"0 0 0 1px #6b7d99"      if neu_active else "none"} !important;
        color: {"#b0bcd4"                     if neu_active else "#3d4f6a"} !important;
        opacity: {"1"                         if neu_active else "0.5"} !important;
    }}
    /* Negative — force onto new line with left margin to visually center */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="baseButton-secondary"])
        > [data-testid="column"]:nth-child(3) {{
        flex-basis: 100% !important;
        display: flex !important;
        justify-content: center !important;
        padding-top: 0 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="baseButton-secondary"])
        > [data-testid="column"]:nth-child(3) .stButton > button {{
        background: {"rgba(255,107,107,0.15)" if neg_active else "transparent"} !important;
        border: 1px solid {"#ff6b6b"          if neg_active else "#2e3a50"} !important;
        box-shadow: {"0 0 0 1px #ff6b6b"      if neg_active else "none"} !important;
        color: {"#ff6b6b"                     if neg_active else "#3d4f6a"} !important;
        opacity: {"1"                         if neg_active else "0.5"} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    pcol1, pcol2, pcol3 = st.columns([1, 1, 1])
    for col, label in zip([pcol1, pcol2, pcol3], ["Positive", "Neutral", "Negative"]):
        is_active = label in st.session_state.sentiment_filters
        with col:
            if st.button(label, key=f"pill_{label}"):
                if is_active:
                    if len(st.session_state.sentiment_filters) > 1:
                        st.session_state.sentiment_filters.remove(label)
                else:
                    st.session_state.sentiment_filters.append(label)
                st.rerun()

    sentiment_filters = list(st.session_state.sentiment_filters)
    
    run_btn       = st.button("Generate Digest", use_container_width=True, type="secondary")
    headlines_btn = st.button("Top Headlines",   use_container_width=True, type="secondary")
    clear_btn     = st.button("Clear",           use_container_width=True, type="secondary")

    st.markdown("""
    <div style="margin-top:2rem;padding-top:20px;border-top:1px solid #131825;">
      <p style="font-family:'JetBrains Mono',monospace;font-size:0.52rem;
        color:#3d4f6a;line-height:2;letter-spacing:0.04em;">
        Powered by<br>
        <span style="color:#4a5e7a;">NewsAPI · HDBSCAN<br>FlanT5 · BART-MNLI<br>
        sentence-transformers</span>
      </p>
    </div>""", unsafe_allow_html=True)

# ── Actions ───────────────────────────────────────────────────────────────────
if clear_btn:
    st.session_state.result           = None
    st.session_state.active_filters   = None
    st.session_state.sentiment_filters = ["Positive", "Neutral", "Negative"]
    if "topic_input" in st.session_state:
        del st.session_state["topic_input"]
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
filters = list(st.session_state.active_filters or st.session_state.sentiment_filters)

if result is not None:
    df_res  = result["dataframe"]
    n_art   = len(df_res)
    n_clust = result["number_of_clusters"]
    pos_pct = round(100 * (df_res["sentiment"].str.capitalize() == "Positive").sum() / max(n_art, 1))
    neg_pct = round(100 * (df_res["sentiment"].str.capitalize() == "Negative").sum() / max(n_art, 1))

    # ── KPI strip — flush borderless stat bar matching mockup ──
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div style="padding:16px 18px;border:1px solid #131825;
          border-right:none;background:#07090f;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.54rem;font-weight:600;
            letter-spacing:0.1em;text-transform:uppercase;color:#2e3a50;margin-bottom:5px;">
            Articles</div>
          <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;
            letter-spacing:-0.05em;color:#fff;">{n_art}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div style="padding:16px 18px;border-top:1px solid #131825;
          border-bottom:1px solid #131825;background:#07090f;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.54rem;font-weight:600;
            letter-spacing:0.1em;text-transform:uppercase;color:#2e3a50;margin-bottom:5px;">
            Clusters</div>
          <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;
            letter-spacing:-0.05em;color:#fff;">{n_clust}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div style="padding:16px 18px;border-top:1px solid #131825;
          border-bottom:1px solid #131825;background:#07090f;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.54rem;font-weight:600;
            letter-spacing:0.1em;text-transform:uppercase;color:#2e3a50;margin-bottom:5px;">
            Positive</div>
          <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;
            letter-spacing:-0.05em;color:#00e5a0;">{pos_pct}%</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div style="padding:16px 18px;border:1px solid #131825;
          border-left:none;background:#07090f;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.54rem;font-weight:600;
            letter-spacing:0.1em;text-transform:uppercase;color:#2e3a50;margin-bottom:5px;">
            Negative</div>
          <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;
            letter-spacing:-0.05em;color:#ff6b6b;">{neg_pct}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Charts row — left border panel / right border panel ──
    col_bar, col_donut = st.columns([3, 2])
    with col_bar:
        st.markdown("""<div style="border:1px solid #131825;border-top:none;
          border-right:none;padding:0;">""", unsafe_allow_html=True)
        st.plotly_chart(
            plot_topic_bar(result),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_donut:
        st.markdown("""<div style="border:1px solid #131825;border-top:none;
          padding:0;">""", unsafe_allow_html=True)
        st.plotly_chart(
            plot_sentiment_donut(result),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Digest header ──
    st.markdown("""
    <div style="padding:10px 0;border-top:1px solid #131825;
      display:flex;align-items:center;justify-content:space-between;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;font-weight:600;
        color:#2e3a50;letter-spacing:0.1em;text-transform:uppercase;">
        Clustered news digest</span>
    </div>""", unsafe_allow_html=True)

    # ── Digest ──
    digest_html, est_height = build_digest_html(result, st.session_state.active_filters or filters)
    components.html(digest_html, height=est_height, scrolling=True)

else:
    # ── Empty state ──
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
      padding:5rem 2rem;border:1px dashed #131825;border-radius:12px;
      background:#0a0d17;margin-top:1rem;text-align:center;">
      <div style="width:50px;height:50px;border-radius:11px;
        background:rgba(0,229,160,0.07);border:1px solid rgba(0,229,160,0.16);
        display:flex;align-items:center;justify-content:center;
        font-size:1.35rem;margin-bottom:1rem;">⚡</div>
      <p style="font-family:'Inter',sans-serif;font-size:0.95rem;font-weight:700;
        color:#dde4f0;margin-bottom:0.4rem;letter-spacing:-0.02em;">Ready to pulse</p>
      <p style="font-family:'Inter',sans-serif;font-size:0.82rem;color:#2e3a50;
        max-width:320px;line-height:1.65;">
        Type a topic in the sidebar, paste URLs, or hit
        <span style="color:#6b7d99;font-weight:500;">Top Headlines</span>
        to analyze the latest news.
      </p>
    </div>""", unsafe_allow_html=True)
