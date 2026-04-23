"""
QuickPulse — Streamlit Edition
Fetch · Summarize · Cluster · Analyze live news.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import gather_news
import cluster_news
import summarizer
import analyze_sentiment
import extract_news

# ── PALETTE ──────────────────────────────────────────────────────────────────
# bg_base      #09090f   deepest surface (near-black with a blue undertone)
# bg_card      #0f1018   card surface
# bg_raised    #161824   raised / hover surface
# border       #1e2130   subtle divider
# accent       #7B61FF   electric violet — primary accent
# accent_b     #38BDF8   sky cyan — secondary accent / neutral indicator
# pos          #34D399   emerald green — positive sentiment
# neg          #F87171   rose red  — negative sentiment
# neu          #94A3B8   slate — neutral sentiment
# txt_hi       #F1F5F9   headings / primary text
# txt_lo       #64748B   muted text / labels
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@600;700;800&display=swap');

/* ── Global reset ── */
html, body, [class*="st-"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #F1F5F9;
}

.stApp {
    background: #09090f !important;
}

/* ── Main content area ── */
.main .block-container {
    background: #09090f !important;
    padding-top: 2rem !important;
    max-width: 1280px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1018 !important;
    border-right: 1px solid #1e2130 !important;
}
[data-testid="stSidebar"] .block-container {
    background: #0f1018 !important;
    padding: 2rem 1.25rem !important;
}

/* ── Input widgets ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #161824 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
    color: #F1F5F9 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #7B61FF !important;
    box-shadow: 0 0 0 3px rgba(123,97,255,0.15) !important;
}

/* ── Labels ── */
label, .stSelectbox label, .stCheckbox label {
    color: #64748B !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #7B61FF !important;
    color: #F1F5F9 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.2s, transform 0.12s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #9580FF !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #161824 !important;
    border: 1px solid #1e2130 !important;
    color: #94A3B8 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #7B61FF !important;
    color: #7B61FF !important;
    background: #161824 !important;
}

/* ── Multiselect / checkbox ── */
[data-testid="stMultiSelect"] > div {
    background: #161824 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
}
.stCheckbox > label {
    color: #94A3B8 !important;
    font-size: 0.85rem !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    font-weight: 400 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0f1018 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
    color: #94A3B8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #0f1018 !important;
    border: 1px solid #1e2130 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Plotly charts ── */
.js-plotly-plot, .plot-container {
    border-radius: 12px !important;
    border: 1px solid #1e2130 !important;
    overflow: hidden !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2130 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── HR divider ── */
hr {
    border: none !important;
    border-top: 1px solid #1e2130 !important;
    margin: 2rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #09090f; }
::-webkit-scrollbar-thumb { background: #1e2130; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7B61FF; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0f1018 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="metric-container"] label {
    color: #64748B !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #7B61FF !important;
}

/* ── Info / warning boxes ── */
.stAlert {
    background: #161824 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
    color: #94A3B8 !important;
}
</style>
"""

_HERO_HTML = """
<div style="
    background: linear-gradient(135deg, #09090f 0%, #0d0c1a 50%, #09090f 100%);
    padding: 3.5rem 2rem 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 2rem;
    border-radius: 16px;
    border: 1px solid #1e2130;
">
  <!-- Background glow -->
  <div style="
    position: absolute; top: -60px; left: 50%; transform: translateX(-50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(123,97,255,0.18) 0%, transparent 70%);
    pointer-events: none;
  "></div>

  <!-- Live badge -->
  <div style="
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(123,97,255,0.1); border: 1px solid rgba(123,97,255,0.3);
    border-radius: 100px; padding: 4px 14px; margin-bottom: 1.5rem;
  ">
    <span style="
      display: inline-block; width: 6px; height: 6px;
      border-radius: 50%; background: #7B61FF;
      box-shadow: 0 0 8px #7B61FF;
      animation: pulse 2s ease-in-out infinite;
    "></span>
    <span style="
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem; font-weight: 600;
      color: #7B61FF; letter-spacing: 0.14em; text-transform: uppercase;
    ">live · multi-source</span>
  </div>

  <!-- Title -->
  <h1 style="
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: clamp(2.2rem, 6vw, 3.2rem);
    letter-spacing: -0.03em; margin: 0 0 0.6rem;
    background: linear-gradient(135deg, #F1F5F9 30%, #7B61FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  ">QuickPulse</h1>

  <!-- Tagline -->
  <p style="
    font-family: 'DM Sans', sans-serif; font-size: 1rem;
    color: #38BDF8; font-weight: 500; margin: 0 auto 0.4rem;
  ">Fetch, cluster &amp; analyze live news — instantly.</p>
  <p style="
    font-family: 'DM Sans', sans-serif; font-size: 0.88rem;
    color: #64748B; max-width: 500px; margin: 0 auto; line-height: 1.65;
  ">Multi-source intelligence with sentiment analysis, topic clustering, and AI summarization.</p>

  <style>
    @keyframes pulse {
      0%, 100% { opacity: 1; box-shadow: 0 0 8px #7B61FF; }
      50% { opacity: 0.5; box-shadow: 0 0 2px #7B61FF; }
    }
  </style>
</div>
"""

# ── CHART PALETTE ─────────────────────────────────────────────────────────────
CHART_BG        = "#09090f"
CHART_PAPER     = "#0f1018"
CHART_GRID      = "#161824"
CHART_BORDER    = "#1e2130"
CHART_TEXT      = "#64748B"
CHART_TITLE     = "#F1F5F9"
ACCENT          = "#7B61FF"
ACCENT_B        = "#38BDF8"
POS_COLOR       = "#34D399"
NEG_COLOR       = "#F87171"
NEU_COLOR       = "#94A3B8"

_chart_layout = dict(
    paper_bgcolor=CHART_PAPER,
    plot_bgcolor=CHART_BG,
    font=dict(family="DM Sans, sans-serif", color=CHART_TEXT, size=12),
    title_font=dict(family="Syne, sans-serif", color=CHART_TITLE, size=15, ),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(gridcolor=CHART_GRID, linecolor=CHART_BORDER,
               tickfont=dict(color=CHART_TEXT, size=10)),
    yaxis=dict(gridcolor=CHART_GRID, linecolor=CHART_BORDER,
               tickfont=dict(color=CHART_TEXT, size=10)),
)


def plot_topic_frequency(result):
    df = result["dataframe"]
    tc = df["cluster_label"].value_counts().reset_index()
    tc.columns = ["Topic", "Count"]
    # Violet → cyan gradient mapped to count
    fig = go.Figure(go.Bar(
        x=tc["Topic"],
        y=tc["Count"],
        marker=dict(
            color=tc["Count"],
            colorscale=[[0, "#3B1FA8"], [0.5, ACCENT], [1, ACCENT_B]],
            showscale=False,
            line=dict(width=0),
        ),
    ))
    fig.update_layout(
        title="Topic Frequency",
        showlegend=False,
        height=320,
        **_chart_layout,
    )
    return fig


def plot_sentiment_donut(result):
    df = result["dataframe"]
    sc = df["sentiment"].str.capitalize().value_counts().reset_index()
    sc.columns = ["Sentiment", "Count"]
    color_map = {
        "Positive": POS_COLOR,
        "Neutral":  NEU_COLOR,
        "Negative": NEG_COLOR,
    }
    colors = [color_map.get(s, ACCENT) for s in sc["Sentiment"]]
    fig = go.Figure(go.Pie(
        labels=sc["Sentiment"],
        values=sc["Count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=CHART_BG, width=3)),
        textinfo="label+percent",
        textfont=dict(family="DM Sans, sans-serif", color="#F1F5F9", size=11),
        hovertemplate="<b>%{label}</b><br>%{value} articles<extra></extra>",
    ))
    fig.update_layout(
        title="Sentiment Breakdown",
        height=320,
        legend=dict(
            bgcolor=CHART_PAPER, bordercolor=CHART_BORDER, borderwidth=1,
            font=dict(color=CHART_TEXT, size=11),
        ),
        **{k: v for k, v in _chart_layout.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


# ── BACKEND (unchanged logic, Gradio refs stripped) ───────────────────────────

def render_top_clusters_table(result, top_n=5):
    df = result["dataframe"]
    cc = df["cluster_label"].value_counts().reset_index()
    cc.columns = ["Cluster", "Articles"]
    return cc.head(top_n)


def extract_summarize_and_analyze_articles(articles):
    extracted = []
    for article in articles:
        content = article.get("text") or article.get("content")
        if not content:
            continue
        title   = article.get("title", "No title")
        summary = summarizer.generate_summary(content)
        sentiment, score = analyze_sentiment.analyze_summary(summary)
        extracted.append({
            "title":       title,
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
        url    = art.get("url")
        title  = art.get("title", "").strip().lower()
        source = art.get("source", "").strip().lower()
        summ   = art.get("summary", "").strip().lower()
        if url and url in seen_urls:             continue
        if (title, source) in seen_ts:           continue
        if (title, summ)   in seen_tsumm:        continue
        deduped.append(art)
        if url: seen_urls.add(url)
        seen_ts.add((title, source))
        seen_tsumm.add((title, summ))
    return deduped


def extract_summarize_from_urls(urls):
    articles = extract_news.extract_news_articles(urls)
    return extract_summarize_and_analyze_articles(articles)


def build_cluster_html(result, sentiment_filters=None):
    df = result["dataframe"]
    cluster_primary = result.get("cluster_primary_topics", {})
    cluster_related = result.get("cluster_related_topics", {})
    df["sentiment"] = df["sentiment"].str.capitalize()

    if sentiment_filters:
        df = df[df["sentiment"].isin(sentiment_filters)]
    if df.empty:
        return "<p style='color:#64748B;'>No articles match the selected filters.</p>"

    sentiment_cfg = {
        "Positive": {"bg": "#0d1a15", "border": POS_COLOR, "dot": POS_COLOR,  "label": "Positive"},
        "Neutral":  {"bg": "#0d1018", "border": NEU_COLOR, "dot": NEU_COLOR,  "label": "Neutral"},
        "Negative": {"bg": "#1a0d0d", "border": NEG_COLOR, "dot": NEG_COLOR,  "label": "Negative"},
    }

    blocks = []
    for cluster_label, arts in df.groupby("cluster_label"):
        lda = arts["lda_topics"].iloc[0] if "lda_topics" in arts else ""
        primary = cluster_primary.get(cluster_label, [])
        related = cluster_related.get(cluster_label, [])

        html = f"""
<div style="
    background:#0f1018;border:1px solid #1e2130;border-radius:14px;
    margin-bottom:20px;padding:22px 24px;font-family:'DM Sans',sans-serif;
">
  <!-- Cluster header -->
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
    <span style="
        background:rgba(123,97,255,0.12);border:1px solid rgba(123,97,255,0.3);
        border-radius:6px;padding:3px 10px;
        font-family:'JetBrains Mono',monospace;
        font-size:0.62rem;font-weight:600;color:{ACCENT};
        letter-spacing:0.12em;text-transform:uppercase;
    ">CLUSTER</span>
    <span style="font-size:0.98rem;font-weight:600;color:#F1F5F9;">{cluster_label}</span>
    <span style="margin-left:auto;font-size:0.78rem;color:#64748B;">
        <b style="color:#F1F5F9;">{len(arts)}</b> articles
    </span>
  </div>
"""
        if lda:
            html += f"""<p style="margin:0 0 5px;font-size:0.8rem;">
    <span style="color:#64748B;font-weight:600;">Main Themes: </span>
    <span style="color:{ACCENT_B};">{lda}</span></p>"""
        if primary:
            html += f"""<p style="margin:0 0 5px;font-size:0.8rem;">
    <span style="color:#64748B;font-weight:600;">Primary: </span>
    <span style="color:{ACCENT};">{', '.join(primary)}</span></p>"""
        if related:
            html += f"""<p style="margin:0 0 14px;font-size:0.8rem;">
    <span style="color:#64748B;font-weight:600;">Related: </span>
    <span style="color:#475569;">{', '.join(related)}</span></p>"""

        for sentiment, cfg in sentiment_cfg.items():
            sent_arts = arts[arts["sentiment"] == sentiment]
            if sent_arts.empty:
                continue
            html += f"""
  <div style="
      background:{cfg['bg']};border-left:3px solid {cfg['border']};
      border-radius:0 10px 10px 0;margin-bottom:14px;padding:14px 16px;
  ">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
      <span style="
          width:7px;height:7px;border-radius:50%;
          background:{cfg['dot']};display:inline-block;
          box-shadow:0 0 6px {cfg['dot']};
      "></span>
      <span style="
          font-size:0.73rem;font-weight:700;color:#F1F5F9;
          letter-spacing:0.06em;text-transform:uppercase;
      ">{cfg['label']} <span style="color:{cfg['dot']};">({len(sent_arts)})</span></span>
    </div>
"""
            for _, art in sent_arts.iterrows():
                score_val = art.get("score")
                score_badge = ""
                if score_val is not None:
                    try:
                        score_badge = f"""<span style="
                            font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                            color:{cfg['dot']};background:rgba(0,0,0,0.35);
                            border:1px solid {cfg['border']};border-radius:4px;
                            padding:1px 7px;margin-left:8px;
                        ">{float(score_val):.2f}</span>"""
                    except (ValueError, TypeError):
                        pass

                html += f"""
    <div style="
        margin:0 0 10px;padding:12px 14px;
        background:#09090f;border:1px solid #1e2130;border-radius:10px;
    ">
      <p style="margin:0 0 6px;font-size:0.87rem;font-weight:600;color:#F1F5F9;line-height:1.4;">
        {art['title']}{score_badge}
      </p>
      <p style="margin:0 0 8px;font-size:0.76rem;color:#475569;">
        <span style="color:#334155;">Source:</span> {art['source']}
      </p>
      <details style="margin:0 0 8px;">
        <summary style="
            cursor:pointer;font-size:0.76rem;font-weight:600;
            color:{ACCENT};list-style:none;user-select:none;
        ">▶ Summary</summary>
        <p style="
            margin:8px 0 0 12px;font-size:0.81rem;
            color:#94A3B8;line-height:1.6;
        ">{art['summary']}</p>
      </details>
      <a href="{art['url']}" target="_blank" style="
          font-size:0.76rem;color:{ACCENT_B};
          text-decoration:none;font-weight:500;
      ">Read full article →</a>
    </div>
"""
            html += "  </div>"  # close sentiment bucket

        html += "</div>"  # close cluster card
        blocks.append(html)

    return "\n".join(blocks)


def run_pipeline(articles, sentiment_filters):
    if not articles:
        return None, None, None, None, None

    articles = sorted(articles, key=lambda x: x.get("publishedAt", ""), reverse=True)
    extracted = extract_summarize_and_analyze_articles(articles)
    deduped   = deduplicate_articles(extracted)
    if not deduped:
        return None, None, None, None, None

    df     = pd.DataFrame(deduped)
    result = cluster_news.cluster_and_label_articles(
        df, content_column="content", summary_column="summary"
    )
    cluster_html  = build_cluster_html(result, sentiment_filters)
    topic_fig     = plot_topic_frequency(result)
    sent_fig      = plot_sentiment_donut(result)
    top_tbl       = render_top_clusters_table(result)
    return cluster_html, topic_fig, sent_fig, top_tbl, result


# ── STREAMLIT APP ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuickPulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(_CSS, unsafe_allow_html=True)
st.markdown(_HERO_HTML, unsafe_allow_html=True)

# ── Session state ──
for key in ("cluster_html", "topic_fig", "sent_fig", "top_tbl", "result"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span style="
          font-family:'Syne',sans-serif;font-size:1.3rem;
          font-weight:800;color:#F1F5F9;
      ">⚡ QuickPulse</span>
    </div>
    """, unsafe_allow_html=True)

    topic_input = st.text_input(
        "Search Topic",
        placeholder="e.g. artificial intelligence",
        help="Leave blank to fetch today's top headlines.",
    )

    sentiment_filters = st.multiselect(
        "Sentiment Filter",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )

    with st.expander("🔗 Batch URL Input"):
        urls_input = st.text_area(
            "URLs (one per line)",
            height=120,
            placeholder="https://example.com/article-1\nhttps://...",
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("Generate", use_container_width=True)
    with col2:
        headlines_btn = st.button("Top News", use_container_width=True)

    clear_btn = st.button("Clear", use_container_width=True)

    st.markdown("""
    <div style="margin-top:2rem;padding-top:1.5rem;border-top:1px solid #1e2130;">
      <p style="font-size:0.7rem;color:#334155;line-height:1.7;">
        Powered by NewsAPI · HDBSCAN · flan-t5 · BART-MNLI<br>
        &copy; QuickPulse
      </p>
    </div>
    """, unsafe_allow_html=True)

# ── Button actions ─────────────────────────────────────────────────────────────
articles = []

if clear_btn:
    for key in ("cluster_html", "topic_fig", "sent_fig", "top_tbl", "result"):
        st.session_state[key] = None
    st.rerun()

if run_btn:
    if topic_input and topic_input.strip():
        with st.spinner("Fetching topic articles…"):
            articles = gather_news.fetch_newsapi_everything(topic_input)
    elif urls_input and urls_input.strip():
        url_list = [u.strip() for u in urls_input.splitlines() if u.strip()]
        with st.spinner(f"Extracting {len(url_list)} URLs…"):
            extracted = extract_news.extract_news_articles(url_list)
            articles  = extracted
    else:
        st.warning("Enter a topic or paste some URLs to get started.")

    if articles:
        with st.spinner("Summarizing & clustering…"):
            ch, tf, sf, tt, res = run_pipeline(articles, sentiment_filters)
        st.session_state.cluster_html = ch
        st.session_state.topic_fig    = tf
        st.session_state.sent_fig     = sf
        st.session_state.top_tbl      = tt
        st.session_state.result       = res

if headlines_btn:
    with st.spinner("Fetching top headlines…"):
        articles = gather_news.fetch_newsapi_top_headlines()
    if articles:
        with st.spinner("Summarizing & clustering…"):
            ch, tf, sf, tt, res = run_pipeline(articles, sentiment_filters)
        st.session_state.cluster_html = ch
        st.session_state.topic_fig    = tf
        st.session_state.sent_fig     = sf
        st.session_state.top_tbl      = tt
        st.session_state.result       = res

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result is not None:
    # ── Analytics row ──
    df_res = st.session_state.result["dataframe"]
    n_articles = len(df_res)
    n_clusters = st.session_state.result["number_of_clusters"]
    pos_pct    = round(100 * (df_res["sentiment"].str.capitalize() == "Positive").sum() / max(n_articles, 1))

    m1, m2, m3 = st.columns(3)
    m1.metric("Articles Processed", n_articles)
    m2.metric("Topic Clusters",     n_clusters)
    m3.metric("Positive Sentiment", f"{pos_pct}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.topic_fig:
            st.plotly_chart(
                st.session_state.topic_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with c2:
        if st.session_state.sent_fig:
            st.plotly_chart(
                st.session_state.sent_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # ── Top clusters table ──
    if st.session_state.top_tbl is not None:
        st.markdown("""
        <p style="
            font-family:'JetBrains Mono',monospace;font-size:0.68rem;
            font-weight:600;color:{};letter-spacing:0.1em;
            text-transform:uppercase;margin:1.5rem 0 0.5rem;
        ">Top Clusters by Volume</p>
        """.format(ACCENT), unsafe_allow_html=True)
        st.dataframe(
            st.session_state.top_tbl,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Digest header ──
    st.markdown("""
    <p style="
        font-family:'JetBrains Mono',monospace;font-size:0.68rem;font-weight:600;
        color:#7B61FF;letter-spacing:0.12em;text-transform:uppercase;
        margin-bottom:1.25rem;
    ">Clustered News Digest</p>
    """, unsafe_allow_html=True)

    if st.session_state.cluster_html:
        st.markdown(st.session_state.cluster_html, unsafe_allow_html=True)

else:
    # ── Empty state ──
    st.markdown("""
    <div style="
        text-align:center;padding:4rem 2rem;
        border:1px dashed #1e2130;border-radius:16px;
        background:#0f1018;margin-top:1rem;
    ">
      <div style="font-size:2.5rem;margin-bottom:1rem;">⚡</div>
      <p style="
          font-family:'Syne',sans-serif;font-size:1.1rem;
          font-weight:700;color:#F1F5F9;margin-bottom:0.5rem;
      ">Ready to pulse</p>
      <p style="font-size:0.88rem;color:#64748B;max-width:360px;margin:0 auto;">
        Enter a topic in the sidebar, paste URLs, or hit <b style="color:#94A3B8;">Top News</b>
        to fetch and analyze the latest headlines.
      </p>
    </div>
    """, unsafe_allow_html=True)
