import os
import json
import feedparser
import numpy as np
from collections import Counter
from textwrap import wrap

import pandas as pd
from openai import OpenAI
from bertopic import BERTopic
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go

# ============================================================
# OpenAI client (supports both v1 & v2 secret wiring)
# ============================================================

_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY_V2")
if not _api_key:
    raise RuntimeError(
        "No OpenAI API key found. "
        "Set either OPENAI_API_KEY or OPEN_AI_KEY_V2 in the environment."
    )

client = OpenAI(api_key=_api_key)

# ============================================================
# CONFIG
# ============================================================

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/markets",
    "https://www.ft.com/rss/home/us",
    "https://www.wsj.com/xml/rss/3_7014.xml",
    "https://www.wsj.com/xml/rss/3_7085.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://rss.cnn.com/rss/edition_business.rss",
    "https://www.ft.com/rss/home/europe",
    "https://www.ft.com/rss/home/asia",
    "https://asia.nikkei.com/rss/feed",
    "https://www.scmp.com/rss/91/feed",
    "https://feeds.reuters.com/reuters/technologyNews",
    # "https://feeds.feedburner.com/TechCrunch/",
    "https://www.ft.com/rss/home",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.investing.com/rss/news_25.rss",
    "https://www.investing.com/rss/news_1.rss",
    "https://www.investing.com/rss/news_285.rss",
    "https://www.federalreserve.gov/feeds/data.xml",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://markets.businessinsider.com/rss",
    "https://www.risk.net/feeds/rss",
    "https://www.forbes.com/finance/feed",
    "https://feeds.feedburner.com/alternativeinvestmentnews",
    "https://www.eba.europa.eu/eba-news-rss",
    "https://www.bis.org/rss/press_rss.xml",
    "https://www.imf.org/external/np/exr/feeds/rss.aspx?type=imfnews",
    "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://feeds.washingtonpost.com/rss/business",
    "https://feeds.washingtonpost.com/rss/business/economy",
    "https://krebsonsecurity.com/feed/",
    "https://www.bleepingcomputer.com/feed/",
    "https://www.darkreading.com/rss.xml",
    "https://www.scmagazine.com/section/feed",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://www.bls.gov/feed/news-release.htm?view=all&format=rss",
    "https://www.bea.gov/rss.xml",
    "https://www.cbo.gov/publications/all/rss.xml",
    "https://fredblog.stlouisfed.org/feed/",
    "https://libertystreeteconomics.newyorkfed.org/feed/",
    "https://pitchbook.com/news/feed",
    "https://www.preqin.com/insights/rss",
    "https://www.privatedebtinvestor.com/feed/",
    "https://www.directlendingdeals.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://www.theblock.co/rss",
    "https://blog.chainalysis.com/feed/",
    "https://www.trmlabs.com/blog?format=rss",
    "https://cryptoslate.com/feed/",
    "https://cointelegraph.com/rss",
    "https://www.circle.com/blog/rss.xml",
    "https://tether.to/en/feed/",
    "https://forum.makerdao.com/latest.rss",
]

THEMES = [
    "Recessionary pressures",
    "Inflation",
    "Private credit",
    "AI",
    "Cyber attacks",
    "Commercial real estate",
    "Consumer debt",
    "Bank lending and credit risk",
    "Digital assets",
]

THEME_DESCRIPTIONS = {
    "Recessionary pressures": "Economic slowdown, declining demand.",
    "Inflation": "Price increases and monetary policy.",
    "Private credit": "Non-bank lending and liquidity risk.",
    "AI": "Artificial intelligence, data centers,hyperscalers and automation trends.",
    "Cyber attacks": "Security breaches and vulnerabilities.",
    "Commercial real estate": "Property market stress and refinancing.",
    "Consumer debt": "Household leverage and affordability issues.",
    "Bank lending and credit risk": "Defaults and regulatory pressure.",
    "Digital assets": (
        "Crypto markets, stablecoins, tokenization, blockchain infrastructure "
        "and systemic spillovers into traditional finance."
    ),
    "Others": "Articles not matching systemic themes.",
}

SIMILARITY_THRESHOLD = 0.20

# Theme-driven importance weights for topic map emphasis
THEME_WEIGHTS = {
    "Recessionary pressures": 1.0,
    "Inflation": 1.0,
    "Private credit": 1.2,
    "AI": 1.0,
    "Cyber attacks": 1.0,
    "Commercial real estate": 1.2,
    "Consumer debt": 1.0,
    "Bank lending and credit risk": 1.2,
    "Digital assets": 0.7,
    "Others": 1.0,
}

# ============================================================
# HELPERS
# ============================================================

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms


def fetch_articles():
    """Fetch & lightly clean articles from RSS feeds."""
    docs = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:20]:
                content = (
                    entry.get("summary")
                    or entry.get("description")
                    or entry.get("title")
                    or ""
                )
                if isinstance(content, str) and len(content.strip()) > 50:
                    docs.append(content.strip()[:1200])
        except Exception as e:
            print(f"Feed error {feed}: {e}")
    print("Fetched articles:", len(docs))
    return docs


def get_representative_doc_ids(doc_ids, doc_embeddings, top_k=8):
    """
    Return the indices of the most representative documents for a topic.
    """
    if not doc_ids:
        return []
    if len(doc_ids) <= top_k:
        return doc_ids

    emb = doc_embeddings[doc_ids]  # (n_docs_in_topic, dim)
    centroid = np.mean(emb, axis=0, keepdims=True)
    sims = cosine_similarity(emb, centroid).ravel()
    ranked = np.argsort(-sims)
    return [doc_ids[i] for i in ranked[:top_k]]


def gpt_summarize_topic(topic_id, docs_for_topic):
    """
    Structured, sharper topic summary with:
      - TITLE
      - OVERVIEW (1–2 sentences)
      - KEY EXAMPLES (2–4 bullets)
    """
    if not docs_for_topic:
        return {"title": f"TOPIC {topic_id}", "summary": "Summary unavailable."}

    articles_block = "\n\n".join(
        [f"ARTICLE {i+1}:\n{doc}" for i, doc in enumerate(docs_for_topic)]
    )

    prompt = f"""
You are summarizing a news topic formed by clustering multiple related articles.

Write a structured, factual, concise summary in this exact layout:

TITLE: <3–5 word topic label>

OVERVIEW:
1–2 sentences summarizing the main common theme across these articles.
Be concrete and specific. Avoid vague macro language and grand conclusions.

KEY EXAMPLES:
- Short, distinct example 1 drawn from one article
- Short, distinct example 2 drawn from another article
- Short, distinct example 3 (optional)
- Short, distinct example 4 (optional)

Rules:
- Use only information that appears in the articles.
- Do not invent entities, events, or numbers.
- Do not mention specific publishers or dates.
- Do not explain your reasoning or mention this prompt.

ARTICLES:
{articles_block}
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"GPT error for topic {topic_id}: {e}")
        return {"title": f"TOPIC {topic_id}", "summary": "Summary unavailable."}

    # ---- Robust parsing for TITLE / SUMMARY ----
    text_upper = out.upper()
    idx = text_upper.find("TITLE:")
    if idx != -1:
        after = out[idx + len("TITLE:"):].strip()
    else:
        after = out

    lines = [ln.strip() for ln in after.splitlines() if ln.strip()]

    if not lines:
        return {"title": f"TOPIC {topic_id}", "summary": "Summary unavailable."}

    title_line = lines[0]
    for prefix in ("- ", "* ", "• "):
        if title_line.startswith(prefix):
            title_line = title_line[len(prefix):].strip()

    summary_body = "\n".join(lines[1:]).strip()
    if not summary_body:
        summary_body = "Summary unavailable."

    return {
        "title": title_line,
        "summary": summary_body,
    }


def run_bertopic_analysis(docs):
    """Fit BERTopic with UMAP + KMeans configuration."""
    umap_model = UMAP(
        n_neighbors=30,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
    )

    kmeans_model = KMeans(
        n_clusters=15,
        random_state=42,
        n_init="auto",
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 3),
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=kmeans_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
    )

    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics


# ============================================================
# LABEL SHORTENER FOR NON–TOP-5 TOPICS
# ============================================================

_STOPWORDS = {"and", "of", "the", "in", "for", "to", "on", "a", "an"}


def _short_label(full_title: str, max_words: int = 4) -> str:
    """
    Take the first 3–4 "significant" words of a topic title.
    Used for *non* top-5 topics to keep the map readable.
    """
    if not full_title:
        return ""

    clean = full_title.replace("·", " ")
    tokens = clean.split()
    if not tokens:
        return ""

    significant = []
    for w in tokens:
        if not significant and w.lower() in _STOPWORDS:
            continue
        significant.append(w)
        if len(significant) >= max_words:
            break

    if not significant:
        significant = tokens[:max_words]

    return " ".join(significant)


# ============================================================
# BUILD IMPROVED TOPIC MAP (TRUE BERTopic EMBEDDINGS)
# ============================================================

def build_topic_map(topic_embeddings, summaries):
    """
    Build Intertopic Distance Map using true BERTopic 2D embeddings.
    Returns raw HTML for dashboard/topic_map.html with PlotlyJS
    loaded from CDN (include_plotlyjs="cdn").
    """
    topic_ids = sorted(topic_embeddings.keys())
    if not topic_ids:
        return "<p>No topic map available.</p>"

    xs = [topic_embeddings[i][0] for i in topic_ids]
    ys = [topic_embeddings[i][1] for i in topic_ids]

    volumes = []
    titles = {}
    weights = {}

    for tid in topic_ids:
        meta = summaries.get(tid, {})
        titles[tid] = meta.get("title", f"TOPIC {tid}")
        volumes.append(meta.get("article_count", 0))
        weights[tid] = meta.get("theme_weight", 1.0)

    # Base marker sizes from volume
    v_min = min(volumes)
    v_max = max(volumes)
    if v_max == v_min:
        base_sizes = np.full(len(volumes), 40.0)
    else:
        base_sizes = np.interp(volumes, (v_min, v_max), (25, 70))

    size_scale = [float(bs * weights[tid]) for bs, tid in zip(base_sizes, topic_ids)]

    # Identify top-5 topics by volume
    vol_array = np.array(volumes, dtype=float)
    idx_sorted = np.argsort(-vol_array)
    top5_idx = idx_sorted[:5]
    top5_ids = {topic_ids[i] for i in top5_idx}

    # Colors for markers
    fill_colors = []
    border_colors = []
    for tid in topic_ids:
        if tid in top5_ids:
            fill_colors.append("rgba(227, 168, 105, 0.35)")   # light brown
            border_colors.append("rgba(191, 120, 52, 0.95)")  # darker brown
        else:
            fill_colors.append("rgba(58, 110, 165, 0.25)")    # blue
            border_colors.append("rgba(58, 110, 165, 0.9)")

    # Marker trace (all topics)
    marker_trace = go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(
            size=size_scale,
            color=fill_colors,
            line=dict(color=border_colors, width=2),
        ),
        hovertext=[titles[tid] for tid in topic_ids],
        hovertemplate="<b>%{hovertext}</b><extra></extra>",
        showlegend=False,
    )

    # Text trace for top-5 topics (full labels, wrapped)
    top5_x = []
    top5_y = []
    top5_text = []
    for tid, x, y in zip(topic_ids, xs, ys):
        if tid in top5_ids:
            wrapped = "<br>".join(wrap(titles[tid], width=22))
            top5_text.append(f"<b>{wrapped}</b>")
            top5_x.append(x)
            top5_y.append(y)

    top5_text_trace = go.Scatter(
        x=top5_x,
        y=top5_y,
        mode="text",
        text=top5_text,
        textposition="top center",
        textfont=dict(size=12, color="#111111"),
        showlegend=False,
        hoverinfo="skip",
    )

    # Text trace for all other topics (short labels, small font)
    other_x = []
    other_y = []
    other_text = []
    for tid, x, y in zip(topic_ids, xs, ys):
        if tid not in top5_ids:
            short = _short_label(titles[tid], max_words=4)
            other_text.append(short)
            other_x.append(x)
            other_y.append(y)

    other_text_trace = go.Scatter(
        x=other_x,
        y=other_y,
        mode="text",
        text=other_text,
        textposition="top center",
        textfont=dict(size=9, color="#333333"),
        showlegend=False,
        hoverinfo="skip",
    )

    fig = go.Figure([marker_trace, top5_text_trace, other_text_trace])

    fig.update_layout(
        title=dict(
            text="<b>Intertopic Distance Map (BERTopic)</b>",
            x=0.5,
            font=dict(size=22),
        ),
        autosize=True,
        height=700,
        margin=dict(l=10, r=10, t=80, b=40),
        xaxis=dict(
            title="Embedding dimension 1",
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="#444",
        ),
        yaxis=dict(
            title="Embedding dimension 2",
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="#444",
        ),
        plot_bgcolor="white",
        hovermode="closest",
    )

    # Plotly JS from CDN
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


# ============================================================
# MAIN TOPIC + THEME PIPELINE
# ============================================================

def generate_topic_results():
    """
    Run full BERTopic pipeline.

    Returns:
      docs            : list of article texts
      summaries       : dict[topic_id] -> {title, summary, article_count, ...}
      topic_model     : BERTopic model
      topic_embeddings: dict[topic_id] -> [x, y] embedding
      theme_metrics   : dict[theme_name] -> metrics + article sets + topic_affinity_pct
      topics          : list of topic ids per document
    """
    docs = fetch_articles()
    if not docs:
        return [], {}, None, {}, {}, []

    # Topic model
    topic_model, topics = run_bertopic_analysis(docs)
    topic_info = topic_model.get_topic_info()
    valid_topic_ids = [t for t in topic_info.Topic if t != -1]

    # Article embeddings (used for both representative docs + themes)
    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    art_emb = _normalize_rows(sent_model.encode(docs, show_progress_bar=False))

    summaries = {}
    topic_embeddings = {}
    topic_doc_ids = {}

    # --- Summaries using representative docs ---
    for topic_id in valid_topic_ids:
        doc_ids = [i for i, t in enumerate(topics) if t == topic_id]
        topic_doc_ids[topic_id] = doc_ids

        rep_ids = get_representative_doc_ids(doc_ids, art_emb, top_k=8)
        topic_docs = [docs[i] for i in rep_ids]

        summaries[topic_id] = gpt_summarize_topic(topic_id, topic_docs)
        summaries[topic_id]["article_count"] = len(doc_ids)
        summaries[topic_id]["topic_id"] = topic_id

        topic_embeddings[topic_id] = topic_model.topic_embeddings_[topic_id].tolist()

    # --- Theme assignment (re-use art_emb) ---
    theme_texts = [f"{t}. {THEME_DESCRIPTIONS[t]}" for t in THEMES]
    theme_emb = _normalize_rows(sent_model.encode(theme_texts, show_progress_bar=False))

    theme_metrics = {
        t: {"volume": 0, "centrality": 0.0, "articles": set()} for t in THEMES
    }
    theme_metrics["Others"] = {"volume": 0, "centrality": 0.0, "articles": set()}

    # dominant theme per article (for topic weights and theme labels)
    dominant_theme = ["Others"] * len(docs)

    for i, emb in enumerate(art_emb):
        sims = cosine_similarity([emb], theme_emb)[0]

        # dominant theme (for weighting)
        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]
        if best_score >= SIMILARITY_THRESHOLD:
            dom = THEMES[best_idx]
        else:
            dom = "Others"
        dominant_theme[i] = dom

        # multi-assignment for theme metrics
        assigned = [
            THEMES[idx]
            for idx, score in enumerate(sims)
            if score >= SIMILARITY_THRESHOLD
        ]
        if not assigned:
            assigned = ["Others"]
        for theme in assigned:
            theme_metrics[theme]["volume"] += 1
            theme_metrics[theme]["articles"].add(i)

    # Centrality (simple overlap-based)
    for t in THEMES:
        overlaps = 0
        Ta = theme_metrics[t]["articles"]
        for other in THEMES:
            if other != t:
                overlaps += len(Ta.intersection(theme_metrics[other]["articles"]))
        theme_metrics[t]["centrality_raw"] = overlaps

    max_c = max(theme_metrics[t].get("centrality_raw", 0) for t in THEMES) or 1
    for t in THEMES:
        theme_metrics[t]["centrality"] = theme_metrics[t]["centrality_raw"] / max_c
    theme_metrics["Others"]["centrality"] = 0.0

    # --- Topic-level theme weights for map emphasis + dominant theme label ---
    for topic_id in valid_topic_ids:
        doc_ids = topic_doc_ids.get(topic_id, [])
        if not doc_ids:
            w = 1.0
            dom_theme_for_topic = "Others"
        else:
            w = float(
                np.mean([THEME_WEIGHTS.get(dominant_theme[i], 1.0) for i in doc_ids])
            )
            th_counts = Counter(dominant_theme[i] for i in doc_ids)
            dom_theme_for_topic, _ = th_counts.most_common(1)[0]

        summaries[topic_id]["theme_weight"] = w
        summaries[topic_id]["dominant_theme"] = dom_theme_for_topic

    # --- Theme × Topic affinity (% of topic's articles assigned to theme) ---
    for t in theme_metrics:
        theme_metrics[t].setdefault("topic_affinity_pct", {})

    for topic_id in valid_topic_ids:
        doc_ids = topic_doc_ids.get(topic_id, [])
        n_docs_topic = len(doc_ids)
        if n_docs_topic == 0:
            for th in theme_metrics:
                theme_metrics[th]["topic_affinity_pct"][str(topic_id)] = 0.0
            continue

        doc_set = set(doc_ids)
        for th in theme_metrics:
            theme_articles = theme_metrics[th]["articles"]
            overlap = len(doc_set.intersection(theme_articles))
            pct = overlap / n_docs_topic if n_docs_topic > 0 else 0.0
            theme_metrics[th]["topic_affinity_pct"][str(topic_id)] = float(pct)

    # -------- JSON compatibility: remove sets & raw centrality --------
    for t_name, t_metrics in theme_metrics.items():
        for key, val in list(t_metrics.items()):
            if isinstance(val, set):
                t_metrics[key] = list(val)
        t_metrics.pop("centrality_raw", None)

    return docs, summaries, topic_model, topic_embeddings, theme_metrics, topics


# ============================================================
# PERSIST RESULTS TO DISK (for dashboard)
# ============================================================

def run_and_persist_bertopic():
    """
    Run the full BERTopic pipeline and persist outputs for the dashboard:
      - topics.json
      - theme_signals.json
      - articles.csv
      - dashboard/topic_map.html
    """
    (
        docs,
        summaries,
        topic_model,
        topic_embeddings,
        theme_metrics,
        topics,
    ) = generate_topic_results()

    if not docs:
        print("⚠️ No docs fetched; skipping persistence.")
        return

    # Ensure dashboard folder exists
    os.makedirs("dashboard", exist_ok=True)

    # --- Build topics.json ---
    topics_out = {}
    for topic_id, meta in summaries.items():
        tid = f"T{topic_id}"
        topics_out[tid] = {
            "topic_id": tid,
            "bertopic_id": int(topic_id),
            "title": meta.get("title", f"TOPIC {topic_id}"),
            "summary": meta.get("summary", ""),
            "article_count": int(meta.get("article_count", 0)),
            "topicality": float(meta.get("article_count", 0.0)),
            "theme_weight": float(meta.get("theme_weight", 1.0)),
            "theme": meta.get("dominant_theme", "Others"),
        }

    topics_path = "topics.json"
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topics_out, f, indent=2)
    print(f"💾 Wrote topics to {topics_path}")

    # --- Build theme_signals.json ---
    theme_signals_path = "theme_signals.json"
    with open(theme_signals_path, "w", encoding="utf-8") as f:
        json.dump(theme_metrics, f, indent=2)
    print(f"💾 Wrote theme signals to {theme_signals_path}")

    # --- Build articles.csv ---
    rows = []
    for idx, text in enumerate(docs):
        t_id = topics[idx]
        tid = f"T{t_id}"
        rows.append(
            {
                "id": idx,
                "text": text,
                "topic_id": tid,
            }
        )

    articles_df = pd.DataFrame(rows)
    articles_path = "articles.csv"
    articles_df.to_csv(articles_path, index=False)
    print(f"💾 Wrote articles to {articles_path} ({len(articles_df)} rows)")

    # --- Build and save BERTopic topic map HTML ---
    try:
        topic_map_html = build_topic_map(topic_embeddings, summaries)
        topic_map_path = os.path.join("dashboard", "topic_map.html")
        with open(topic_map_path, "w", encoding="utf-8") as f:
            f.write(topic_map_html)
        print(f"💾 Saved BERTopic topic map to {topic_map_path}")
    except Exception as e:
        print(f"⚠️ Could not build/save BERTopic topic map: {e}")


# ============================================================
# TEST RUN (local)
# ============================================================

if __name__ == "__main__":
    run_and_persist_bertopic()
    print("✅ BERTopic engine run and persisted successfully.")

