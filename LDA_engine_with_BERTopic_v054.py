import os
import json
import feedparser
import requests
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
# OpenAI client — unified key loading + required User-Agent
# ============================================================

_api_key = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPEN_AI_KEY_V2")
    or os.getenv("OPEN_API_KEY_V2")
)

if not _api_key:
    raise RuntimeError(
        "No OpenAI API key found. Set one of: OPENAI_API_KEY, OPEN_AI_KEY_V2, OPEN_API_KEY_V2."
    )

client = OpenAI(
    api_key=_api_key,
    default_headers={"User-Agent": "Mozilla/5.0"}
)

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
    #"https://forum.makerdao.com/latest.rss",
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

SIMILARITY_THRESHOLD = 0.15

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

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms


def fetch_articles():
    """Fetch RSS feeds with safe skipping and clear logging."""
    docs = []
    print("🔍 Starting RSS fetch...")

    headers = {"User-Agent": "Mozilla/5.0"}

    for feed in RSS_FEEDS:
        print(f"\n--- Checking feed: {feed}")

        try:
            r = requests.get(feed, timeout=12, headers=headers)
            print(f"HTTP status: {r.status_code}")

            if r.status_code != 200:
                print("❌ Non-200 response, skipping.")
                continue

            parsed = feedparser.parse(r.text)
            print(f"Entries parsed: {len(parsed.entries)}")

            for entry in parsed.entries[:20]:
                text = (
                    entry.get("summary")
                    or entry.get("description")
                    or entry.get("title")
                    or ""
                ).strip()

                if len(text) > 50:
                    docs.append(text[:1200])

        except Exception as e:
            print(f"🔥 ERROR reading feed:\n    {e}")
            print("➡️ Skipping feed.")
            continue

    print(f"\n📊 Total extracted articles: {len(docs)}")
    return docs


def get_representative_doc_ids(doc_ids, doc_embeddings, top_k=8):
    if len(doc_ids) <= top_k:
        return doc_ids
    emb = doc_embeddings[doc_ids]
    centroid = np.mean(emb, axis=0, keepdims=True)
    sims = cosine_similarity(emb, centroid).ravel()
    ranked = np.argsort(-sims)
    return [doc_ids[i] for i in ranked[:top_k]]


def gpt_summarize_topic(topic_id, docs_for_topic):
    block = "\n\n".join(f"ARTICLE {i+1}:\n{d}" for i, d in enumerate(docs_for_topic))

    prompt = f"""
You are summarizing a cluster of related news articles.

Write a structured summary:

TITLE: <3–5 word topic label>

OVERVIEW:
1–2 sentences describing the common theme.

KEY EXAMPLES:
- One short example
- Another example
- Optional example
- Optional example

Use **only** information in the articles.

ARTICLES:
{block}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        out = resp.choices[0].message.content or ""

        if "TITLE:" in out:
            _, after = out.split("TITLE:", 1)
            lines = after.strip().splitlines()
            title = lines[0].strip()
            summary = "\n".join(lines[1:]).strip() or "Summary unavailable."
            return {"title": title, "summary": summary}

    except Exception as e:
        print(f"GPT error for topic {topic_id}: {e}")

    return {"title": f"TOPIC {topic_id}", "summary": "Summary unavailable."}


# ------------------------------------------------------------
# BERTopic
# ------------------------------------------------------------

def run_bertopic_analysis(docs):
    umap_model = UMAP(n_neighbors=30, n_components=2, min_dist=0, metric="cosine")
    kmeans_model = KMeans(n_clusters=15, random_state=42, n_init="auto")
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1,3))

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=kmeans_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
    )

    topics, probs = model.fit_transform(docs)
    return model, topics


# ------------------------------------------------------------
# Intertopic distance map
# ------------------------------------------------------------

def build_topic_map(topic_embeddings, summaries):
    topic_ids = sorted(topic_embeddings.keys())
    if not topic_ids:
        return "<p>No topic map available.</p>"

    xs = [topic_embeddings[i][0] for i in topic_ids]
    ys = [topic_embeddings[i][1] for i in topic_ids]

    volumes = [summaries[tid]["article_count"] for tid in topic_ids]
    weights = [summaries[tid]["theme_weight"] for tid in topic_ids]
    titles = [summaries[tid]["title"] for tid in topic_ids]

    vmin, vmax = min(volumes), max(volumes)
    base_sizes = (
        np.full(len(volumes), 40)
        if vmin == vmax else
        np.interp(volumes, (vmin, vmax), (25, 70))
    )
    marker_sizes = [float(bs * weights[i]) for i, bs in enumerate(base_sizes)]

    fig = go.Figure(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=titles,
        textposition="top center",
        marker=dict(
            size=marker_sizes,
            color="rgba(58,110,165,0.4)",
            line=dict(color="rgba(58,110,165,1.0)", width=2),
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))

    # *** CHANGED: explicit axes + layout so axes show up clearly
    fig.update_layout(
        title=dict(
            text="<b>Intertopic Distance Map (BERTopic)</b>",
            x=0.5,
            font=dict(size=22),
        ),
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
        hovermode="closest",
        height=700,
        margin=dict(l=10, r=10, t=80, b=40),
        plot_bgcolor="white",
        template=None,  # avoid any template overriding axis styling
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def generate_topic_results():
    docs = fetch_articles()
    if not docs:
        return [], {}, None, {}, {}, []

    model, topics = run_bertopic_analysis(docs)

    topic_info = model.get_topic_info()
    valid = [t for t in topic_info.Topic if t != -1]

    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    art_emb = _normalize_rows(sent_model.encode(docs, show_progress_bar=False))

    summaries = {}
    topic_embeddings = {}
    topic_doc_ids = {}

    for tid in valid:
        doc_ids = [i for i, t in enumerate(topics) if t == tid]
        topic_doc_ids[tid] = doc_ids

        rep_ids = get_representative_doc_ids(doc_ids, art_emb)
        rep_docs = [docs[i] for i in rep_ids]

        summaries[tid] = gpt_summarize_topic(tid, rep_docs)
        summaries[tid]["article_count"] = len(doc_ids)
        summaries[tid]["topic_id"] = tid
        topic_embeddings[tid] = model.topic_embeddings_[tid].tolist()

    # Theme assignment
    theme_texts = [f"{t}. {THEME_DESCRIPTIONS[t]}" for t in THEMES]
    theme_emb = _normalize_rows(sent_model.encode(theme_texts, show_progress_bar=False))

    theme_metrics = {
        t: {"volume": 0, "centrality": 0.0, "articles": set()} for t in THEMES
    }
    theme_metrics["Others"] = {"volume": 0, "centrality": 0.0, "articles": set()}
    dominant_theme = ["Others"] * len(docs)

    for i, emb in enumerate(art_emb):
        sims = cosine_similarity([emb], theme_emb)[0]
        best_idx = int(np.argmax(sims))
        score = sims[best_idx]
        main = THEMES[best_idx] if score >= SIMILARITY_THRESHOLD else "Others"
        dominant_theme[i] = main

        assigned = [THEMES[j] for j, s in enumerate(sims) if s >= SIMILARITY_THRESHOLD]
        if not assigned:
            assigned = ["Others"]

        for th in assigned:
            theme_metrics[th]["volume"] += 1
            theme_metrics[th]["articles"].add(i)

    # Centrality
    for th in THEMES:
        Ta = theme_metrics[th]["articles"]
        overlaps = sum(
            len(Ta & theme_metrics[o]["articles"]) for o in THEMES if o != th
        )
        theme_metrics[th]["centrality_raw"] = overlaps

    max_c = max(theme_metrics[t]["centrality_raw"] for t in THEMES) or 1
    for t in THEMES:
        theme_metrics[t]["centrality"] = theme_metrics[t]["centrality_raw"] / max_c
    theme_metrics["Others"]["centrality"] = 0.0

    # Theme weights
    for tid in valid:
        ids = topic_doc_ids[tid]
        if not ids:
            w = 1.0
            dom = "Others"
        else:
            dom = Counter(dominant_theme[i] for i in ids).most_common(1)[0][0]
            w = float(np.mean([THEME_WEIGHTS[dominant_theme[i]] for i in ids]))

        summaries[tid]["theme_weight"] = w
        summaries[tid]["dominant_theme"] = dom

    # Affinity
    for t in theme_metrics:
        theme_metrics[t]["topic_affinity_pct"] = {}

    for tid in valid:
        docs_in_topic = set(topic_doc_ids[tid])
        N = len(docs_in_topic)
        for th in theme_metrics:
            overlap = len(docs_in_topic & theme_metrics[th]["articles"])
            theme_metrics[th]["topic_affinity_pct"][str(tid)] = (
                overlap / N if N else 0.0
            )

    # JSON cleanup
    for th in theme_metrics:
        for k, v in list(theme_metrics[th].items()):
            if isinstance(v, set):
                theme_metrics[th][k] = list(v)
        theme_metrics[th].pop("centrality_raw", None)

    return docs, summaries, model, topic_embeddings, theme_metrics, topics


# ------------------------------------------------------------
# WRITE OUTPUTS
# ------------------------------------------------------------

def run_and_persist_bertopic():
    docs, summaries, model, topic_embeddings, theme_metrics, topics = generate_topic_results()
    if not docs:
        print("⚠️ No docs fetched; skipping.")
        return

    os.makedirs("dashboard", exist_ok=True)

    # topics.json
    topics_out = {}
    for tid, meta in summaries.items():
        tname = f"T{tid}"
        topics_out[tname] = {
            "topic_id": tname,
            "bertopic_id": tid,
            "title": meta["title"],
            "summary": meta["summary"],
            "article_count": meta["article_count"],
            "topicality": float(meta["article_count"]),
            "theme_weight": float(meta["theme_weight"]),
            "theme": meta["dominant_theme"],
        }

    with open("topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_out, f, indent=2)

    # theme_signals.json
    with open("theme_signals.json", "w", encoding="utf-8") as f:
        json.dump(theme_metrics, f, indent=2)

    # articles.csv
    rows = [{"id": i, "text": docs[i], "topic_id": f"T{topics[i]}"} for i in range(len(docs))]
    pd.DataFrame(rows).to_csv("articles.csv", index=False)

    # topic map
    try:
        html = build_topic_map(topic_embeddings, summaries)
        with open("dashboard/topic_map.html", "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        print(f"⚠️ Topic map failed: {e}")

    print("BERTopic engine run completed.")


if __name__ == "__main__":
    run_and_persist_bertopic()
