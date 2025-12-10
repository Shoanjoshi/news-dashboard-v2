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
# FIXED — USE NEW OPENAI SDK INITIALIZATION
# ============================================================
client = OpenAI()   # uses OPENAI_API_KEY from environment


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
    "AI": "Artificial intelligence, data centers, hyperscalers and automation.",
    "Cyber attacks": "Security breaches and vulnerabilities.",
    "Commercial real estate": "Property market stress and refinancing.",
    "Consumer debt": "Household leverage and affordability issues.",
    "Bank lending and credit risk": "Defaults and regulatory pressure.",
    "Digital assets": "Crypto markets, blockchain, tokenization trends.",
    "Others": "Articles not matching systemic themes.",
}

SIMILARITY_THRESHOLD = 0.20

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

def _normalize(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1
    return mat / n


def fetch_articles():
    docs = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:20]:
                content = (entry.get("summary") or entry.get("description") or entry.get("title") or "")
                if isinstance(content, str) and len(content.strip()) > 50:
                    docs.append(content.strip()[:1200])
        except Exception as e:
            print(f"Feed error {feed}: {e}")
    print("Fetched", len(docs), "articles")
    return docs


def gpt_summary(topic_id, docs):
    """Generate GPT summary + title."""
    if not docs:
        return {"title": f"TOPIC {topic_id}", "summary": "Summary unavailable."}

    articles_block = "\n\n".join([f"ARTICLE {i+1}:\n{d}" for i, d in enumerate(docs)])

    prompt = f"""
Summarize the following cluster of related news articles.

FORMAT:
TITLE: 3–5 words describing the topic
OVERVIEW: 1–2 factual sentences
KEY EXAMPLES:
- Example 1
- Example 2
- Example 3 (optional)

ARTICLES:
{articles_block}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.choices[0].message.content.strip()

        # Extract title (first line after "TITLE:")
        if "TITLE:" in out:
            _, rest = out.split("TITLE:", 1)
            rest = rest.strip()
            lines = rest.splitlines()

            title = lines[0].strip()
            summary_body = "\n".join(lines[1:]).strip()

            return {
                "title": title or f"TOPIC {topic_id}",
                "summary": summary_body or "Summary unavailable.",
            }

    except Exception as e:
        print("GPT error:", e)

    return {"title": f"TOPIC {topic_id}", "summary": "Summary unavailable."}


# ============================================================
# CORE PIPELINE
# ============================================================

def run_bertopic():
    docs = fetch_articles()
    if not docs:
        return [], {}, None, {}, {}, []

    # BERTopic model
    umap_model = UMAP(n_neighbors=30, n_components=2, min_dist=0.0, metric="cosine")
    kmeans = KMeans(n_clusters=15, random_state=42, n_init="auto")
    vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 3))

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=kmeans,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
    )

    topics, _ = model.fit_transform(docs)
    topic_info = model.get_topic_info()
    valid_ids = [t for t in topic_info.Topic if t != -1]

    # Embeddings
    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    art_emb = _normalize(sent_model.encode(docs, show_progress_bar=False))

    summaries = {}
    embeddings = {}
    topic_docs = {}

    for t_id in valid_ids:
        ids = [i for i, t in enumerate(topics) if t == t_id]
        topic_docs[t_id] = ids

        rep_ids = ids[:8]
        docs_for_topic = [docs[i] for i in rep_ids]

        summaries[t_id] = gpt_summary(t_id, docs_for_topic)
        summaries[t_id]["article_count"] = len(ids)

        embeddings[t_id] = model.topic_embeddings_[t_id].tolist()

    # THEME embeddings
    theme_emb = _normalize(sent_model.encode(
        [f"{t}. {THEME_DESCRIPTIONS[t]}" for t in THEMES],
        show_progress_bar=False
    ))

    theme_metrics = {t: {"volume": 0, "articles": set()} for t in THEMES}
    theme_metrics["Others"] = {"volume": 0, "articles": set()}

    dominant_theme = ["Others"] * len(docs)

    # Assign themes
    for i, emb in enumerate(art_emb):
        sims = cosine_similarity([emb], theme_emb)[0]

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score >= SIMILARITY_THRESHOLD:
            dom = THEMES[best_idx]
        else:
            dom = "Others"

        dominant_theme[i] = dom
        theme_metrics[dom]["volume"] += 1
        theme_metrics[dom]["articles"].add(i)

    # Compute theme centrality
    for th in THEMES:
        overlaps = 0
        A = theme_metrics[th]["articles"]
        for other in THEMES:
            if other != th:
                overlaps += len(A.intersection(theme_metrics[other]["articles"]))
        theme_metrics[th]["centrality"] = overlaps

    max_c = max(theme_metrics[t]["centrality"] for t in THEMES) or 1
    for th in THEMES:
        theme_metrics[th]["centrality"] /= max_c
    theme_metrics["Others"]["centrality"] = 0.0

    # Topic weights + theme
    for t_id in valid_ids:
        ids = topic_docs[t_id]
        if not ids:
            w = 1.0
            dom = "Others"
        else:
            weights = [THEME_WEIGHTS.get(dominant_theme[i], 1.0) for i in ids]
            w = float(np.mean(weights))

            counts = Counter(dominant_theme[i] for i in ids)
            dom, _ = counts.most_common(1)[0]

        summaries[t_id]["theme_weight"] = w
        summaries[t_id]["dominant_theme"] = dom

    return docs, summaries, model, embeddings, theme_metrics, topics


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

def run_and_persist_bertopic():
    docs, summaries, model, embeddings, theme_metrics, topics = run_bertopic()

    if not docs:
        print("No docs found, skipping.")
        return

    os.makedirs("dashboard", exist_ok=True)

    # Topics.json
    topics_out = {}
    for t_id, meta in summaries.items():
        tid = f"T{t_id}"
        topics_out[tid] = {
            "topic_id": tid,
            "bertopic_id": int(t_id),
            "title": meta.get("title", tid),
            "summary": meta.get("summary", ""),
            "article_count": meta.get("article_count", 0),
            "topicality": meta.get("article_count", 0),
            "theme_weight": meta.get("theme_weight", 1.0),
            "theme": meta.get("dominant_theme", "Others"),
        }

    with open("topics.json", "w") as f:
        json.dump(topics_out, f, indent=2)

    # Theme signals
    with open("theme_signals.json", "w") as f:
        json.dump(theme_metrics, f, indent=2)

    # Articles
    rows = [{"id": i, "text": d, "topic_id": f"T{topics[i]}"} for i, d in enumerate(docs)]
    pd.DataFrame(rows).to_csv("articles.csv", index=False)

    # Topic map
    try:
        html = build_topic_map(embeddings, summaries)
        with open("dashboard/topic_map.html", "w") as f:
            f.write(html)
    except Exception as e:
        print("Topic map error:", e)


if __name__ == "__main__":
    run_and_persist_bertopic()
    print("BERTopic pipeline completed.")

