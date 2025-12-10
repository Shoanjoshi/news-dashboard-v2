
import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyvis.network import Network

# ============================================================
# Ensure repo root is importable (so dashboard_template works)
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dashboard_template import (
    write_dashboard_html,
    build_topic_table_html,
)

# ============================================================
# Helpers
# ============================================================

def load_json(path, label=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label or 'file'} not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if label:
        print(f"Loaded {label} from {path}")
    return data


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# Load core inputs
# ============================================================

def load_inputs():
    topics = load_json("topics.json", "topics")
    theme_signals = load_json("theme_signals.json", "theme_signals")
    articles_df = pd.read_csv("articles.csv")
    print(f"Loaded articles.csv with {len(articles_df)} rows")
    return topics, theme_signals, articles_df


# ============================================================
# Compute Δ volume vs prior run for each theme
# Writes values into theme_signals[theme]["delta_volume_pct"]
# ============================================================

def compute_theme_deltas(theme_signals):
    """
    Update theme_signals in-place with delta_volume_pct vs yesterday.

    - If yesterday_theme_signals.json exists, use its volumes as baseline.
    - If not, or yesterday volume is 0, set delta to 0.0.
    """
    yesterday_path = "yesterday_theme_signals.json"
    if os.path.exists(yesterday_path):
        try:
            yesterday = load_json(yesterday_path, "yesterday_theme_signals")
        except Exception as e:
            print(f"⚠️ Could not load {yesterday_path}: {e}")
            yesterday = {}
    else:
        yesterday = {}

    for theme, vals in theme_signals.items():
        v_today = safe_float(vals.get("volume", 0.0))
        v_yest = safe_float(yesterday.get(theme, {}).get("volume", 0.0))

        if v_yest <= 0:
            delta_pct = 0.0
        else:
            delta_pct = (v_today - v_yest) / v_yest * 100.0

        vals["delta_volume_pct"] = delta_pct

    return theme_signals


# ============================================================
# Theme Scatter Plot  (Centrality vs %Δ Volume)
# x-axis: theme_signals[theme]["delta_volume_pct"]
# ============================================================

def build_theme_scatter(theme_signals):
    themes = list(theme_signals.keys())

    delta_pct = [
        safe_float(theme_signals[t].get("delta_volume_pct", 0.0))
        for t in themes
    ]
    centrality = [
        safe_float(theme_signals[t].get("centrality", 0.0))
        for t in themes
    ]
    volumes = [
        safe_float(theme_signals[t].get("volume", 0.0))
        for t in themes
    ]

    # Marker sizes proportional to volume
    if volumes:
        v_min, v_max = min(volumes), max(volumes)
        if v_max == v_min:
            marker_sizes = [40.0] * len(volumes)
        else:
            marker_sizes = list(np.interp(volumes, (v_min, v_max), (28.0, 80.0)))
    else:
        marker_sizes = [40.0] * len(volumes)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=delta_pct,
            y=centrality,
            mode="markers+text",
            text=themes,
            textposition="top center",
            marker=dict(
                size=marker_sizes,
                color="rgba(227,168,105,0.35)",
                line=dict(color="rgba(191,120,52,0.95)", width=2),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Δ Volume vs prior run: %{x:.1f}%<br>"
                "Centrality: %{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(text="<b>Theme Signals — Centrality vs Δ Volume</b>", x=0.5),
        xaxis=dict(
            title="Δ Volume vs prior run (%)",
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="#444",
        ),
        yaxis=dict(
            title="Centrality (overlap-based)",
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="#444",
        ),
        height=420,
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Plotly is loaded globally in dashboard_template.html
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ============================================================
# Theme × Topic Heatmap
# Uses theme_signals[theme]["topic_affinity_pct"][bertopic_id]
# mapped onto topics dict with "bertopic_id" field.
# ============================================================

def build_heatmap(topics, theme_signals):
    # Sort topics by numeric BERTopic id for stable order
    topic_ids = sorted(
        topics.keys(),
        key=lambda tid: int(topics[tid].get("bertopic_id", 0))
    )

    topic_titles = [topics[tid].get("title", tid) for tid in topic_ids]
    theme_names = list(theme_signals.keys())

    z = []
    for th in theme_names:
        row = []
        affinity = theme_signals[th].get("topic_affinity_pct", {})
        for tid in topic_ids:
            bert_id = topics[tid].get("bertopic_id", 0)
            val = safe_float(affinity.get(str(bertopic_id := bert_id), 0.0))
            row.append(val)
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=topic_titles,
            y=theme_names,
            colorscale="Blues",
            colorbar=dict(title="% overlap"),
            zmin=0,
            zmax=1,
            hovertemplate=(
                "Theme: %{y}<br>Topic: %{x}<br>"
                "% overlap: %{z:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(text="<b>Theme × Topic Affinity Heatmap</b>", x=0.5),
        xaxis=dict(showgrid=False, tickangle=40),
        yaxis=dict(showgrid=False),
        height=420,
        plot_bgcolor="white",
        margin=dict(l=80, r=20, t=60, b=120),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


# ============================================================
# WEF-style Network (Themes ↔ Topics + sampled Articles)
# ============================================================

def build_network(topics, theme_signals, articles_df):
    nt = Network(
        height="780px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
    )

    # Natural layout
    nt.barnes_hut()

    # -----------------------------
    # 1. Theme nodes
    # -----------------------------
    for theme, vals in theme_signals.items():
        vol = safe_float(vals.get("volume", 0))
        node_size = 20.0 + vol * 0.2

        nt.add_node(
            theme,
            label=theme,
            shape="dot",
            color="rgba(244,194,159,0.95)",  # light orange
            size=node_size,
        )

    # -----------------------------
    # 2. Topic nodes
    # -----------------------------
    sorted_topics = sorted(
        topics.keys(),
        key=lambda tid: topics[tid].get("topicality", topics[tid].get("article_count", 0)),
        reverse=True,
    )
    top5 = set(sorted_topics[:5])

    for tid, meta in topics.items():
        topicality = safe_float(meta.get("topicality", meta.get("article_count", 0)))
        node_size = 10.0 + (topicality ** 0.5) * 3.0

        label = meta.get("title", tid) if tid in top5 else ""

        nt.add_node(
            tid,
            label=label,
            shape="dot",
            color="rgba(106,142,187,0.95)",  # blue
            size=node_size,
        )

    # -----------------------------
    # 3. Theme ↔ Topic edges
    # -----------------------------
    for theme, vals in theme_signals.items():
        affinity = vals.get("topic_affinity_pct", {})
        for tid in topics.keys():
            bert_id = topics[tid].get("bertopic_id", 0)
            pct = safe_float(affinity.get(str(bertopic_id := bert_id), 0.0))
            if pct <= 0:
                continue

            width = 1.0 + pct * 6.0
            alpha = 0.25 + pct * 0.55  # 0.25 – 0.8
            edge_color = f"rgba(90,120,170,{alpha:.2f})"

            nt.add_edge(
                theme,
                tid,
                value=width,
                width=width,
                color=edge_color,
            )

    # -----------------------------
    # 4. Sampled Article nodes (up to 5 per topic)
    # -----------------------------
    if not articles_df.empty and "topic_id" in articles_df.columns:
        grouped = articles_df.groupby("topic_id")
        for tid, group in grouped:
            if tid not in topics:
                continue

            sample = group.head(5)
            for _, row in sample.iterrows():
                art_id = f"art_{row['id']}"
                nt.add_node(
                    art_id,
                    label="",
                    shape="dot",
                    size=4,
                    color="rgba(100,120,140,0.20)",
                )
                nt.add_edge(
                    tid,
                    art_id,
                    value=1,
                    width=1,
                    color="rgba(100,120,140,0.15)",
                )

    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs("dashboard", exist_ok=True)
    output_path = os.path.join("dashboard", "network_institutional.html")
    nt.save_graph(output_path)
    print(f"Saved network to {output_path}")

    return "network_institutional.html"


# ============================================================
# Main orchestration
# ============================================================

def main():
    print("Building dashboard...")

    topics, theme_signals, articles_df = load_inputs()

    # Compute Δ volume vs prior run for themes (for scatter x-axis)
    theme_signals = compute_theme_deltas(theme_signals)

    # 1) Theme scatter
    theme_scatter_html = build_theme_scatter(theme_signals)

    # 2) Heatmap
    heatmap_html = build_heatmap(topics, theme_signals)

    # 3) Network
    network_file = build_network(topics, theme_signals, articles_df)

    # 4) Topic table
    topic_table_html = build_topic_table_html(topics)

    # 5) Load BERTopic topic map HTML
    topic_map_path = os.path.join("dashboard", "topic_map.html")
    if os.path.exists(topic_map_path):
        with open(topic_map_path, "r", encoding="utf-8") as f:
            topic_map_html = f.read()
        print(f"Loaded topic map from {topic_map_path}")
    else:
        topic_map_html = "<p>Topic map not available.</p>"
        print(f"⚠️ No topic_map.html found at {topic_map_path}")

    # 6) Render final dashboard
    write_dashboard_html(
        topics_today=topics,
        themes_today=theme_signals,
        articles_df=articles_df,
        network_institutional_file=network_file,
        topic_map_html=topic_map_html,
        theme_scatter_html=theme_scatter_html,
        heatmap_html=heatmap_html,
        topic_table_html=topic_table_html,
    )

    print("Dashboard built successfully.")

    # 7) Snapshot today's theme signals for next run's deltas
    with open("yesterday_theme_signals.json", "w", encoding="utf-8") as f:
        json.dump(theme_signals, f, indent=2)
    print("Stored yesterday_theme_signals.json for next run.")


if __name__ == "__main__":
    main()
