import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyvis.network import Network

# Make repo root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dashboard_template import write_dashboard_html, build_topic_table_html


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_json(path, label=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label or 'file'} not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_inputs():
    topics = load_json("topics.json", "topics")
    theme_signals = load_json("theme_signals.json", "theme_signals")
    articles_df = pd.read_csv("articles.csv")
    return topics, theme_signals, articles_df


# ------------------------------------------------------------
# Network graph — WEF-style zoomed-in layout
# ------------------------------------------------------------

def build_network(topics, theme_signals, articles_df):
    # Much larger canvas
    nt = Network(
        height="1100px",           # increased from 780
        width="100%", 
        bgcolor="#ffffff",
        font_color="#222222"
    )

    # Use force simulation (non-radial)
    nt.force_atlas_2based(
        gravity=-40,               # spreads nodes apart
        central_gravity=0.005,
        spring_length=180,         # pushes nodes further apart
        spring_strength=0.06,
        damping=0.6,
        overlap=0.2
    )

    # -----------------------
    # 1. THEME NODES (large)
    # -----------------------
    for th, vals in theme_signals.items():
        vol = safe_float(vals.get("volume", 0))
        nt.add_node(
            th,
            label=th,
            size=55 + vol * 0.3,              # bigger
            shape="dot",
            color="rgba(244,165,130,0.95)",   # soft orange
            font={"size": 32, "face": "Arial"}  # BIG labels
        )

    # -----------------------
    # 2. TOPIC NODES (medium)
    # -----------------------
    sorted_topics = sorted(
        topics.keys(),
        key=lambda t: topics[t].get("topicality", topics[tid]["article_count"]),
        reverse=True
    )
    top10 = set(sorted_topics[:10])

    for tid in topics:
        topval = safe_float(topics[tid]["topicality"])
        size = 25 + (topval ** 0.5) * 2       # bump size

        label = topics[tid]["title"] if tid in top10 else ""

        nt.add_node(
            tid,
            label=label,
            size=size,
            shape="dot",
            color="rgba(80,120,190,0.95)",
            font={"size": 22, "face": "Arial"}  # readable
        )

    # -----------------------
    # 3. EDGES (thicker, smoother)
    # -----------------------
    for th, vals in theme_signals.items():
        aff = vals.get("topic_affinity_pct", {})

        for tid in topics:
            pct = safe_float(aff.get(str(topics[t]["bertopic_id"]), 0))
            if pct <= 0:
                continue

            width = 1 + pct * 8               # stronger
            alpha = 0.25 + pct * 0.55

            nt.add_edge(
                th,
                tid,
                width=width,
                color=f"rgba(70,100,160,{alpha})",
                smooth={"type": "dynamic"}     # curved edges
            )

    # -----------------------
    # 4. OPTIONAL — Hide article dots (cleaner network)
    # -----------------------
    # Comment back in if you want them again
    """
    if "topic_id" in articles_df.columns:
        for tid, grp in articles_df.groupby("topic_id"):
            for _, row in grp.head(3).iterrows():
                aid = f"art_{row['id']}"
                nt.add_node(aid, size=2, color="rgba(120,120,140,0.1)")
                nt.add_edge(tid, aid, width=1, color="rgba(120,120,140,0.15)")
    """

    # -----------------------
    # Save network
    # -----------------------
    os.makedirs("dashboard", exist_ok=True)
    nt.show_buttons(filter_=["physics"])  # ability to adjust if needed
    nt.save_graph("dashboard/network_institutional.html")

    return "network_institutional.html"


    # -----------------------------------------------------
    # SAVE OUTPUT
    # -----------------------------------------------------
    os.makedirs("dashboard", exist_ok=True)
    nt.save_graph("dashboard/network_institutional.html")

    return "network_institutional.html"



# ------------------------------------------------------------
# (unchanged functions below)
# ------------------------------------------------------------

def build_theme_scatter(theme_signals):
    themes = list(theme_signals.keys())

    delta_pct = [safe_float(theme_signals[t].get("delta_volume_pct", 0)) for t in themes]
    centrality = [safe_float(theme_signals[t].get("centrality", 0)) for t in themes]
    volumes = [safe_float(theme_signals[t].get("volume", 0)) for t in themes]

    if volumes:
        vmin, vmax = min(volumes), max(volumes)
        if vmin == vmax:
            sizes = [40] * len(volumes)
        else:
            sizes = list(np.interp(volumes, (vmin, vmax), (28, 80)))
    else:
        sizes = [40] * len(volumes)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=delta_pct,
            y=centrality,
            mode="markers+text",
            text=themes,
            textposition="top center",
            marker=dict(
                size=sizes,
                color="rgba(227,168,105,0.35)",
                line=dict(color="rgba(191,120,52,0.95)", width=2),
            ),
        )
    )

    fig.update_layout(
        title="<b>Theme Signals — Centrality vs Δ Volume</b>",
        xaxis=dict(title="Δ Volume (%)", showline=True, linecolor="#444"),
        yaxis=dict(title="Centrality", showline=True, linecolor="#444"),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_heatmap(topics, theme_signals):
    topic_ids = sorted(topics.keys(), key=lambda t: int(topics[t]["bertopic_id"]))
    topic_titles = [topics[t]["title"] for t in topic_ids]
    theme_names = list(theme_signals.keys())

    z = []
    for th in theme_names:
        row = []
        aff = theme_signals[th].get("topic_affinity_pct", {})
        for t in topic_ids:
            bid = str(topics[t]["bertopic_id"])
            row.append(safe_float(aff.get(bid, 0)))
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=topic_titles,
            y=theme_names,
            colorscale="Blues",
            zmin=0, zmax=1,
        )
    )
    fig.update_layout(
        title="<b>Theme × Topic Affinity Heatmap</b>",
        height=420,
        margin=dict(l=80, r=20, t=60, b=120),
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def main():
    topics, theme_signals, articles_df = load_inputs()

    scatter_html = build_theme_scatter(theme_signals)
    heatmap_html = build_heatmap(topics, theme_signals)
    network_file = build_network(topics, theme_signals, articles_df)
    table_html = build_topic_table_html(topics)

    topic_map_path = "dashboard/topic_map.html"
    if os.path.exists(topic_map_path):
        topic_map_html = open(topic_map_path, "r").read()
    else:
        topic_map_html = "<p>No topic map generated.</p>"

    write_dashboard_html(
        topics_today=topics,
        themes_today=theme_signals,
        articles_df=articles_df,
        network_institutional_file=network_file,
        topic_map_html=topic_map_html,
        theme_scatter_html=scatter_html,
        heatmap_html=heatmap_html,
        topic_table_html=table_html,
    )

    print("Dashboard build complete.")


if __name__ == "__main__":
    main()
