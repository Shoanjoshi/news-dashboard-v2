import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyvis.network import Network

# ================================================================
# DEBUG MODE
# ================================================================
DEBUG = True


def debug(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")


# ================================================================
# Make repo root importable
# ================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dashboard_template import write_dashboard_html, build_topic_table_html


# ================================================================
# Helpers
# ================================================================
def load_json(path, label=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label or 'file'} not found: {path}")
    debug(f"Loaded {label or path}: {os.path.getsize(path)} bytes")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_inputs():
    debug("Loading topics.json, theme_signals.json and articles.csv ...")
    topics = load_json("topics.json", "topics")
    theme_signals = load_json("theme_signals.json", "theme_signals")
    articles_df = pd.read_csv("articles.csv")
    debug(f"Topics loaded: {len(topics)}")
    debug(f"Themes loaded : {len(theme_signals)}")
    debug(f"Articles loaded: {len(articles_df)}")
    return topics, theme_signals, articles_df


# ================================================================
# NETWORK — WEF-style natural layout + readable labels + white outline
# ================================================================
def build_network(topics, theme_signals, articles_df):

    debug("Building PyVis network...")

    nt = Network(
        height="1250px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111111",
        directed=False,
    )

    # Enable physics so ForceAtlas2 can run
    nt.toggle_physics(True)

    # ============================================================
    # THEME NODES (large bubbles + white outline)
    # ============================================================
    debug("Adding theme nodes...")

    for theme, vals in theme_signals.items():
        vol = safe_float(vals.get("volume", 0))
        size = 120 + vol * 0.35        # MUCH larger labels + bubble

        nt.add_node(
            theme,
            label=theme,
            shape="dot",
            size=size,
            color="rgba(240,150,90,0.92)",
            borderWidth=8,
            borderWidthSelected=10,
            color_border="#FFFFFF",      # white outline
            font={"size": 90, "face": "Arial", "bold": True, "color": "#111"}
        )

    # ============================================================
    # TOPIC NODES (medium bubbles, limited labels)
    # ============================================================
    debug("Adding topic nodes...")

    sorted_topics = sorted(
        topics.keys(),
        key=lambda t: topics[t].get("topicality", topics[t]["article_count"]),
        reverse=True
    )
    visible_labels = set(sorted_topics[:10])

    for tid, data in topics.items():
        topicality = safe_float(data.get("topicality", 1))
        size = 45 + (topicality ** 0.55) * 4
        label = data["title"] if tid in visible_labels else ""

        nt.add_node(
            tid,
            label=label,
            shape="dot",
            size=size,
            color="rgba(90,140,210,0.92)",
            borderWidth=4,
            borderWidthSelected=6,
            color_border="#FFFFFF",
            font={"size": 45, "face": "Arial"}
        )

    # ============================================================
    # EDGES — affinity corrected
    # ============================================================
    debug("Creating edges...")

    for theme, vals in theme_signals.items():
        aff = vals.get("topic_affinity_pct", {})

        for tid, tdata in topics.items():
            bid = str(tdata["bertopic_id"])
            pct = safe_float(aff.get(bid, 0))

            if pct <= 0:
                continue

            width = 1 + pct * 10
            alpha = 0.25 + pct * 0.6

            nt.add_edge(
                theme,
                tid,
                width=width,
                color=f"rgba(70,100,160,{alpha})",
                smooth={"type": "dynamic"}
            )

    # ============================================================
    # Use PyVis built-in ForceAtlas2 (no JSON config needed)
    # ============================================================
    nt.force_atlas_2based(
        gravity=-40,
        central_gravity=0.002,
        spring_length=150,
        spring_strength=0.05,
        damping=0.65,
        overlap=0.1
    )

    # ============================================================
    # SAVE
    # ============================================================
    os.makedirs("dashboard", exist_ok=True)
    output_file = "dashboard/network_institutional.html"
    nt.save_graph(output_file)

    debug(f"Network saved → {output_file} ({os.path.getsize(output_file)} bytes)")
    return "network_institutional.html"


# ================================================================
# THEME SCATTER — unchanged
# ================================================================
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
            sizes = list(np.interp(volumes, (vmin, vmax), (28, 85)))
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
                line=dict(color="white", width=3),
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


# ================================================================
# HEATMAP — unchanged
# ================================================================
def build_heatmap(topics, theme_signals):
    topic_ids = sorted(topics.keys(), key=lambda t: int(topics[t]["bertopic_id"]))
    topic_titles = [topics[t]["title"] for t in topic_ids]
    theme_names = list(theme_signals.keys())

    z = []
    for th in theme_names:
        aff = theme_signals[th].get("topic_affinity_pct", {})
        row = []
        for tid in topic_ids:
            bid = str(topics[tid]["bertopic_id"])
            row.append(safe_float(aff.get(bid, 0)))
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=topic_titles,
            y=theme_names,
            colorscale="Blues",
            zmin=0,
            zmax=1,
        )
    )
    fig.update_layout(
        title="<b>Theme × Topic Affinity Heatmap</b>",
        height=420,
        margin=dict(l=80, r=20, t=60, b=120),
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


# ================================================================
# MAIN
# ================================================================
def main():
    debug("=== BEGIN DASHBOARD BUILD ===")

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

    debug("Dashboard build complete.")


if __name__ == "__main__":
    main()
