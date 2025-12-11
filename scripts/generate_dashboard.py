import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================================================
# DEBUG HEADER — CONFIRMS THE FILE IS ACTUALLY EXECUTING
# =====================================================
print("\n🟦 [DEBUG] Running generate_dashboard.py (ACTIVE VERSION)")
print("🟦 [DEBUG] File location:", os.path.abspath(__file__), "\n")

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
    print("🟦 [DEBUG] Loading input JSON + CSV files...")
    topics = load_json("topics.json", "topics")
    theme_signals = load_json("theme_signals.json", "theme_signals")
    articles_df = pd.read_csv("articles.csv")

    print(f"🟩 Loaded {len(topics)} topics")
    print(f"🟩 Loaded {len(theme_signals)} themes")
    print(f"🟩 Loaded {len(articles_df)} articles\n")

    return topics, theme_signals, articles_df


# ------------------------------------------------------------
# Network graph — WEF-like force layout
# ------------------------------------------------------------
def build_network(topics, theme_signals, articles_df):
    from pyvis.network import Network

    print("\n🟦 [DEBUG] Building WEF-style network graph...\n")

    nt = Network(
        height="1500px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=False,
        select_menu=True
    )

    # Strong force layout — non-radial
    nt.force_atlas_2based(
        gravity=-65,
        central_gravity=0.002,
        spring_length=220,
        spring_strength=0.06,
        damping=0.55,
        overlap=0.25
    )

    # ============================================================
    # THEME NODES — large, orange, white outlines
    # ============================================================
    for th, vals in theme_signals.items():
        vol = safe_float(vals.get("volume", 0))

        nt.add_node(
            th,
            label=th,
            shape="dot",
            size=85 + vol * 0.35,
            color={
                "background": "rgba(244,165,130,0.95)",
                "border": "#FFFFFF",
                "highlight": {"background": "rgba(244,165,130,1.0)", "border": "#FFFFFF"},
            },
            borderWidth=5,
            font={"size": 80, "face": "Arial", "bold": True}
        )

    # ============================================================
    # TOPIC NODES — blue, medium size, bold labels for top 10
    # ============================================================
    sorted_topics = sorted(
        topics.keys(),
        key=lambda t: topics[t].get("topicality", topics[t]["article_count"]),
        reverse=True
    )
    top10 = set(sorted_topics[:10])

    for tid in topics:
        topval = safe_float(topics[tid]["topicality"])
        label = topics[tid]["title"] if tid in top10 else ""

        nt.add_node(
            tid,
            label=label,
            shape="dot",
            size=38 + (topval ** 0.5) * 3.0,
            color={
                "background": "rgba(80,120,190,0.95)",
                "border": "#FFFFFF",
                "highlight": {"background": "rgba(80,120,190,1.0)", "border": "#FFFFFF"},
            },
            borderWidth=4,
            font={"size": 52, "face": "Arial"}
        )

    # ============================================================
    # EDGES — smooth, strength scaled by affinity %
    # ============================================================
    for th, vals in theme_signals.items():
        aff = vals.get("topic_affinity_pct", {})

        for tid in topics:
            pct = safe_float(aff.get(str(topics[tid]["bertopic_id"]), 0))
            if pct <= 0:
                continue

            width = 1 + pct * 10
            alpha = 0.25 + pct * 0.6

            nt.add_edge(
                th,
                tid,
                width=width,
                color=f"rgba(70,100,160,{alpha})",
                smooth=True
            )

    # ============================================================
    # SAVE
    # ============================================================
    os.makedirs("dashboard", exist_ok=True)
    outfile = "dashboard/network_institutional.html"
    nt.save_graph(outfile)

    print("🟩 [DEBUG] Network graph saved to:", outfile, "\n")

    return "network_institutional.html"


# ------------------------------------------------------------
# Theme scatter plot (already working)
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
            sizes = list(np.interp(volumes, (vmin, vmax), (28, 88)))
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
                color="rgba(227,168,105,0.40)",
                line=dict(color="#FFFFFF", width=3),
            ),
        )
    )

    fig.update_layout(
        title="<b>Theme Signals — Centrality vs Δ Volume</b>",
        xaxis=dict(title="Δ Volume (%)", showline=True, linecolor="#444"),
        yaxis=dict(title="Centrality", showline=True, linecolor="#444"),
        height=460,
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------
def build_heatmap(topics, theme_signals):
    topic_ids = sorted(topics.keys(), key=lambda t: int(topics[t]["bertopic_id"]))
    topic_titles = [topics[t]["title"] for t in topic_ids]
    theme_names = list(theme_signals.keys())

    z = []
    for th in theme_names:
        aff = theme_signals[th].get("topic_affinity_pct", {})
        row = [safe_float(aff.get(str(topics[t]["bertopic_id"]), 0)) for t in topic_ids]
        z.append(row)

    fig = go.Figure(data=go.Heatmap(z=z, x=topic_titles, y=theme_names,
                                   colorscale="Blues", zmin=0, zmax=1))
    fig.update_layout(
        title="<b>Theme × Topic Affinity Heatmap</b>",
        height=420,
        margin=dict(l=80, r=20, t=60, b=120),
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def main():
    print("\n==============================")
    print("🚀 GENERATE DASHBOARD STARTING")
    print("==============================\n")

    topics, theme_signals, articles_df = load_inputs()

    scatter_html = build_theme_scatter(theme_signals)
    heatmap_html = build_heatmap(topics, theme_signals)
    network_file = build_network(topics, theme_signals, articles_df)
    table_html = build_topic_table_html(topics)

    topic_map_path = "dashboard/topic_map.html"
    topic_map_html = open(topic_map_path).read() if os.path.exists(topic_map_path) else "<p>No topic map generated.</p>"

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

    print("\n==============================")
    print("🎉 DASHBOARD BUILD COMPLETE")
    print("==============================\n")


if __name__ == "__main__":
    main()
