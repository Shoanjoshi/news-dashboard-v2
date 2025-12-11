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
# Delta volume diagnostics + snapshot handling
# ------------------------------------------------------------

SNAPSHOT_PATH = "yesterday_theme_signals.json"


def load_yesterday_snapshot():
    """
    Try to load yesterday_theme_signals.json from repo root.
    Used only to compute Δ volume (%) in the scatter plot.
    """
    print("\n[diag] load_yesterday_snapshot()")
    print("  [diag] CWD:", os.getcwd())
    print("  [diag] Looking for snapshot at:", SNAPSHOT_PATH)

    if not os.path.exists(SNAPSHOT_PATH):
        print("  [diag] Snapshot file does NOT exist yet.")
        return None

    try:
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [diag] Loaded snapshot with {len(data)} themes.")
        return data
    except Exception as e:
        print(f"  [diag] ERROR reading snapshot: {e}")
        return None


def apply_delta_volume(theme_signals, yesterday_signals):
    """
    Compute delta_volume_pct for each theme based on yesterday_signals.

    - If no snapshot or no prior volume: delta = 0.0
    """
    if not yesterday_signals:
        print("[diag] No yesterday snapshot; setting delta_volume_pct = 0.0 for all themes.")
        for t in theme_signals:
            theme_signals[t]["delta_volume_pct"] = 0.0
        return theme_signals

    print("\n[diag] Computing Δ volume vs yesterday:")
    for theme_name, today_vals in theme_signals.items():
        today_vol = safe_float(today_vals.get("volume", 0.0))

        y_vals = yesterday_signals.get(theme_name, {})
        yesterday_vol_raw = y_vals.get("volume", None)

        if yesterday_vol_raw is None:
            delta_pct = 0.0
        else:
            yesterday_vol = safe_float(yesterday_vol_raw, 0.0)
            if yesterday_vol == 0:
                delta_pct = 0.0
            else:
                delta_pct = (today_vol - yesterday_vol) / yesterday_vol * 100.0

        today_vals["delta_volume_pct"] = float(delta_pct)

        print(
            f"  [diag] {theme_name}: "
            f"today={today_vol}, yesterday={yesterday_vol_raw}, "
            f"Δ={delta_pct:.1f}%"
        )

    return theme_signals


def write_yesterday_snapshot(theme_signals):
    """
    Persist today's theme_signals as yesterday_theme_signals.json
    for the next run.
    """
    try:
        with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
            json.dump(theme_signals, f, indent=2)
        print(f"\n[diag] Wrote snapshot for next run to {SNAPSHOT_PATH}")
    except Exception as e:
        print(f"[diag] Failed to write yesterday snapshot: {e}")


# ------------------------------------------------------------
# Network graph — WEF-style zoomed-in layout
# ------------------------------------------------------------

def build_network(topics, theme_signals, articles_df):
    # BIG canvas
    nt = Network(
        height="1200px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111111",
    )

    # Force-based (non-radial) layout
    nt.force_atlas_2based(
        gravity=-45,
        central_gravity=0.004,
        spring_length=190,
        spring_strength=0.07,
        damping=0.55,
        overlap=0.25,
    )

    # 1. THEME NODES — very large labels, white outline
    for th, vals in theme_signals.items():
        vol = safe_float(vals.get("volume", 0))

        nt.add_node(
            th,
            label=th,
            shape="dot",
            size=70 + vol * 0.35,
            color="rgba(244,165,130,0.95)",   # orange fill
            borderWidth=4,
            borderWidthSelected=6,
            color_border="#FFFFFF",          # white outline
            font={"size": 90, "face": "Arial", "bold": True},
        )

    # 2. TOPIC NODES — medium labels, white outline
    sorted_topics = sorted(
        topics.keys(),
        key=lambda t: topics[t].get("topicality", topics[t]["article_count"]),
        reverse=True,
    )
    top10 = set(sorted_topics[:10])  # show labels only for most important topics

    for tid in topics:
        topval = safe_float(topics[tid]["topicality"])
        size = 30 + (topval ** 0.5) * 2.5
        label = topics[tid]["title"] if tid in top10 else ""

        nt.add_node(
            tid,
            label=label,
            shape="dot",
            size=size,
            color="rgba(80,120,190,0.95)",   # blue fill
            borderWidth=3,
            borderWidthSelected=5,
            color_border="#FFFFFF",
            font={"size": 60, "face": "Arial"},
        )

    # 3. EDGES — thickness from affinity, smooth & curved
    for th, vals in theme_signals.items():
        aff = vals.get("topic_affinity_pct", {})

        for tid in topics:
            pct = safe_float(aff.get(str(topics[tid]["bertopic_id"]), 0))
            if pct <= 0:
                continue

            width = 1 + pct * 9
            alpha = 0.28 + pct * 0.55

            nt.add_edge(
                th,
                tid,
                width=width,
                color=f"rgba(70,100,160,{alpha})",
                smooth={"type": "dynamic"},
            )

    # Save HTML
    os.makedirs("dashboard", exist_ok=True)
    nt.save_graph("dashboard/network_institutional.html")

    return "network_institutional.html"


# ------------------------------------------------------------
# Theme scatter — bigger labels + white outlines
# ------------------------------------------------------------

def build_theme_scatter(theme_signals):
    themes = list(theme_signals.keys())

    delta_pct = [safe_float(theme_signals[t].get("delta_volume_pct", 0)) for t in themes]
    centrality = [safe_float(theme_signals[t].get("centrality", 0)) for t in themes]
    volumes = [safe_float(theme_signals[t].get("volume", 0)) for t in themes]

    # Marker sizes from volume
    if volumes:
        vmin, vmax = min(volumes), max(volumes)
        if vmin == vmax:
            sizes = [60] * len(volumes)
        else:
            sizes = list(np.interp(volumes, (vmin, vmax), (40, 120)))
    else:
        sizes = [60] * len(volumes)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=delta_pct,
            y=centrality,
            mode="markers+text",
            text=themes,
            textposition="top center",
            textfont=dict(size=14),
            marker=dict(
                size=sizes,
                color="rgba(227,168,105,0.55)",
                line=dict(color="#FFFFFF", width=2.5),  # white outline
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Δ Volume vs prior run: %{x:.1f}%<br>"
                "Centrality: %{y:.2f}<extra></extra>"
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


# ------------------------------------------------------------
# Theme × Topic heatmap
# ------------------------------------------------------------

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
            zmin=0,
            zmax=1,
            colorbar=dict(title="% overlap"),
        )
    )
    fig.update_layout(
        title="<b>Theme × Topic Affinity Heatmap</b>",
        height=420,
        margin=dict(l=80, r=20, t=60, b=120),
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("\n=== Generate Dashboard: starting ===")
    topics, theme_signals, articles_df = load_inputs()
    print(f"[diag] Loaded {len(topics)} topics, {len(theme_signals)} themes, "
          f"{len(articles_df)} articles.")

    # Load yesterday snapshot + compute deltas
    yesterday_signals = load_yesterday_snapshot()
    theme_signals = apply_delta_volume(theme_signals, yesterday_signals)

    # Build visual pieces
    scatter_html = build_theme_scatter(theme_signals)
    heatmap_html = build_heatmap(topics, theme_signals)
    network_file = build_network(topics, theme_signals, articles_df)
    table_html = build_topic_table_html(topics)

    # Topic map (already generated by BERTopic script)
    topic_map_path = "dashboard/topic_map.html"
    if os.path.exists(topic_map_path):
        with open(topic_map_path, "r", encoding="utf-8") as f:
            topic_map_html = f.read()
    else:
        topic_map_html = "<p>No topic map generated.</p>"

    # Render dashboard HTML
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

    # Snapshot for next run
    write_yesterday_snapshot(theme_signals)

    print("Dashboard build complete.")


if __name__ == "__main__":
    main()
