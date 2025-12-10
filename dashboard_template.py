"""
dashboard_template.py
Pure helper module to render dashboard HTML via Jinja2.
No business logic, no imports from generate_dashboard.
"""

import os
from jinja2 import Environment, FileSystemLoader


# ---------------------------------------------------------
# Build Topic Summary Table
# ---------------------------------------------------------
def build_topic_table_html(topics):
    rows = []

    for tid, meta in topics.items():
        title = meta.get("title", tid)
        summ = meta.get("summary", "").replace("\n", "<br>")
        count = meta.get("article_count", 0)

        rows.append(f"""
            <tr>
                <td><b>{tid}</b></td>
                <td>{title}</td>
                <td class="summary-cell">{summ}</td>
                <td>{count}</td>
            </tr>
        """)

    return f"""
    <table class="topic-table">
        <thead>
            <tr>
                <th>Topic</th>
                <th>Title</th>
                <th>Summary</th>
                <th>Articles</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


# ---------------------------------------------------------
# Write Dashboard HTML
# ---------------------------------------------------------
def write_dashboard_html(
    topics_today,
    themes_today,
    articles_df,
    network_institutional_file,
    topic_map_html,
    theme_scatter_html,
    heatmap_html,
    topic_table_html,
):
    """Render dashboard_template.html into dashboard/index.html"""

    os.makedirs("dashboard", exist_ok=True)

    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=False   # IMPORTANT: allows Plotly HTML to render
    )

    template = env.get_template("dashboard_template.html")

    context = {
        "topics_today": topics_today,
        "themes_today": themes_today,
        "articles": articles_df.to_dict(orient="records"),
        "network_institutional": network_institutional_file,
        "topic_map_html": topic_map_html,
        "theme_scatter_html": theme_scatter_html,
        "heatmap_html": heatmap_html,
        "topic_table_html": topic_table_html,
    }

    rendered_html = template.render(context)

    output_path = os.path.join("dashboard", "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print(f"Dashboard successfully written to {output_path}")

