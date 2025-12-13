import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
import os

# --------------------------------------------------------------
# 1. Re-generate PCA plot as base64
# --------------------------------------------------------------
buf = BytesIO()
plt.figure(figsize=(10, 8))
for c in sorted(behavior_df["gmm_cluster"].unique()):
    subset = behavior_df[behavior_df["gmm_cluster"] == c]
    plt.scatter(subset["pca_1"], subset["pca_2"], s=30, alpha=0.8, label=f"Cluster {c}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Visualization of GMM Clusters")
plt.legend()
plt.tight_layout()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
plot_base64 = base64.b64encode(buf.read()).decode('ascii')
plt.close()

# --------------------------------------------------------------
# 2. Tables
# --------------------------------------------------------------
kmeans_summary = behavior_df.drop(columns=["id_student", "pca_1", "pca_2"], errors='ignore')\
                           .groupby("kmeans_cluster").mean()
gmm_summary = behavior_df.drop(columns=["id_student", "pca_1", "pca_2"], errors='ignore')\
                         .groupby("gmm_cluster").mean()

kmeans_summary_html = kmeans_summary.round(3).to_html(classes="table table-striped")
gmm_summary_html = gmm_summary.round(3).to_html(classes="table table-striped")

outcome_gmm = behavior_df.groupby(["gmm_cluster", "final_result"]).size().unstack(fill_value=0)
outcome_html = outcome_gmm.to_html(classes="table table-striped")

# --------------------------------------------------------------
# 3. Radar chart JSON
# --------------------------------------------------------------
features = ['total_clicks', 'active_days', 'avg_clicks_per_day', 'unique_resources', 'engagement_span']

def make_radar_data(df):
    datasets = []
    colors = ['255, 99, 132', '54, 162, 235', '75, 192, 192', '255, 206, 86']
    for i, cluster in enumerate(df.index):
        values = df.loc[cluster].round(3).tolist()
        datasets.append({
            "label": f"Cluster {cluster}",
            "data": values,
            "backgroundColor": f"rgba({colors[i]}, 0.2)",
            "borderColor": f"rgb({colors[i]})",
            "pointBackgroundColor": f"rgb({colors[i]})",
            "borderWidth": 2
        })
    return json.dumps({"labels": features, "datasets": datasets})

kmeans_radar_json = make_radar_data(kmeans_summary)
gmm_radar_json = make_radar_data(gmm_summary)

# --------------------------------------------------------------
# 4. Interactive PCA scatter data
# --------------------------------------------------------------
pca_datasets = []
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i, c in enumerate(sorted(behavior_df["gmm_cluster"].unique())):
    subset = behavior_df[behavior_df["gmm_cluster"] == c]
    points = [{"x": float(row.pca_1), "y": float(row.pca_2)} for _, row in subset.iterrows()]
    pca_datasets.append({
        "label": f"Cluster {c}",
        "data": points,
        "backgroundColor": colors[i % len(colors)],
        "pointRadius": 4
    })
pca_json = json.dumps({"datasets": pca_datasets})

# --------------------------------------------------------------
# 5. Load HTML template and inject everything
# --------------------------------------------------------------
with open("dashboard_template.html", "r", encoding="utf-8") as f:
    html = f.read()

html = html.replace("{{kmeans_silhouette:.3f}}", f"{kmeans_silhouette:.3f}")
html = html.replace("{{gmm_silhouette:.3f}}", f"{gmm_silhouette:.3f}")
html = html.replace("{{gmm_log_likelihood:.2f}}", f"{gmm_log_likelihood:.2f}")
html = html.replace("{{kmeans_summary_html|safe}}", kmeans_summary_html)
html = html.replace("{{gmm_summary_html|safe}}", gmm_summary_html)
html = html.replace("{{outcome_html|safe}}", outcome_html)
html = html.replace("{{kmeans_radar_json|safe}}", kmeans_radar_json)
html = html.replace("{{gmm_radar_json|safe}}", gmm_radar_json)
html = html.replace("{{pca_json|safe}}", pca_json)
html = html.replace("{{plot_base64}}", plot_base64)

with open("clustering_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Dashboard created successfully: clustering_dashboard.html")
print("Open it in your browser and enjoy!")