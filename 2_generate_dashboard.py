# ============================
# full_dashboard.py
# Run this ONE file only!
# ============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json

print("Starting clustering analysis...")

# load data
student_vle = pd.read_csv("studentVle.csv")

# aggregate behavior per student
behavior_df = student_vle.groupby("id_student").agg(
    total_clicks=("sum_click", "sum"),
    active_days=("date", "nunique"),
    avg_clicks_per_day=("sum_click", "mean"),
    unique_resources=("id_site", "nunique")
).reset_index()

# engagement span
span_df = student_vle.groupby("id_student")["date"].agg(
    engagement_span=lambda x: x.max() - x.min()
).reset_index()

behavior_df = behavior_df.merge(span_df, on="id_student", how="left")

# cleaning
behavior_df = behavior_df[behavior_df["total_clicks"] > 0]
behavior_df["total_clicks"] = np.log1p(behavior_df["total_clicks"])

# feature matrix
X = behavior_df.drop(columns=["id_student"])

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# K-MEANS
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
behavior_df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(X_scaled, behavior_df["kmeans_cluster"])

print("K-Means Silhouette:", kmeans_silhouette)

# -----------------------------
# GAUSSIAN MIXTURE MODEL
# -----------------------------
gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
behavior_df["gmm_cluster"] = gmm.fit_predict(X_scaled)

gmm_silhouette = silhouette_score(X_scaled, behavior_df["gmm_cluster"])
gmm_log_likelihood = gmm.score(X_scaled)

print("GMM Silhouette:", gmm_silhouette)
print("GMM Log-Likelihood:", gmm_log_likelihood)

# -----------------------------
# PCA for visualization
# -----------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
behavior_df["pca_1"] = X_pca[:, 0]
behavior_df["pca_2"] = X_pca[:, 1]

# -----------------------------
# Attach outcomes
# -----------------------------
student_info = pd.read_csv("studentInfo.csv")[["id_student", "final_result"]]
behavior_df = behavior_df.merge(student_info, on="id_student", how="left")

print("\nClustering complete. Now generating dashboard...")

# =============================================
# DASHBOARD GENERATION STARTS HERE
# =============================================

# 1. PCA Plot as base64
buf = BytesIO()
plt.figure(figsize=(10, 8))
for c in sorted(behavior_df["gmm_cluster"].unique()):
    subset = behavior_df[behavior_df["gmm_cluster"] == c]
    plt.scatter(subset["pca_1"], subset["pca_2"], s=40, alpha=0.8, label=f"Cluster {c}")
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)
plt.title("PCA Visualization of GMM Clusters", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
buf.seek(0)
plot_base64 = base64.b64encode(buf.read()).decode('ascii')
plt.close()

# 2. Tables
cols_to_drop = ["id_student", "pca_1", "pca_2", "kmeans_cluster", "gmm_cluster", "final_result"]
kmeans_summary = behavior_df.drop(columns=[c for c in cols_to_drop if c in behavior_df.columns], errors='ignore') \
                           .groupby(behavior_df["kmeans_cluster"]).mean().round(3)
gmm_summary = behavior_df.drop(columns=[c for c in cols_to_drop if c in behavior_df.columns], errors='ignore') \
                         .groupby(behavior_df["gmm_cluster"]).mean().round(3)

kmeans_summary_html = kmeans_summary.to_html(classes="table table-striped table-hover", border=0)
gmm_summary_html = gmm_summary.to_html(classes="table table-striped table-hover", border=0)

outcome_gmm = behavior_df.groupby(["gmm_cluster", "final_result"]).size().unstack(fill_value=0)
outcome_html = outcome_gmm.to_html(classes="table table-striped table-hover", border=0)

# 3. Radar charts
features = ['total_clicks', 'active_days', 'avg_clicks_per_day', 'unique_resources', 'engagement_span']

def make_radar_data(df):
    datasets = []
    colors = ['255, 99, 132', '54, 162, 235', '75, 192, 192', '255, 206, 86']
    for i, cluster in enumerate(sorted(df.index)):
        values = df.loc[cluster].round(3).tolist()
        color = colors[i % len(colors)]
        datasets.append({
            "label": f"Cluster {cluster}",
            "data": values,
            "backgroundColor": f"rgba({color}, 0.2)",
            "borderColor": f"rgb({color})",
            "pointBackgroundColor": f"rgb({color})",
            "borderWidth": 3
        })
    return json.dumps({"labels": features, "datasets": datasets})

kmeans_radar_json = make_radar_data(kmeans_summary)
gmm_radar_json = make_radar_data(gmm_summary)

# 4. PCA scatter data
pca_datasets = []
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i, c in enumerate(sorted(behavior_df["gmm_cluster"].unique())):
    subset = behavior_df[behavior_df["gmm_cluster"] == c]
    points = [{"x": float(row.pca_1), "y": float(row.pca_2)} for _, row in subset.iterrows()]
    pca_datasets.append({
        "label": f"Cluster {c}",
        "data": points,
        "backgroundColor": colors[i % len(colors)],
        "pointRadius": 5,
        "pointHoverRadius": 8
    })
pca_json = json.dumps({"datasets": pca_datasets})

# 5. Fill HTML template
with open("dashboard_template.html", "r", encoding="utf-8") as f:
    html = f.read()

replacements = {
    "{{kmeans_silhouette:.3f}}": f"{kmeans_silhouette:.3f}",
    "{{gmm_silhouette:.3f}}": f"{gmm_silhouette:.3f}",
    "{{gmm_log_likelihood:.2f}}": f"{gmm_log_likelihood:.2f}",
    "{{kmeans_summary_html|safe}}": kmeans_summary_html,
    "{{gmm_summary_html|safe}}": gmm_summary_html,
    "{{outcome_html|safe}}": outcome_html,
    "{{kmeans_radar_json|safe}}": kmeans_radar_json,
    "{{gmm_radar_json|safe}}": gmm_radar_json,
    "{{pca_json|safe}}": pca_json,
    "{{plot_base64}}": plot_base64
}

for placeholder, value in replacements.items():
    html = html.replace(placeholder, value)

with open("clustering_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print("\nSUCCESS! Dashboard created: clustering_dashboard.html")
print("Now open clustering_dashboard.html in your browser!")