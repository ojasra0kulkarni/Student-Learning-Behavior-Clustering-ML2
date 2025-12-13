import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# basic cleaning
behavior_df = behavior_df[behavior_df["total_clicks"] > 0]
behavior_df["total_clicks"] = np.log1p(behavior_df["total_clicks"])

# feature matrix
X = behavior_df.drop(columns=["id_student"])

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# silhouette scores for K selection
scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

print("Silhouette scores:", scores)

# final clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
behavior_df["cluster"] = kmeans.fit_predict(X_scaled)

# cluster summary (exclude id)
cluster_summary = (
    behavior_df
    .drop(columns=["id_student"])
    .groupby("cluster")
    .mean()
)

print("\nCluster-wise behavior summary:")
print(cluster_summary)

# PCA visualization (must be before any merges)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

behavior_df["pca_1"] = X_pca[:, 0]
behavior_df["pca_2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
for c in sorted(behavior_df["cluster"].unique()):
    subset = behavior_df[behavior_df["cluster"] == c]
    plt.scatter(
        subset["pca_1"],
        subset["pca_2"],
        s=20,
        alpha=0.7,
        label=f"Cluster {c}"
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Visualization of Student Behavior Clusters")
plt.legend()
plt.tight_layout()
plt.show()

# attach outcomes for interpretation only
student_info = pd.read_csv("studentInfo.csv")[["id_student", "final_result"]]
behavior_df = behavior_df.merge(student_info, on="id_student", how="left")

# outcome distribution per cluster
outcome_summary = (
    behavior_df
    .groupby(["cluster", "final_result"])
    .size()
    .unstack(fill_value=0)
)

print("\nFinal result distribution per cluster:")
print(outcome_summary)
