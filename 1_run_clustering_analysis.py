import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
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
# CLUSTER SUMMARIES
# -----------------------------
print("\nK-Means cluster summary:")
print(
    behavior_df
    .drop(columns=["id_student"])
    .groupby("kmeans_cluster")
    .mean()
)

print("\nGMM cluster summary:")
print(
    behavior_df
    .drop(columns=["id_student"])
    .groupby("gmm_cluster")
    .mean()
)

# -----------------------------
# PCA VISUALIZATION (GMM) - prepare columns for later use
# -----------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

behavior_df["pca_1"] = X_pca[:, 0]
behavior_df["pca_2"] = X_pca[:, 1]

# -----------------------------
# OPTIONAL: attach outcomes (post hoc only)
# -----------------------------
student_info = pd.read_csv("studentInfo.csv")[["id_student", "final_result"]]
behavior_df = behavior_df.merge(student_info, on="id_student", how="left")

print("\nOutcome distribution (GMM):")
print(
    behavior_df
    .groupby(["gmm_cluster", "final_result"])
    .size()
    .unstack(fill_value=0)
)

print("\nClustering analysis completed successfully!")
print("Next step: run 2_generate_dashboard.py to create the interactive HTML dashboard.")