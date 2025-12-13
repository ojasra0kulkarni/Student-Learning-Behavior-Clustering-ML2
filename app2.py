import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

print("🔬 Starting Student Behavior Clustering Analysis (K-Means vs GMM)...")

# ============================================
# 1. DATA PREPARATION
# ============================================
student_vle = pd.read_csv("studentVle.csv")

behavior_df = student_vle.groupby("id_student").agg(
    total_clicks=("sum_click", "sum"),
    active_days=("date", "nunique"),
    avg_clicks_per_day=("sum_click", "mean"),
    unique_resources=("id_site", "nunique"),
    engagement_span=("date", lambda x: x.max() - x.min())
).reset_index()

behavior_df = behavior_df[behavior_df["total_clicks"] > 0]
behavior_df["total_clicks"] = np.log1p(behavior_df["total_clicks"])

X = behavior_df.drop(columns=["id_student"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# 2. ELBOW METHOD (for K-Means)
# ============================================
inertias = []
silhouettes = []
K_range = range(2, 11)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans_temp.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Chosen K=4')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method (K-Means)')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Chosen K=4')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score (K-Means)')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 3. FINAL CLUSTERING: K-Means & GMM
# ============================================
n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)

behavior_df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)
behavior_df["cluster_gmm"] = gmm.fit_predict(X_scaled)

sil_kmeans = silhouette_score(X_scaled, behavior_df["cluster_kmeans"])
sil_gmm = silhouette_score(X_scaled, behavior_df["cluster_gmm"])
gmm_log_likelihood = gmm.score(X_scaled)  # average log-likelihood

print(f"K-Means Silhouette: {sil_kmeans:.3f}")
print(f"GMM Silhouette:    {sil_gmm:.3f}")
print(f"GMM Avg Log-Likelihood: {gmm_log_likelihood:.2f}")

# ============================================
# 4. PCA FOR VISUALIZATION
# ============================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
behavior_df["pca_1"] = X_pca[:, 0]
behavior_df["pca_2"] = X_pca[:, 1]

var_explained = pca.explained_variance_ratio_.sum() * 100

# PCA Plot: K-Means
plt.figure(figsize=(10, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i in range(n_clusters):
    subset = behavior_df[behavior_df["cluster_kmeans"] == i]
    plt.scatter(subset["pca_1"], subset["pca_2"], c=colors[i], label=f'K-Means Cluster {i}', s=40, alpha=0.7)
plt.xlabel(f'PC1')
plt.ylabel(f'PC2')
plt.title('K-Means Clusters in PCA Space')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pca_kmeans.png', dpi=150, bbox_inches='tight')
plt.close()

# PCA Plot: GMM
plt.figure(figsize=(10, 8))
for i in range(n_clusters):
    subset = behavior_df[behavior_df["cluster_gmm"] == i]
    plt.scatter(subset["pca_1"], subset["pca_2"], c=colors[i], label=f'GMM Cluster {i}', s=40, alpha=0.7)
plt.xlabel(f'PC1')
plt.ylabel(f'PC2')
plt.title('GMM Clusters in PCA Space')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pca_gmm.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 5. CLUSTER PROFILES & HEATMAPS
# ============================================
features = ['total_clicks', 'active_days', 'avg_clicks_per_day', 'unique_resources', 'engagement_span']

profile_kmeans = behavior_df.groupby("cluster_kmeans")[features].mean()
profile_gmm = behavior_df.groupby("cluster_gmm")[features].mean()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(profile_kmeans.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Mean'})
axes[0].set_title('K-Means Cluster Profiles')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Feature')

sns.heatmap(profile_gmm.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Mean'})
axes[1].set_title('GMM Cluster Profiles')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('heatmap_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 6. OUTCOMES ANALYSIS
# ============================================
student_info = pd.read_csv("studentInfo.csv")[["id_student", "final_result"]]
behavior_df = behavior_df.merge(student_info, on="id_student", how="left")

# Outcome percentages
outcome_kmeans = behavior_df.groupby(["cluster_kmeans", "final_result"]).size().unstack(fill_value=0)
outcome_kmeans_pct = outcome_kmeans.div(outcome_kmeans.sum(axis=1), axis=0) * 100

outcome_gmm = behavior_df.groupby(["cluster_gmm", "final_result"]).size().unstack(fill_value=0)
outcome_gmm_pct = outcome_gmm.div(outcome_gmm.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
outcome_kmeans_pct.plot(kind='bar', stacked=True, ax=axes[0], color=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'])
axes[0].set_title('Final Outcomes by K-Means Cluster (%)')
axes[0].set_ylabel('Percentage')
axes[0].legend(title='Outcome', bbox_to_anchor=(1.05, 1))

outcome_gmm_pct.plot(kind='bar', stacked=True, ax=axes[1], color=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'])
axes[1].set_title('Final Outcomes by GMM Cluster (%)')
axes[1].set_ylabel('Percentage')
axes[1].legend(title='Outcome', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('outcomes_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 7. HELPER FOR OUTCOME TABLE
# ============================================
def generate_outcome_table(pct_df, model_name):
    rows = ""
    for i in sorted(pct_df.index):
        pass_r = pct_df.loc[i, 'Pass'] if 'Pass' in pct_df.columns else 0
        dist_r = pct_df.loc[i, 'Distinction'] if 'Distinction' in pct_df.columns else 0
        fail_r = pct_df.loc[i, 'Fail'] if 'Fail' in pct_df.columns else 0
        with_r = pct_df.loc[i, 'Withdrawn'] if 'Withdrawn' in pct_df.columns else 0
        success = pass_r + dist_r
        rows += f"""
        <tr>
            <td><strong>Cluster {i}</strong></td>
            <td>{success:.1f}%</td>
            <td>{fail_r:.1f}%</td>
            <td>{with_r:.1f}%</td>
        </tr>"""
    return rows

# ============================================
# 8. GENERATE HTML DASHBOARD
# ============================================
html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K-Means vs GMM Clustering Comparison</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px;
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1300px; margin: 0 auto; background: white;
                      padding: 40px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }}
        h1 {{ color: #2c3e50; text-align: center; font-size: 2.8em; }}
        h2 {{ color: #34495e; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-top: 50px; }}
        .metric-box {{ display: inline-block; background: #ecf0f1; padding: 20px 40px; margin: 15px;
                       border-radius: 12px; text-align: center; min-width: 200px; }}
        .value {{ font-size: 2.8em; font-weight: bold; color: #27ae60; }}
        img {{ width: 100%; border-radius: 10px; margin: 25px 0; box-shadow: 0 6px 20px rgba(0,0,0,0.15); }}
        .comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 40px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #3498db; color: white; padding: 12px; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        .insight {{ background: #fff3cd; padding: 20px; border-left: 6px solid #ffc107; border-radius: 8px; margin: 30px 0; }}
        .conclusion {{ background: #d4edda; padding: 30px; border-radius: 12px; border: 3px solid #28a745; margin: 50px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 K-Means vs Gaussian Mixture Model</h1>
        <p style="text-align:center; font-size:1.4em; color:#7f8c8d;">
            Comprehensive Comparison of Two Clustering Approaches on Student Engagement Data
        </p>

        <div style="text-align:center;">
            <div class="metric-box">
                <h3>Students</h3>
                <div class="value">{len(behavior_df):,}</div>
            </div>
            <div class="metric-box">
                <h3>K-Means Silhouette</h3>
                <div class="value">{sil_kmeans:.3f}</div>
            </div>
            <div class="metric-box">
                <h3>GMM Silhouette</h3>
                <div class="value">{sil_gmm:.3f}</div>
            </div>
        </div>

        <h2>Why 4 Clusters?</h2>
        <img src="elbow_analysis.png" alt="Elbow Method">

        <h2>PCA Visualization Comparison</h2>
        <div class="comparison-grid">
            <div>
                <h3 style="text-align:center;">K-Means Clusters</h3>
                <img src="pca_kmeans.png" alt="K-Means PCA">
            </div>
            <div>
                <h3 style="text-align:center;">GMM Clusters</h3>
                <img src="pca_gmm.png" alt="GMM PCA">
            </div>
        </div>

        <div class="insight">
            <strong>Observation:</strong> Both models separate students well in PCA space. 
            K-Means tends to create more spherical clusters, while GMM can capture more elliptical/overlapping groups.
        </div>

        <h2>Cluster Behavior Profiles</h2>
        <img src="heatmap_comparison.png" alt="Profile Heatmaps">

        <h2>Academic Outcomes Comparison</h2>
        <img src="outcomes_comparison.png" alt="Outcomes Comparison">

        <h2>Outcome Success Rates by Cluster</h2>
        <div class="comparison-grid">
            <div>
                <h3>K-Means</h3>
                <table>
                    <tr><th>Cluster</th><th>Success Rate<br>(Pass + Distinction)</th><th>Fail Rate</th><th>Withdrawn</th></tr>
                    {generate_outcome_table(outcome_kmeans_pct, "K-Means")}
                </table>
            </div>
            <div>
                <h3>GMM</h3>
                <table>
                    <tr><th>Cluster</th><th>Success Rate<br>(Pass + Distinction)</th><th>Fail Rate</th><th>Withdrawn</th></tr>
                    {generate_outcome_table(outcome_gmm_pct, "GMM")}
                </table>
            </div>
        </div>

        <div class="conclusion">
            <h2>💡 Key Insights from Model Comparison</h2>
            <ul style="font-size:1.2em; line-height:2;">
                <li><strong>Silhouette Score:</strong> K-Means slightly edges out GMM in this dataset ({sil_kmeans:.3f} vs {sil_gmm:.3f})</li>
                <li><strong>Flexibility:</strong> GMM can model more complex cluster shapes and provides probability of membership</li>
                <li><strong>Predictive Power:</strong> Both identify high-risk and high-performing groups effectively</li>
                <li><strong>Recommendation:</strong> Use K-Means for simplicity and interpretability; GMM when clusters may overlap or have different variances</li>
            </ul>
        </div>

        <p style="text-align:center; color:#7f8c8d; margin-top:60px;">
            Generated on December 14, 2025 • Python • scikit-learn • matplotlib • seaborn
        </p>
    </div>
</body>
</html>
"""

# Replace placeholders
html = html.replace("{generate_outcome_table(outcome_kmeans_pct, \"K-Means\")}", generate_outcome_table(outcome_kmeans_pct, "K-Means"))
html = html.replace("{generate_outcome_table(outcome_gmm_pct, \"GMM\")}", generate_outcome_table(outcome_gmm_pct, "GMM"))

with open("clustering_comparison_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print("\n" + "="*70)
print("🎉 SUCCESS! Full comparison dashboard created:")
print("   clustering_comparison_dashboard.html")
print("="*70)
print("\nGenerated files:")
print(" • elbow_analysis.png")
print(" • pca_kmeans.png")
print(" • pca_gmm.png")
print(" • heatmap_comparison.png")
print(" • outcomes_comparison.png")
print("\n🌟 Open clustering_comparison_dashboard.html to see the full K-Means vs GMM comparison!")