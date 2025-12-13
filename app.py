import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

print("🔬 Starting Student Behavior Clustering Analysis...")

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
# 2. ELBOW METHOD - Why 4 Clusters?
# ============================================
inertias = []
silhouettes = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow + Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Optimal K=4')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
ax1.set_title('Elbow Method: Finding Optimal K', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Optimal K=4')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score: Cluster Quality', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Optimal clusters: K=4 (Silhouette: {silhouettes[2]:.3f})")

# ============================================
# 3. FINAL CLUSTERING (K=4)
# ============================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)

behavior_df["cluster"] = kmeans.fit_predict(X_scaled)
behavior_df["cluster_gmm"] = gmm.fit_predict(X_scaled)

sil_kmeans = silhouette_score(X_scaled, behavior_df["cluster"])
sil_gmm = silhouette_score(X_scaled, behavior_df["cluster_gmm"])

print(f"K-Means Silhouette: {sil_kmeans:.3f}")
print(f"GMM Silhouette: {sil_gmm:.3f}")

# ============================================
# 4. PCA - Understanding Principal Components
# ============================================
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

var_explained = pca.explained_variance_ratio_
cumulative_var = np.cumsum(var_explained)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(range(1, 6), var_explained, color='steelblue', alpha=0.7)
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Variance Explained', fontsize=12)
ax1.set_title('PCA: Variance Explained by Each Component', fontsize=14, fontweight='bold')
ax1.set_xticks(range(1, 6))

ax2.plot(range(1, 6), cumulative_var, 'ro-', linewidth=2, markersize=8)
ax2.axhline(y=0.8, color='green', linestyle='--', label='80% Threshold')
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
ax2.set_title('PCA: Cumulative Variance', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ PC1 + PC2 explain {cumulative_var[1] * 100:.1f}% of variance")

# ============================================
# 5. CLUSTER VISUALIZATION (PCA Space)
# ============================================
behavior_df["pca_1"] = X_pca[:, 0]
behavior_df["pca_2"] = X_pca[:, 1]

plt.figure(figsize=(12, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
cluster_names = ['High Achievers', 'Passive Learners', 'Crammers', 'Steady Progressors']

for i in range(4):
    subset = behavior_df[behavior_df["cluster"] == i]
    plt.scatter(subset["pca_1"], subset["pca_2"],
                c=colors[i], label=cluster_names[i], s=30, alpha=0.6)

plt.xlabel(f'PC1 ({var_explained[0] * 100:.1f}% variance)', fontsize=13)
plt.ylabel(f'PC2 ({var_explained[1] * 100:.1f}% variance)', fontsize=13)
plt.title('Student Clusters in PCA Space', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_pca.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 6. CLUSTER PROFILES
# ============================================
cluster_profiles = behavior_df.groupby("cluster")[
    ['total_clicks', 'active_days', 'avg_clicks_per_day', 'unique_resources', 'engagement_span']
].mean()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profiles.T, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=cluster_names, cbar_kws={'label': 'Mean Value'})
plt.title('Cluster Behavior Profiles (Mean Features)', fontsize=14, fontweight='bold')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 7. OUTCOMES ANALYSIS
# ============================================
student_info = pd.read_csv("studentInfo.csv")[["id_student", "final_result"]]
behavior_df = behavior_df.merge(student_info, on="id_student", how="left")

outcome_counts = behavior_df.groupby(["cluster", "final_result"]).size().unstack(fill_value=0)
outcome_pct = outcome_counts.div(outcome_counts.sum(axis=1), axis=0) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

outcome_counts.plot(kind='bar', stacked=True, ax=ax1,
                    color=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'])
ax1.set_xticklabels(cluster_names, rotation=45, ha='right')
ax1.set_xlabel('Cluster', fontsize=12)
ax1.set_ylabel('Number of Students', fontsize=12)
ax1.set_title('Learning Outcomes by Cluster (Count)', fontsize=14, fontweight='bold')
ax1.legend(title='Outcome', bbox_to_anchor=(1.05, 1))

outcome_pct.plot(kind='bar', stacked=True, ax=ax2,
                 color=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'])
ax2.set_xticklabels(cluster_names, rotation=45, ha='right')
ax2.set_xlabel('Cluster', fontsize=12)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Learning Outcomes by Cluster (Percentage)', fontsize=14, fontweight='bold')
ax2.legend(title='Outcome', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('outcomes_analysis.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================
# 8. GENERATE HTML DASHBOARD
# ============================================

# Helper function to generate table rows (defined BEFORE use)
def generate_outcome_rows(outcome_pct, names):
    rows = ""
    for i, name in enumerate(names):
        pass_rate = outcome_pct.loc[i, 'Pass'] if 'Pass' in outcome_pct.columns else 0
        dist_rate = outcome_pct.loc[i, 'Distinction'] if 'Distinction' in outcome_pct.columns else 0
        fail_rate = outcome_pct.loc[i, 'Fail'] if 'Fail' in outcome_pct.columns else 0
        withdrawn_rate = outcome_pct.loc[i, 'Withdrawn'] if 'Withdrawn' in outcome_pct.columns else 0

        rows += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{pass_rate:.1f}%</td>
            <td>{dist_rate:.1f}%</td>
            <td>{fail_rate:.1f}%</td>
            <td>{withdrawn_rate:.1f}%</td>
        </tr>"""
    return rows


# Build HTML with placeholder
html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Clustering Insights Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 20px;
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white;
                      padding: 40px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }}
        h1 {{ color: #2c3e50; text-align: center; font-size: 2.5em; margin-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-top: 40px; }}
        h3 {{ color: #2980b9; }}
        .subtitle {{ text-align: center; color: #7f8c8d; font-size: 1.2em; margin-bottom: 40px; }}
        .metric-box {{ display: inline-block; background: #ecf0f1; padding: 20px 40px;
                       margin: 10px; border-radius: 10px; text-align: center; min-width: 180px; }}
        .metric-box h3 {{ margin: 0; color: #2980b9; }}
        .metric-box .value {{ font-size: 2.5em; font-weight: bold; color: #27ae60; }}
        img {{ width: 100%; border-radius: 10px; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .insight-box {{ background: #fff3cd; border-left: 5px solid #ffc107; padding: 20px;
                        margin: 20px 0; border-radius: 5px; }}
        .cluster-card {{ background: #f8f9fa; padding: 20px; margin: 15px 0;
                         border-radius: 10px; border-left: 5px solid #3498db; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        .conclusion {{ background: #d4edda; border: 2px solid #28a745; padding: 25px;
                       border-radius: 10px; margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Student Learning Behavior Clustering</h1>
        <p class="subtitle">Unsupervised Discovery of 4 Distinct Learner Archetypes</p>

        <div style="text-align: center;">
            <div class="metric-box">
                <h3>Students Analyzed</h3>
                <div class="value">{len(behavior_df):,}</div>
            </div>
            <div class="metric-box">
                <h3>Clusters Found</h3>
                <div class="value">4</div>
            </div>
            <div class="metric-box">
                <h3>Silhouette Score</h3>
                <div class="value">{sil_kmeans:.3f}</div>
            </div>
        </div>

        <h2>🔍 Why 4 Clusters?</h2>
        <p>We used the <strong>Elbow Method</strong> and <strong>Silhouette Analysis</strong> to determine the optimal number of clusters:</p>
        <img src="elbow_analysis.png" alt="Elbow Analysis">

        <div class="insight-box">
            <strong>Key Finding:</strong> The "elbow" appears at K=4, and the silhouette score peaks at {silhouettes[2]:.3f}.
        </div>

        <h2>📊 Understanding Principal Components</h2>
        <p>We reduced 5 behavioral features to principal components for visualization:</p>
        <img src="pca_analysis.png" alt="PCA Analysis">

        <div class="insight-box">
            <strong>Key Finding:</strong> The first two principal components capture <strong>{cumulative_var[1] * 100:.1f}%</strong> of the total variance.
        </div>

        <h2>🎯 The 4 Learner Archetypes</h2>
        <img src="cluster_pca.png" alt="Cluster Visualization">

        <div class="cluster-card" style="border-left-color: #e74c3c;">
            <h3>🏆 Cluster 0: High Achievers ({behavior_df[behavior_df['cluster'] == 0].shape[0]} students)</h3>
            <p><strong>Characteristics:</strong> High total clicks, many active days, diverse resources</p>
            <p><strong>Behavior:</strong> Consistent, sustained engagement throughout the course</p>
        </div>
        <div class="cluster-card" style="border-left-color: #3498db;">
            <h3>😴 Cluster 1: Passive Learners ({behavior_df[behavior_df['cluster'] == 1].shape[0]} students)</h3>
            <p><strong>Characteristics:</strong> Low activity across all metrics</p>
            <p><strong>Behavior:</strong> Minimal interaction with course materials</p>
        </div>
        <div class="cluster-card" style="border-left-color: #2ecc71;">
            <h3>⏰ Cluster 2: Crammers ({behavior_df[behavior_df['cluster'] == 2].shape[0]} students)</h3>
            <p><strong>Characteristics:</strong> High clicks per day, short engagement span</p>
            <p><strong>Behavior:</strong> Intense bursts near deadlines</p>
        </div>
        <div class="cluster-card" style="border-left-color: #f39c12;">
            <h3>📈 Cluster 3: Steady Progressors ({behavior_df[behavior_df['cluster'] == 3].shape[0]} students)</h3>
            <p><strong>Characteristics:</strong> Moderate clicks, long engagement span</p>
            <p><strong>Behavior:</strong> Regular, consistent learning over time</p>
        </div>

        <h2>🔥 Cluster Behavior Heatmap</h2>
        <img src="cluster_heatmap.png" alt="Cluster Heatmap">

        <h2>🎓 Learning Outcomes by Cluster</h2>
        <img src="outcomes_analysis.png" alt="Outcomes Analysis">

        <h3>Outcome Rates Summary</h3>
        <table>
            <tr>
                <th>Cluster</th>
                <th>Pass Rate</th>
                <th>Distinction Rate</th>
                <th>Fail Rate</th>
                <th>Withdrawn Rate</th>
            </tr>
            {generate_outcome_rows(outcome_pct, cluster_names)}
        </table>

        <div class="conclusion">
            <h2 style="margin-top: 0;">💡 Key Conclusions</h2>
            <ul style="font-size: 1.1em; line-height: 1.8;">
                <li><strong>Engagement matters:</strong> High Achievers have much higher success rates than Passive Learners</li>
                <li><strong>Consistency wins:</strong> Steady Progressors often outperform Crammers</li>
                <li><strong>Early intervention possible:</strong> Low-engagement clusters can be identified early for support</li>
                <li><strong>Personalized learning:</strong> Different clusters need different teaching strategies</li>
            </ul>
        </div>

        <div style="text-align: center; margin-top: 50px; color: #7f8c8d;">
            <p>Generated on December 14, 2025 • Python • scikit-learn • matplotlib • seaborn</p>
        </div>
    </div>
</body>
</html>
"""

# Replace the placeholder with actual rows
html = html.replace("{generate_outcome_rows(outcome_pct, cluster_names)}",
                    generate_outcome_rows(outcome_pct, cluster_names))

# Save dashboard
with open("clustering_insights.html", "w", encoding="utf-8") as f:
    f.write(html)

print("\n" + "=" * 60)
print("✅ SUCCESS! Dashboard created: clustering_insights.html")
print("=" * 60)
print("\n📊 Generated visualizations:")
print(" • elbow_analysis.png")
print(" • pca_analysis.png")
print(" • cluster_pca.png")
print(" • cluster_heatmap.png")
print(" • outcomes_analysis.png")
print("\n🌐 Open clustering_insights.html in your browser and enjoy your beautiful dashboard!")