# Student Learning Behavior Clustering (ML2)

## Overview
This project applies unsupervised machine learning to segment students based on their online learning behavior in a Virtual Learning Environment (VLE). Using only interaction data from the LMS, the model discovers distinct learner archetypes — such as High Achievers, Passive Learners, Crammers, and Steady Progressors — **without using academic outcomes during training**.

The pipeline includes:
- Feature engineering from clickstream data
- K-Means and Gaussian Mixture Model (GMM) clustering
- Elbow method + Silhouette analysis for optimal K
- PCA for visualization and interpretation
- Post-hoc analysis linking behavior clusters to final results

A beautiful standalone HTML dashboard is generated with all visualizations, cluster profiles, and model comparison.

## Objectives
- Identify meaningful behavioral segments using unsupervised learning
- Compare hard clustering (K-Means) vs probabilistic clustering (GMM)
- Visualize high-dimensional engagement patterns with PCA
- Reveal relationships between engagement and academic success
- Deliver interpretable, presentation-ready results

## Techniques Used
- Feature engineering & aggregation
- StandardScaler normalization
- K-Means clustering
- Gaussian Mixture Models (GMM)
- Silhouette Score & Elbow Method
- Principal Component Analysis (PCA)
- matplotlib & seaborn visualization
- Standalone HTML report generation

## Dataset
**Open University Learning Analytics Dataset (OULAD)**  
Anonymized data from The Open University online courses.

### Required Files (place in project root)
- `studentVle.csv` → Student interactions with VLE sites
- `studentInfo.csv` → Student demographics and final results

### Download
- Official source: [analyse.kmi.open.ac.uk/open_dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- Mirror on Kaggle: [OULAD dataset](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)

> Note: Files are not included in the repo due to size limits.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python 1_run_clustering_analysis.py   # runs feature engineering + K-Means/GMM clustering, saves results
python 2_generate_dashboard.py        # builds the standalone HTML dashboard (clustering_insights_with_comparison.html)
```

Outputs include PCA/elbow/heatmap plots (`*.png`) and two HTML dashboards: `clustering_comparison_dashboard.html` (model comparison) and `clustering_insights_with_comparison.html` (full report).

> `app.py`, `app2.py`, `app3.py`, `main.py`, `main_backup.py`, and `GMM.py` are earlier iterations/experiments kept for reference — `1_run_clustering_analysis.py` and `2_generate_dashboard.py` are the current pipeline.

