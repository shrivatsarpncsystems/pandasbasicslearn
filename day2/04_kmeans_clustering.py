"""
=============================================================================
DAY 2 - SCRIPT 4: K-MEANS CLUSTERING (UNSUPERVISED SEGMENTATION)
=============================================================================
Goal: Group unlabelled data points into K clusters based on similarity,
      with NO target variable and NO teacher labels!
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("=" * 70)
print(">>> 1. CREATING UNLABELLED CUSTOMER DATASET")
print("=" * 70)

# Simulate 90 retail customers with 2 metrics: Annual Income ($k) and Spending Score (1 to 100)
# Notice: THERE IS NO 'y' COLUMN! NO TARGET!
np.random.seed(42)

# Cluster 1: High Income, Low Spenders ("Savers / Wealth Preservers")
c1_income = np.random.normal(90, 8, 30)
c1_spend = np.random.normal(25, 6, 30)

# Cluster 2: Low Income, High Spenders ("Carefree Shoppers")
c2_income = np.random.normal(35, 7, 30)
c2_spend = np.random.normal(80, 8, 30)

# Cluster 3: High Income, High Spenders ("VIP / Luxury Shoppers")
c3_income = np.random.normal(105, 9, 30)
c3_spend = np.random.normal(85, 7, 30)

income = np.concatenate([c1_income, c2_income, c3_income])
spend = np.concatenate([c1_spend, c2_spend, c3_spend])

df = pd.DataFrame({"Annual_Income_k": income.round(1), "Spending_Score": spend.round(1)})
print("First 6 Customers (Notice there are NO LABELS):")
print(df.head(6))


# ---------------------------------------------------------------------------
# 2. RUNNING K-MEANS CLUSTERING (K = 3)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 2. RUNNING K-MEANS ALGORITHM (K = 3)")
print("=" * 70)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# Notice: We only pass X! There is no y!
df["Assigned_Cluster"] = kmeans.fit_predict(df[["Annual_Income_k", "Spending_Score"]])

print("\n--- Cluster Centers (Centroids / Average Profile of Each Group) ---")
centroids = pd.DataFrame(
    kmeans.cluster_centers_, 
    columns=["Avg_Income_k", "Avg_Spend_Score"],
    index=["Cluster 0", "Cluster 1", "Cluster 2"]
)
print(centroids.round(1))


# ---------------------------------------------------------------------------
# 3. INTERPRETING THE DISCOVERED CLUSTERS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 3. BUSINESS INSIGHT FROM UNSUPERVISED DISCOVERY")
print("=" * 70)
for cluster_id in range(3):
    count = (df["Assigned_Cluster"] == cluster_id).sum()
    avg_inc = df.loc[df["Assigned_Cluster"] == cluster_id, "Annual_Income_k"].mean()
    avg_sp = df.loc[df["Assigned_Cluster"] == cluster_id, "Spending_Score"].mean()
    print(f"Cluster {cluster_id}: {count} customers -> Avg Income: ${avg_inc:.1f}k, Avg Spending Score: {avg_sp:.1f}/100")


# ---------------------------------------------------------------------------
# 4. VISUALIZING THE CLUSTERS & CENTROIDS
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 6))
colors = ["#2563eb", "#10b981", "#f59e0b"]
labels = ["Cluster 0", "Cluster 1", "Cluster 2"]

for cluster_id in range(3):
    cluster_points = df[df["Assigned_Cluster"] == cluster_id]
    plt.scatter(
        cluster_points["Annual_Income_k"], 
        cluster_points["Spending_Score"], 
        color=colors[cluster_id], 
        label=labels[cluster_id],
        alpha=0.7,
        s=50
    )

# Plot Centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0], 
    kmeans.cluster_centers_[:, 1], 
    color="#111827", 
    marker="X", 
    s=200, 
    label="Cluster Centers (Centroids)"
)

plt.title("K-Means Clustering: Automatic Customer Segmentation", fontsize=14, fontweight="bold")
plt.xlabel("Annual Income ($k)")
plt.ylabel("Spending Score (1 to 100)")
plt.legend()
plt.tight_layout()

plot_file = "day2/kmeans_clusters.png"
plt.savefig(plot_file, dpi=180)
plt.close()
print(f"\n-> Saved cluster map to: {plot_file}")
print("=" * 70)
