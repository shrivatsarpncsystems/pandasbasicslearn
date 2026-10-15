"""
=============================================================================
DAY 1 - MODULE 4: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
=============================================================================
Topics:
  1. Five-Number Summary & Statistics
  2. Correlation Analysis
  3. Visualizations using Matplotlib & Seaborn:
     - Distribution Histogram (Salary)
     - Countplot (Target Variable: Left_Company)
     - Boxplot (Salary distribution across Departments)
     - Scatter Plot (Experience vs Salary colored by Churn)
     - Correlation Heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# Load enriched clean data from Module 3
df = pd.read_csv("day1/enriched_employee_data.csv")

print("=" * 70)
print(">>> 1. STATISTICAL EDA")
print("=" * 70)

print("--- Five-Number Summary of Numerical Features ---")
print(df[["Age", "Salary", "Experience_Years", "Rating"]].describe().round(2))

print("\n--- Target Variable Distribution (Left_Company) ---")
target_counts = df["Left_Company"].value_counts()
print(target_counts)
print(f"Attrition Rate: {(target_counts[1] / len(df)) * 100:.1f}%")

print("\n--- Correlation Matrix ---")
corr_matrix = df[["Age", "Salary", "Experience_Years", "Rating", "Budget_Million", "Left_Company"]].corr()
print(corr_matrix.round(2))


print("\n" + "=" * 70)
print(">>> 2. GENERATING EDA VISUALIZATION DASHBOARD")
print("=" * 70)

# Create a 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Day 1 EDA Dashboard: Employee Insights", fontsize=16, fontweight="bold")

# Plot 1: Histogram & KDE of Salary
sns.histplot(df["Salary"], kde=True, color="#2b5c8f", ax=axes[0, 0])
axes[0, 0].set_title("1. Salary Distribution")
axes[0, 0].set_xlabel("Salary ($)")

# Plot 2: Countplot of Department by Churn Status
sns.countplot(data=df, x="Department", hue="Left_Company", palette=["#2ecc71", "#e74c3c"], ax=axes[0, 1])
axes[0, 1].set_title("2. Employee Count & Churn by Department")
axes[0, 1].legend(title="Left Company", labels=["Stayed (0)", "Left (1)"])

# Plot 3: Boxplot of Salary by Department (Outlier Detection)
sns.boxplot(data=df, x="Department", y="Salary", hue="Department", palette="Set2", legend=False, ax=axes[1, 0])
axes[1, 0].set_title("3. Salary Range & Outliers per Department")

# Plot 4: Correlation Heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 1], cbar=True)
axes[1, 1].set_title("4. Feature Correlation Matrix")

plt.tight_layout()

# Save the dashboard image
plot_path = "day1/eda_dashboard.png"
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"-> Successfully saved 4-panel EDA visualization to '{plot_path}'!")
print("\n>>> Module 4 Complete! You now know how to uncover trends, outliers, and patterns.")
