"""
=============================================================================
DAY 2 - SCRIPT 3: DECISION TREES VS. RANDOM FOREST (THE WISDOM OF THE CROWD)
=============================================================================
Goal: Compare a Single Decision Tree (which can overfit) against a
      Random Forest (committee of trees that prevents overfitting).
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=" * 70)
print(">>> 1. CREATING COMPLEX DATASET (LOAN DEFAULT PREDICTION)")
print("=" * 70)

# Simulate 120 loan applicants: Income, Credit Score, Debt-to-Income Ratio -> Defaulted (1) or Paid (0)
np.random.seed(42)
income = np.random.uniform(25000, 150000, 120)
credit_score = np.random.uniform(450, 850, 120)
debt_ratio = np.random.uniform(0.1, 0.7, 120)

# Realistic rule with noise
default_prob = (
    (credit_score < 600) * 0.45 +
    (debt_ratio > 0.45) * 0.35 +
    (income < 40000) * 0.20
)
defaulted = (np.random.rand(120) < default_prob).astype(int)

df = pd.DataFrame({
    "Income": income.round(-2),
    "Credit_Score": credit_score.round(0),
    "Debt_Ratio": debt_ratio.round(2),
    "Defaulted": defaulted
})

print(df.head())

X = df[["Income", "Credit_Score", "Debt_Ratio"]]
y = df["Defaulted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# ---------------------------------------------------------------------------
# 2. SINGLE DECISION TREE (OBSERVING OVERFITTING)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 2. MODEL A: SINGLE DECISION TREE")
print("=" * 70)

# An unconstrained decision tree splits until every single training sample is memorized
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

tree_train_acc = accuracy_score(y_train, tree.predict(X_train))
tree_test_acc = accuracy_score(y_test, tree.predict(X_test))

print(f"Decision Tree Training Accuracy: {tree_train_acc * 100:.1f}%  <-- Memorized practice homework!")
print(f"Decision Tree Testing Accuracy:  {tree_test_acc * 100:.1f}%  <-- Drops on the real exam (Overfitting!)")

print("\n--- The Flowchart Questions Learned by the Tree (First 3 levels) ---")
tree_rules = export_text(tree, feature_names=list(X.columns), max_depth=2)
print(tree_rules)


# ---------------------------------------------------------------------------
# 3. RANDOM FOREST (COMMITTEE OF 100 TREES VOTING TOGETHER)
# ---------------------------------------------------------------------------
print("=" * 70)
print(">>> 3. MODEL B: RANDOM FOREST ENSEMBLE (100 TREES)")
print("=" * 70)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"Random Forest Training Accuracy: {rf_train_acc * 100:.1f}%")
print(f"Random Forest Testing Accuracy:  {rf_test_acc * 100:.1f}%  <-- Higher generalization score!")

print("\n--- Feature Importance (Which features mattered most to the Forest?) ---")
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": (rf.feature_importances_ * 100).round(2)
}).sort_values(by="Importance", ascending=False)
print(importance_df)

print("\n" + "=" * 70)
print("KEY LESSON:")
print("Single trees are prone to overfitting (100% train vs lower test).")
print("Random Forests average 100 trees together to eliminate noise and boost test accuracy!")
print("=" * 70)
