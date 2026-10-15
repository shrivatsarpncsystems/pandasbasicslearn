"""
=============================================================================
DAY 2 - SCRIPT 2: LOGISTIC REGRESSION (YES/NO CLASSIFICATION & PROBABILITIES)
=============================================================================
Goal: Understand how Logistic Regression converts numbers into 0% to 100%
      probabilities, and evaluate it with Accuracy, Precision, Recall & F1.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("=" * 70)
print(">>> 1. CREATING DATASET (STUDY HOURS VS. PASS/FAIL)")
print("=" * 70)

# Simulate 60 students: Hours studied (1 to 10) vs Pass (1) or Fail (0)
np.random.seed(42)
hours = np.sort(np.random.uniform(1, 10, 60))
# Probability of passing increases with hours studied
prob_pass = 1 / (1 + np.exp(-(hours - 5.5) * 1.5))
passed = (np.random.rand(60) < prob_pass).astype(int)

df = pd.DataFrame({"Hours_Studied": hours.round(1), "Passed": passed})
print(df.head(8))
print(f"\nClass balance: {df['Passed'].value_counts().to_dict()} (0 = Fail, 1 = Pass)")


# ---------------------------------------------------------------------------
# 2. TRAIN-TEST SPLIT
# ---------------------------------------------------------------------------
X = df[["Hours_Studied"]]
y = df["Passed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# ---------------------------------------------------------------------------
# 3. TRAINING LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 2. TRAINING LOGISTIC REGRESSION (FITTING S-CURVE)")
print("=" * 70)

model = LogisticRegression()
model.fit(X_train, y_train)

# Test predictions
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Pass)

print("Sample Predictions on Unseen Test Students:")
comparison = pd.DataFrame({
    "Hours": X_test["Hours_Studied"].values,
    "Actual": y_test.values,
    "Calculated_Probability": (y_probs * 100).round(1),
    "Model_Prediction": y_pred
})
print(comparison.head(6))


# ---------------------------------------------------------------------------
# 4. CLASSIFICATION METRICS (ACCURACY, PRECISION, RECALL, F1)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 3. CLASSIFICATION EVALUATION METRICS")
print("=" * 70)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"1. Accuracy:  {acc * 100:.1f}% (Overall correct predictions)")
print(f"2. Precision: {prec * 100:.1f}% (When model says 'Pass', how often is it right?)")
print(f"3. Recall:    {rec * 100:.1f}% (Of all students who actually passed, how many did we catch?)")
print(f"4. F1-Score:  {f1 * 100:.1f}% (Harmonic balance between Precision and Recall)")

print("\n--- Confusion Matrix ---")
print(f"True Negatives (Correct Fails):  {cm[0, 0]} | False Positives (False Alarms):  {cm[0, 1]}")
print(f"False Negatives (Missed Passes): {cm[1, 0]} | True Positives (Correct Passes): {cm[1, 1]}")


# ---------------------------------------------------------------------------
# 5. VISUALIZING THE SIGMOID S-CURVE
# ---------------------------------------------------------------------------
X_smooth = np.linspace(1, 10, 300).reshape(-1, 1)
y_smooth_prob = model.predict_proba(X_smooth)[:, 1]

plt.figure(figsize=(9, 5))
plt.scatter(df["Hours_Studied"], df["Passed"], color="#2563eb", alpha=0.6, label="Actual Students (0 or 1)")
plt.plot(X_smooth, y_smooth_prob, color="#e11d48", linewidth=2.5, label="Logistic S-Curve (Probability of Passing)")
plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="50% Decision Threshold")
plt.title("Logistic Regression: S-Curve Probability Modeling", fontsize=14, fontweight="bold")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing (0.0 to 1.0)")
plt.legend()
plt.tight_layout()

plot_file = "day2/logistic_regression_scurve.png"
plt.savefig(plot_file, dpi=180)
plt.close()
print(f"\n-> Saved S-curve visualization to: {plot_file}")
print("=" * 70)
