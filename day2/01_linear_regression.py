"""
=============================================================================
DAY 2 - SCRIPT 1: LINEAR REGRESSION (PREDICTING CONTINUOUS NUMBERS)
=============================================================================
Goal: Fit a straight trend line to predict a continuous numeric output,
      and evaluate it using MAE, RMSE, and R-squared.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print(">>> 1. CREATING SAMPLE DATASET (EXPERIENCE VS. SALARY)")
print("=" * 70)

# Simulate 50 realistic employee data points: Salary = Starting + (Experience * 7500) + Noise
np.random.seed(42)
experience = np.random.uniform(1, 15, 50).round(1)
# Real relationship: Base $45,000 + $7,200 per year of experience + some random fluctuation
salary = 45000 + (experience * 7200) + np.random.normal(0, 5000, 50)

df = pd.DataFrame({"Experience_Years": experience, "Salary": salary.round(-2)})
print("First 5 records:")
print(df.head())


# ---------------------------------------------------------------------------
# 2. TRAIN-TEST SPLIT
# ---------------------------------------------------------------------------
X = df[["Experience_Years"]]  # 2D feature matrix
y = df["Salary"]              # 1D target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nSplit complete: {len(X_train)} training samples, {len(X_test)} test samples.")


# ---------------------------------------------------------------------------
# 3. TRAINING LINEAR REGRESSION MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 2. TRAINING THE MODEL (FINDING THE BEST-FIT LINE)")
print("=" * 70)

model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]
intercept = model.intercept_

print(f"Learned Equation: Salary = (${slope:,.2f} * Experience) + ${intercept:,.2f}")
print("Meaning:")
print(f"  • Base starting salary (0 years exp): ${intercept:,.2f}")
print(f"  • Annual raise per year of experience: ${slope:,.2f}")


# ---------------------------------------------------------------------------
# 4. EVALUATION METRICS (MAE, RMSE, R-SQUARED)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 3. EVALUATION METRICS ON UNSEEN TEST EXAM")
print("=" * 70)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"1. Mean Absolute Error (MAE): ${mae:,.2f}")
print("   -> Interpretation: On average, our salary predictions are off by only this amount.")

print(f"\n2. Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print("   -> Interpretation: Penalizes large mistakes more heavily.")

print(f"\n3. R-Squared (R2 Score): {r2 * 100:.2f}%")
print("   -> Interpretation: The model explains over this percentage of salary variation!")


# ---------------------------------------------------------------------------
# 5. MAKING A REAL-WORLD PREDICTION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 4. PREDICTING SALARY FOR A NEW APPLICANT")
print("=" * 70)
new_applicant_exp = [[8.5]]
predicted_sal = model.predict(new_applicant_exp)[0]
print(f"Applicant with 8.5 years of experience: Predicted Salary = ${predicted_sal:,.2f}")


# ---------------------------------------------------------------------------
# 6. VISUALIZE AND SAVE PLOT
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.scatter(X_train, y_train, color="#2563eb", label="Training Data", alpha=0.7)
plt.scatter(X_test, y_test, color="#e11d48", label="Unseen Test Data", marker="s", s=60)
plt.plot(X, model.predict(X), color="#1e293b", linewidth=2.5, label="Best-Fit Line (Linear Regression)")
plt.title("Linear Regression: Experience vs. Salary", fontsize=14, fontweight="bold")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary ($)")
plt.legend()
plt.tight_layout()

plot_file = "day2/linear_regression_fit.png"
plt.savefig(plot_file, dpi=180)
plt.close()
print(f"\n-> Saved visualization to: {plot_file}")
print("=" * 70)
