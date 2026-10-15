"""
=============================================================================
DAY 1 - MODULE 5: PREPROCESSING DATA FOR MACHINE LEARNING
=============================================================================
Topics:
  1. Feature Selection (Dropping Identifiers & High-Cardinality text)
  2. Encoding Categorical Variables (Binary & One-Hot Encoding)
  3. Train-Test Split (Avoiding Data Leakage with Stratification)
  4. Feature Scaling (StandardScaler vs MinMaxScaler)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load enriched clean data
# =============================================================================
# CONCEPT 1: SEPARATING THE 'SECRET ANSWER' (y) FROM 'THE CLUES' (X)
# =============================================================================
# Real-Life Analogy:
# Imagine you are a Detective trying to figure out if an employee will QUIT.
# 
# 1. Target (y) = The Mystery / The Answer Key we want the model to guess:
#    Did the employee quit? (1 = Yes, 0 = No)
y = df["Left_Company"]

# 2. Features (X) = The Clues the model is allowed to look at:
#    (Age, Department, Salary, Experience, Rating, Remote_Worker, Budget)
#
# WHY DROP 'Emp_ID', 'Name', and 'Dept_Head'?
# -> If a detective says: "Employee #104 named David quit, so ANYONE named David will quit!",
#    that is terrible logic! Names and IDs are random badges, not human behavior patterns.
#    Keeping them makes the model memorize names instead of learning real rules!
#
# WHY DROP 'Left_Company' FROM X?
# -> Because Left_Company IS the answer! If you leave the answer in the clues,
#    the model will cheat and learn nothing.
X = df.drop(columns=["Emp_ID", "Name", "Dept_Head", "Left_Company"])

print(f"Features (Clues) before encoding:\n{X.columns.tolist()}\n")


print("\n" + "=" * 70)
print(">>> 2. CATEGORICAL ENCODING")
print("=" * 70)

# 2.1 Binary Encoding for 2-value categories (e.g. Remote_Worker: Yes/No)
X["Remote_Worker"] = X["Remote_Worker"].map({"Yes": 1, "No": 0})

# 2.2 One-Hot Encoding for multi-class categories (e.g. Department)
# Converts 'IT', 'HR', 'FINANCE', 'SALES', 'MARKETING' into separate columns (0 or 1)
# drop_first=True prevents multi-collinearity (the dummy variable trap)
X_encoded = pd.get_dummies(X, columns=["Department"], drop_first=True)
print("Features after One-Hot Encoding:")
print(X_encoded.head(3))
print(f"New Feature Matrix Shape: {X_encoded.shape} (Rows: {X_encoded.shape[0]}, Features: {X_encoded.shape[1]})")


print("\n" + "=" * 70)
print(">>> 3. TRAIN / TEST SPLIT (AVOIDING DATA LEAKAGE)")
print("=" * 70)

# Always split BEFORE scaling!
# stratify=y preserves the exact proportion of 0s and 1s in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")


print("\n" + "=" * 70)
print(">>> 4. FEATURE SCALING")
print("=" * 70)

# Continuous columns with vastly different scales:
# e.g., Salary is ~$70,000, while Rating is ~4.0, Age is ~30
numeric_features = ["Age", "Salary", "Experience_Years", "Rating", "Budget_Million"]

# 4.1 StandardScaler (Standardization: z = (x - mean) / std)
scaler = StandardScaler()

# CRITICAL RULE:
# .fit_transform() ONLY on training data!
# .transform() on testing data! (Never fit on test data!)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("Scaled Numerical Features (Train Set - First 3 Rows):")
print(X_train_scaled[numeric_features].head(3).round(2))

# Save the final preprocessed datasets ready for any ML model!
X_train_scaled.to_csv("day1/X_train_ready_for_ml.csv", index=False)
X_test_scaled.to_csv("day1/X_test_ready_for_ml.csv", index=False)
y_train.to_csv("day1/y_train.csv", index=False)
y_test.to_csv("day1/y_test.csv", index=False)

print("\n-> Successfully exported X_train, X_test, y_train, and y_test to CSV!")
print("\n>>> Module 5 Complete! Your dataset is 100% prepared for Machine Learning.")
