"""
=============================================================================
DAY 1 - CAPSTONE CHALLENGE: FULL END-TO-END PIPELINE
=============================================================================
Goal: Take a messy dataset from scratch -> Clean it -> Explore it ->
      Prepare it for ML -> Train & Evaluate a Baseline Machine Learning Model.
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 75)
print("DAY 1 CAPSTONE: END-TO-END MACHINE LEARNING DATA PIPELINE")
print("=" * 75)

# ---------------------------------------------------------------------------
# STAGE 1: LOAD RAW MESSY DATA
# ---------------------------------------------------------------------------
print("\n[STAGE 1] Loading Raw Messy Datasets...")
df = pd.read_csv("day1/messy_employee_data.csv")
df_budget = pd.read_excel("day1/department_budget.xlsx")
print(f"Raw shape: {df.shape[0]} rows, {df.shape[1]} columns")


# ---------------------------------------------------------------------------
# STAGE 2: CLEANING
# ---------------------------------------------------------------------------
print("\n[STAGE 2] Cleaning & Sanitizing Data...")

# 2.1 Deduplication
df = df.drop_duplicates(subset=["Emp_ID"])

# 2.2 Text cleaning & Normalization
df["Name"] = df["Name"].str.strip()
df["Department"] = df["Department"].astype(str).str.strip().str.upper().replace({"NAN": "UNKNOWN"})
df["Remote_Worker"] = df["Remote_Worker"].str.strip().str.capitalize()

# 2.3 Parsing numeric columns
df["Salary"] = pd.to_numeric(df["Salary"].astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# 2.4 Handling missing values (Imputation)
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Experience_Years"] = df["Experience_Years"].fillna(df["Experience_Years"].median())

# 2.5 Outlier capping for Age (valid working age 18 to 70)
valid_age = df.loc[(df["Age"] >= 18) & (df["Age"] <= 70), "Age"].median()
df.loc[(df["Age"] < 18) | (df["Age"] > 70) | (df["Age"].isna()), "Age"] = valid_age

# 2.6 Merging with Department Budget
df_budget["Department"] = df_budget["Department"].str.upper()
df = pd.merge(df, df_budget, on="Department", how="left")
df["Budget_Million"] = df["Budget_Million"].fillna(df["Budget_Million"].median())

print(f"Cleaned dataset rows: {len(df)}. Total remaining nulls: {df.isnull().sum().sum()}")


# ---------------------------------------------------------------------------
# STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------
print("\n[STAGE 3] Exploratory Data Analysis...")
print("Numerical Feature Averages:")
print(df[["Age", "Salary", "Experience_Years", "Rating"]].mean().round(2))

print("\nDepartment Salary Medians:")
print(df.groupby("Department")["Salary"].median())

print(f"\nTarget Variable (Left_Company) Class Distribution:")
print(df["Left_Company"].value_counts())


# ---------------------------------------------------------------------------
# STAGE 4: PREPARING FOR MACHINE LEARNING
# ---------------------------------------------------------------------------
print("\n[STAGE 4] Preprocessing for Machine Learning...")

# 4.1 Separate Target and Features (Drop identifiers)
y = df["Left_Company"]
X = df.drop(columns=["Emp_ID", "Name", "Dept_Head", "Left_Company"])

# 4.2 Encoding Categorical features
X["Remote_Worker"] = X["Remote_Worker"].map({"Yes": 1, "No": 0})
X = pd.get_dummies(X, columns=["Department"], drop_first=True)

# 4.3 Train-Test Split (BEFORE scaling!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4.4 Feature Scaling
num_cols = ["Age", "Salary", "Experience_Years", "Rating", "Budget_Million"]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

print(f"Features ready for ML: {X_train_scaled.shape[1]} input dimensions.")


# ---------------------------------------------------------------------------
# STAGE 5: TRAIN & EVALUATE ML MODEL
# ---------------------------------------------------------------------------
print("\n[STAGE 5] Training Random Forest Classifier Model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Model Test Set Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("=" * 75)
print("SUCCESS: YOU HAVE MASTERED ALL REQUIREMENTS FOR DAY 1!")
print("=" * 75)
