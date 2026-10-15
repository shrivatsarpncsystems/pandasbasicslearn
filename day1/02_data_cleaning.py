"""
=============================================================================
DAY 1 - MODULE 2: DATA CLEANING MASTERCLASS
=============================================================================
Topics:
  1. Detecting & Dropping Duplicates
  2. Missing Values (Detection, dropna vs fillna imputation)
  3. String Inconsistencies (whitespace, mixed casing, typos)
  4. Parsing Dirty Numeric Strings ($ signs, commas, text)
  5. Outlier Detection & Treatment
"""

import pandas as pd
import numpy as np

# Load our messy dataset
df = pd.read_csv("day1/messy_employee_data.csv")
print("=" * 70)
print(">>> RAW MESSY DATA PREVIEW")
print("=" * 70)
print(df.head(6))

print("\nMissing values before cleaning:")
print(df.isnull().sum())


# ---------------------------------------------------------------------------
# 1. DUPLICATE DETECTION & REMOVAL
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 1. DUPLICATE HANDLING")
print("=" * 70)
num_dups = df.duplicated(subset=["Emp_ID"]).sum()
print(f"Duplicates found by Emp_ID: {num_dups}")

# Drop duplicates (keeping the first occurrence)
df = df.drop_duplicates(subset=["Emp_ID"], keep="first")
print(f"Rows remaining after dropping duplicates: {len(df)}")


# ---------------------------------------------------------------------------
# 2. STRING CLEANING (WHITESPACE & INCONSISTENT CASING)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 2. STRING CLEANING & CASING")
print("=" * 70)

# 2.1 Name column: has messy spaces like "  Alice Smith "
df["Name"] = df["Name"].str.strip()

# 2.2 Department column: has "IT", "it ", "hr", "HR", and missing values
df["Department"] = df["Department"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
print("Clean Department unique values:", df["Department"].unique())

# 2.3 Remote_Worker: "Yes", "yes", "NO", "no"
# Normalize to standardized Title Case ("Yes" / "No")
df["Remote_Worker"] = df["Remote_Worker"].str.strip().str.capitalize()
print("Clean Remote_Worker unique values:", df["Remote_Worker"].unique())


# ---------------------------------------------------------------------------
# 3. FIXING DIRTY NUMERIC STRINGS ($ signs, commas, strings like "unknown")
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 3. PARSING DIRTY NUMERIC DATA")
print("=" * 70)

# Salary column has '$', ',', and 'N/A'
# Step A: Remove symbols with regex / string replace
df["Salary"] = df["Salary"].astype(str).str.replace(r"[\$,]", "", regex=True)
# Step B: Convert to float, coercing invalid text like 'N/A' into NaN
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
print("Clean Salary (first 5):\n", df["Salary"].head())

# Age column has "unknown", "N/A", and "250"
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")


# ---------------------------------------------------------------------------
# 4. MISSING VALUE IMPUTATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 4. IMPUTING MISSING VALUES")
print("=" * 70)

# Rule of thumb:
# - For numeric columns: Impute with Median (robust to outliers) or Mean
# - For categorical columns: Impute with Mode (most frequent) or "Unknown"

salary_median = df["Salary"].median()
df["Salary"] = df["Salary"].fillna(salary_median)
print(f"Imputed Salary missing values with Median: ${salary_median:,.2f}")

exp_median = df["Experience_Years"].median()
df["Experience_Years"] = df["Experience_Years"].fillna(exp_median)


# ---------------------------------------------------------------------------
# 5. OUTLIER HANDLING (Domain logic & Statistical capping)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> 5. OUTLIER DETECTION & CORRECTION")
print("=" * 70)

# Check Age: 250 is an impossible human age!
print("Ages before correction:", df["Age"].tolist())
# Filter out impossible ages (> 100 or < 18) and replace with median
valid_age_median = df.loc[(df["Age"] >= 18) & (df["Age"] <= 100), "Age"].median()
df.loc[(df["Age"] < 18) | (df["Age"] > 100) | (df["Age"].isna()), "Age"] = valid_age_median
print("Ages after correction:", df["Age"].tolist())


# ---------------------------------------------------------------------------
# SAVE CLEANED DATA
# ---------------------------------------------------------------------------
df.to_csv("day1/cleaned_employee_data.csv", index=False)
print("\n-> Clean dataset exported to 'day1/cleaned_employee_data.csv'!")
print(f"Remaining null values across all columns:\n{df.isnull().sum()}")
print("\n>>> Module 2 Complete! Your data is now fully sanitized.")
