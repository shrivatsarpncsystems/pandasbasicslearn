"""
=============================================================================
DAY 1 - MODULE 3: DATA MANIPULATION & AGGREGATIONS
=============================================================================
Topics:
  1. Advanced Filtering (Boolean indexing, .isin(), .query())
  2. Sorting (Single & Multi-column)
  3. GroupBy & Multi-metric Aggregations
  4. Merging DataFrames (Inner, Left, Outer joins with Excel budgets)
"""

import pandas as pd

# Load the clean dataset generated in Module 2
df = pd.read_csv("day1/cleaned_employee_data.csv")
df_dept = pd.read_excel("day1/department_budget.xlsx")
# Standardize Excel department column casing to match df
df_dept["Department"] = df_dept["Department"].str.upper()

print("=" * 70)
print(">>> 1. ADVANCED FILTERING")
print("=" * 70)

# 1.1 Multiple conditions with & (AND) / | (OR)
# Always wrap each condition in parentheses ()!
high_earning_young = df[(df["Salary"] >= 80000) & (df["Age"] <= 35)]
print("Employees earning >= $80k AND Age <= 35:")
print(high_earning_young[["Name", "Department", "Salary", "Age"]])

# 1.2 Using .isin() for matching a list of categories
tech_finance = df[df["Department"].isin(["IT", "FINANCE"])]
print(f"\nEmployees in IT or FINANCE: {len(tech_finance)} found.")

# 1.3 Using .query() (Very readable syntax!)
top_performers = df.query("Rating >= 4.5 and Remote_Worker == 'Yes'")
print("\nTop Remote Performers (Rating >= 4.5 & Remote):")
print(top_performers[["Name", "Department", "Rating"]])


print("\n" + "=" * 70)
print(">>> 2. SORTING VALUES")
print("=" * 70)

# Sort by Department alphabetically, then Salary descending
sorted_df = df.sort_values(by=["Department", "Salary"], ascending=[True, False])
print(sorted_df[["Department", "Name", "Salary"]].head(8))


print("\n" + "=" * 70)
print(">>> 3. GROUPBY & AGGREGATIONS")
print("=" * 70)

# Calculate summary metrics per Department
dept_summary = df.groupby("Department").agg(
    Avg_Salary=("Salary", "mean"),
    Median_Salary=("Salary", "median"),
    Avg_Experience=("Experience_Years", "mean"),
    Total_Employees=("Emp_ID", "count")
).round(2)
print("Department Summary Statistics:")
print(dept_summary)


print("\n" + "=" * 70)
print(">>> 4. MERGING DATAFRAMES (JOINS)")
print("=" * 70)

# Merge our employee data with the department budget Excel table
# We use a LEFT join so we don't lose any employees even if their dept budget isn't found
merged_df = pd.merge(df, df_dept, on="Department", how="left")

print("Merged DataFrame Preview (with Dept Head & Budget):")
print(merged_df[["Name", "Department", "Dept_Head", "Budget_Million", "Salary"]].head(6))

# Save the enriched dataset
merged_df.to_csv("day1/enriched_employee_data.csv", index=False)
print("\n-> Saved enriched data to 'day1/enriched_employee_data.csv'!")
print("\n>>> Module 3 Complete! You can now filter, sort, group, and merge like a pro.")
