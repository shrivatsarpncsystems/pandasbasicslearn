"""
=============================================================================
DAY 1 - MODULE 1: PANDAS FOUNDATIONS (SERIES, DATAFRAMES, & FILE I/O)
=============================================================================
Goal: Understand what Series and DataFrames actually are, how to inspect them,
and how to load/save CSV, Excel, and JSON files without errors.
"""

import pandas as pd
import numpy as np

print("=" * 70)
print(">>> 1. WHAT IS A PANDAS SERIES?")
print("=" * 70)
# A Series is a 1-Dimensional labeled array (like a single column with an index)
salaries = pd.Series([75000, 82000, 95000, 60000], index=["Alice", "Bob", "Charlie", "David"], name="Salary")
print(salaries)
print("\nAccessing by label: salaries['Alice'] =", salaries["Alice"])
print("Accessing by position: salaries.iloc[0] =", salaries.iloc[0])
print("Vectorized operation (bonus 10%):")
print(salaries * 1.10)


print("\n" + "=" * 70)
print(">>> 2. WHAT IS A PANDAS DATAFRAME?")
print("=" * 70)
# A DataFrame is a 2-Dimensional labeled data structure (table with rows & columns)
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Department": ["IT", "HR", "IT", "Marketing"],
    "Salary": [75000, 82000, 95000, 60000],
    "Experience": [3, 5, 8, 2]
}
df = pd.DataFrame(data)
print(df)

print("\n--- Key DataFrame Attributes ---")
print("Shape (rows, cols):", df.shape)
print("Columns:", df.columns.tolist())
print("Data Types (dtypes):\n", df.dtypes)
print("Index:", df.index)


print("\n" + "=" * 70)
print(">>> 3. SELECTION & INDEXING (.loc vs .iloc)")
print("=" * 70)
# .loc uses LABELS / NAMES
# .iloc uses INTEGER POSITIONS (0-indexed)

print("Select column as Series: df['Salary']")
print(df["Salary"].values)

print("\nSelect multiple columns: df[['Name', 'Salary']]")
print(df[["Name", "Salary"]])

print("\nSelect row 0 by integer index using .iloc[0]:")
print(df.iloc[0])

print("\nSelect rows 0 to 2, specific columns using .iloc[0:2, 0:2]:")
print(df.iloc[0:2, 0:2])


print("\n" + "=" * 70)
print(">>> 4. READING AND WRITING FILES (CSV, EXCEL, JSON)")
print("=" * 70)

# 4.1 CSV
df_csv = pd.read_csv("day1/messy_employee_data.csv")
print(f"Loaded CSV successfully! Rows: {len(df_csv)}, Columns: {len(df_csv.columns)}")

# 4.2 Excel
df_excel = pd.read_excel("day1/department_budget.xlsx")
print(f"Loaded Excel successfully! Departments:\n{df_excel}")

# 4.3 Exporting (Saving)
# Always set index=False unless you specifically want the 0,1,2 row index saved as a column!
df.to_csv("day1/sample_export.csv", index=False)
print("-> Exported sample DataFrame to 'day1/sample_export.csv'!")

print("\n>>> Module 1 Complete! You now understand Series, DataFrames, and I/O.")
