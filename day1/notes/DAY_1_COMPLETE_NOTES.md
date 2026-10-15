# Day 1: Python, Pandas, and Data Preprocessing for Machine Learning

---

## 1. Core Foundations: Series vs. DataFrames

### 1.1 What is a Pandas Series?
A **Series** is a one-dimensional array-like object containing an array of data and an associated array of data labels, called its **Index**.
- Formula representation: A mapping $f: \text{Index} \to \text{Value}$.
- Vectorized operations apply element-wise without manual loops.

```python
import pandas as pd

# Creating a Series
s = pd.Series([75000, 82000, 95000], index=["Alice", "Bob", "Charlie"], name="Salary")
print(s["Alice"])  # Access by label
print(s * 1.10)    # Vectorized 10% bonus
```

### 1.2 What is a Pandas DataFrame?
A **DataFrame** is a two-dimensional labeled data structure with columns of potentially different types (like a spreadsheet or SQL table).
- Mathematically, a collection of Series sharing the same row index.

### 1.3 Label vs. Position Indexing (`.loc` vs `.iloc`)
| Method | Type of Indexing | Example |
| :--- | :--- | :--- |
| **`.loc`** | Label-based | `df.loc["row_label", "col_name"]` or `df.loc[df['Age'] > 30]` |
| **`.iloc`** | Integer-position based | `df.iloc[0:5, 1:3]` (first 5 rows, columns index 1 and 2) |

---

## 2. File I/O: CSV, Excel, and JSON

```python
# Reading files
df_csv = pd.read_csv("data.csv")
df_excel = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df_json = pd.read_json("data.json")

# Writing files (always set index=False unless index has meaning)
df.to_csv("clean_data.csv", index=False)
df.to_excel("clean_data.xlsx", index=False)
```

---

## 3. Data Cleaning Masterclass

### 3.1 Duplicate Handling
Duplicates bias machine learning models by giving redundant weight to certain samples.
```python
# Identify duplicates
print(df.duplicated(subset=["Emp_ID"]).sum())

# Drop duplicates
df = df.drop_duplicates(subset=["Emp_ID"], keep="first")
```

### 3.2 Handling Missing Values: Drop vs. Impute
- **Dropping (`df.dropna()`)**: Only suitable if missing rows account for $< 3\%$ of total data and are missing completely at random (MCAR).
- **Imputing (`df.fillna()`)**:
  - **Numeric Features**: Use **Median** if data is skewed or has outliers; use **Mean** for normal distributions.
  - **Categorical Features**: Use **Mode** (most frequent) or fill with a dedicated category `"Unknown"`.

```python
# Median imputation for numeric
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

# Constant imputation for categorical
df["Department"] = df["Department"].fillna("Unknown")
```

### 3.3 Text Normalization & Regex Cleansing
Real-world data has whitespace, currency symbols, and mixed casing:
```python
# 1. Strip whitespace
df["Name"] = df["Name"].str.strip()

# 2. Standardize casing
df["Department"] = df["Department"].str.upper()

# 3. Clean currency strings: "$75,000" -> 75000.0
df["Salary"] = pd.to_numeric(
    df["Salary"].astype(str).str.replace(r"[\$,]", "", regex=True),
    errors="coerce"  # turns unparseable text into NaN
)
```

### 3.4 Outlier Detection and Treatment
- **Interquartile Range (IQR) Method**:
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
- **Domain Logic Capping**: Replace biologically or physically impossible values (e.g. `Age = 250`).

---

## 4. Data Manipulation

### 4.1 Filtering
```python
# Multi-condition boolean masking (parentheses required)
filtered = df[(df["Salary"] > 70000) & (df["Department"] == "IT")]

# Membership check
tech_df = df[df["Department"].isin(["IT", "FINANCE"])]

# Query method
subset = df.query("Salary > 80000 and Age <= 35")
```

### 4.2 GroupBy & Multi-metric Aggregations
```python
dept_metrics = df.groupby("Department").agg(
    Avg_Salary=("Salary", "mean"),
    Median_Salary=("Salary", "median"),
    Employee_Count=("Emp_ID", "count")
)
```

### 4.3 Merging & Joining DataFrames
| Join Type | Pandas Syntax | Behavior |
| :--- | :--- | :--- |
| **Left Join** | `pd.merge(A, B, on="col", how="left")` | Keeps all records from A, fills missing with NaN |
| **Inner Join** | `pd.merge(A, B, on="col", how="inner")` | Keeps only records with matching keys in both |

---

## 5. Exploratory Data Analysis (EDA)

1. **Five-Number Summary**: `df.describe()` yields count, mean, std, min, 25%, 50% (median), 75%, and max.
2. **Correlation Analysis**:
   $$\rho_{X, Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$
   `df.corr(numeric_only=True)`
3. **Key Visualizations**:
   - **Histograms / KDE**: Check feature distributions (skewness).
   - **Boxplots**: Detect outliers and interquartile spread.
   - **Countplots**: Check categorical balance and target variable distribution.
   - **Heatmaps**: Spot multicollinearity between predictive features.

---

## 6. Preprocessing for Machine Learning (In Simple Terms)

> **The Golden Law:** A Machine Learning model is just a math formula:
> $$\text{Prediction} = (W_1 \times \text{Feature}_1) + (W_2 \times \text{Feature}_2) + \dots + \text{Bias}$$
> Because it is pure math, it cannot understand words, gets biased by huge numbers, and will cheat if given exam questions in advance!

---

### Concept 1: The Secret Answer ($y$) vs. The Clues ($X$)
* **Target ($y$):** The mystery we want to predict (e.g., `y = df['Left_Company']` &rarr; 1 = Quit, 0 = Stayed).
* **Features ($X$):** The clues the model is allowed to inspect (Age, Salary, Experience, Rating, Department).
* **Why drop `Emp_ID`, `Name`, and `Dept_Head`?**
  - If a detective says: *"Employee #104 named David quit, so any person named David will quit!"*, that is bad logic.
  - Names and IDs are unique badges. Keeping them causes the model to memorize labels instead of discovering real human patterns.
* **Why drop `Left_Company` from $X$?**
  - Because `Left_Company` is the secret answer! If it remains in $X$, the model cheats and learns nothing.

---

### Concept 2: Turning Words into Numbers (One-Hot Encoding)
* **Why not label `IT = 1, HR = 2, Sales = 3`?**
  - Computers do arithmetic! If you do that, the model calculates that `Sales (3)` is 3x bigger than `IT (1)`, and `IT (1) + HR (2) = Sales (3)`. That is absurd!
* **The Solution (Light Switches):**
  - Create separate 0/1 indicator columns for each department:
    - If Alice is in IT &rarr; `Dept_IT = 1`, `Dept_HR = 0`, `Dept_Sales = 0`.
  - `drop_first=True` drops 1 column because if `Dept_IT=0` and `Dept_HR=0`, the model already knows the person is in Sales. This avoids mathematical redundancy (the dummy variable trap).

---

### Concept 3: Train-Test Split (Homework vs. Surprise Exam)
* **Training Set (75%):** The practice textbook exercises the model studies.
* **Testing Set (25%):** The surprise final exam. The model has never seen these rows before.
* **`stratify=y`:** Guarantees that both the training set and the test set have the exact same ratio of quitters (33%) and stayers (67%).

---

### Concept 4: Feature Scaling (The Giant vs. The Ant)
* **The Problem:** Look at an employee's numbers:
  - $\text{Salary} = \$90,000$ (A giant!)
  - $\text{Performance Rating} = 4.5$ (A tiny ant!)
  - When multiplying weights, $\$90,000$ completely crushes $4.5$. The model will ignore Rating entirely, even though Rating is critical!
* **The Solution (`StandardScaler`):**
  - Converts numbers into standard deviations from the average:
    $$z = \frac{\text{Value} - \text{Mean}}{\text{Std Dev}}$$
  - Average salary becomes **0**. High salary becomes **+1.5**. Low salary becomes **-1.2**.
  - Average rating becomes **0**. High rating becomes **+1.4**.
  - Now Salary and Rating compete on an equal, fair playing field!
* **The "No Peeking" Rule:**
  - `fit_transform(X_train)`: Calculate average from homework and scale it.
  - `transform(X_test)`: Use homework average on the final exam. **Never calculate average from the test exam!** (Prevents data leakage).

---

## 7. Day 1 Quick Reference Cheat Sheet

| Task | Code Snippet |
| :--- | :--- |
| Check Dimensions | `df.shape` |
| Column Types & Nulls | `df.info()` |
| Count Missing Values | `df.isnull().sum()` |
| Remove Duplicates | `df.drop_duplicates(subset=['id'], keep='first')` |
| Strip Whitespace | `df['col'] = df['col'].str.strip()` |
| Parse Numeric String | `pd.to_numeric(df['col'].str.replace('$',''), errors='coerce')` |
| Filter Data | `df[(df['col1'] > 10) & (df['col2'] == 'A')]` |
| GroupBy Aggregation | `df.groupby('category')['value'].mean()` |
| One-Hot Encoding | `pd.get_dummies(df, columns=['category'], drop_first=True)` |
| Train-Test Split | `train_test_split(X, y, test_size=0.2, stratify=y)` |
| Scale Features | `scaler.fit_transform(X_train)` and `scaler.transform(X_test)` |
