# Machine Learning Foundations & Data Science

A practical repository documenting daily progress from Python & Pandas to Machine Learning engineering.

---

## 📅 Roadmap Overview

### Day 1: Python, Pandas & Data Preprocessing
- **Pandas Core:** Series vs. DataFrames, `.loc` vs. `.iloc`, CSV and Excel file I/O.
- **Data Cleaning:** Deduplication, missing value imputation (median vs. mode), dirty string parsing with regex, and outlier capping.
- **Data Manipulation:** Multi-condition filtering, sorting, multi-metric `groupby`, and table joins (`pd.merge`).
- **Exploratory Data Analysis (EDA):** Statistical summaries, correlation matrices, and 4-panel Seaborn visual dashboards.
- **Machine Learning Preprocessing:** Feature selection, One-Hot Encoding (`pd.get_dummies`), train-test splitting (stratified), and feature scaling (`StandardScaler`).
- **Hands-on Capstone:** End-to-end pipeline training a Random Forest Classifier.
- **Reference Handbook:** High-yield PDF study notes and LaTeX documentation in `day1/notes/`.

---

## 📂 Project Structure

```
pandasbasicslearn/
├── day1/
│   ├── 01_pandas_basics_and_io.py    # Series, DataFrames, I/O
│   ├── 02_data_cleaning.py           # Missing values, regex, outliers
│   ├── 03_data_manipulation.py       # Filtering, groupby, joins
│   ├── 04_eda_and_visualization.py   # Statistical & visual EDA
│   ├── 05_ml_preprocessing.py        # One-Hot Encoding, scaling, split
│   ├── 06_hands_on_capstone.py       # Full end-to-end pipeline
│   ├── notes/
│   │   ├── Day_1_Complete_Notes.pdf  # PDF Study Handbook
│   │   ├── DAY_1_COMPLETE_NOTES.md   # Markdown Notes
│   │   └── Day_1_Notes.tex           # LaTeX source
│   ├── messy_employee_data.csv       # Raw messy dataset
│   ├── department_budget.xlsx        # Supplementary Excel data
│   └── eda_dashboard.png             # Visual EDA Dashboard
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/shrivatsarpncsystems/<your-repo-name>.git
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl reportlab
   ```
3. Run any module:
   ```bash
   python day1/06_hands_on_capstone.py
   ```
