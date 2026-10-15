"""
================================================================================
END-TO-END DATA SCIENCE WORKFLOW:
MESSY DATA -> CLEANING -> EXPLORATORY DATA ANALYSIS (EDA) -> ML PREPARATION
================================================================================

This tutorial walks through the 4 stages of preparing data for Machine Learning:
  Step 1: Inspecting the Raw / Messy Data
  Step 2: Cleaning the Data (Handling Nulls, Duplicates, Types, and Text)
  Step 3: Exploratory Data Analysis (EDA) (Stats, Distributions, Relationships)
  Step 4: Machine Learning Preprocessing (Feature Selection, Encoding, Scaling, Train-Test Split)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 1: INSPECTING THE RAW DATA
# ==============================================================================
print("\n" + "=" * 60)
print(">>> STEP 1: INSPECTING THE RAW DATA")
print("=" * 60)

# Load the dataset
df = pd.read_csv("data.csv")

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("First 5 rows:")
print(df.head())

print("\n--- Summary Info ---")
print(df.info())

print("\n--- Missing Values Count per Column ---")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0])
# Observation:
# Type2 has 83 missing values!
# Why? Because in Pokémon, many Pokémon are single-typed.
# If we simply run df.dropna(), we would lose 83 out of 150 rows (>50% of the data)!
# So we should IMPUTE (fill) missing values instead of dropping rows.


# ==============================================================================
# STEP 2: DATA CLEANING
# ==============================================================================
print("\n" + "=" * 60)
print(">>> STEP 2: CLEANING THE DATA")
print("=" * 60)

# Make a copy for cleaning so we don't accidentally mutate raw data
clean_df = df.copy()

# 2.1 Handle Duplicate Rows
before_len = len(clean_df)
clean_df = clean_df.drop_duplicates()
after_len = len(clean_df)
print(f"[Clean 1] Duplicates removed: {before_len - after_len}")

# 2.2 Handle Missing Values (Imputation)
# For Type2, missing means 'None' (Single-type Pokemon).
clean_df["Type2"] = clean_df["Type2"].fillna("None")
print(f"[Clean 2] Missing Type2 filled with 'None'. Remaining nulls: {clean_df['Type2'].isnull().sum()}")

# 2.3 Clean Strings / Whitespace
# In real-world data, text often has accidental spaces like ' Fire ' or 'fire'.
# Let's clean all string columns by stripping spaces and ensuring proper title-casing.
string_columns = ["Name", "Type1", "Type2"]
for col in string_columns:
    clean_df[col] = clean_df[col].astype(str).str.strip().str.capitalize()
print("[Clean 3] String columns stripped of whitespace and normalized.")

# 2.4 Verify Correct Data Types
clean_df["Height"] = pd.to_numeric(clean_df["Height"], errors="coerce")
clean_df["Weight"] = pd.to_numeric(clean_df["Weight"], errors="coerce")
clean_df["Legendary"] = clean_df["Legendary"].astype(int)  # 0 or 1 for binary classification
print("[Clean 4] Data types verified (Height & Weight = float, Legendary = int).")

# Save cleaned dataset for reference
clean_df.to_csv("cleaned_pokemon_data.csv", index=False)
print("-> Cleaned dataset saved as 'cleaned_pokemon_data.csv'!\n")


# ==============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n" + "=" * 60)
print(">>> STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# 3.1 Numerical Summary Statistics
print("--- Numerical Feature Statistics ---")
print(clean_df[["Height", "Weight"]].describe())

# 3.2 Target Variable Exploration
print("\n--- Target Variable Distribution (Legendary) ---")
legendary_counts = clean_df["Legendary"].value_counts()
print(legendary_counts)
print(f"Percentage Legendary: {(legendary_counts.get(1, 0) / len(clean_df)) * 100:.2f}%")
print("Insight: Notice the severe class imbalance! Almost all Pokemon are non-legendary (0).")

# 3.3 GroupBy Analysis
print("\n--- Average Height and Weight by Primary Type ---")
type_stats = clean_df.groupby("Type1")[["Height", "Weight"]].mean().sort_values(by="Weight", ascending=False)
print(type_stats.head())

# 3.4 Outlier & Relationship Inspection
print("\n--- Top 3 Heaviest Pokemon ---")
print(clean_df.nlargest(3, "Weight")[["Name", "Type1", "Height", "Weight"]])

# (Optional visualization - uncomment if you want to display interactive plots)
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# clean_df["Type1"].value_counts().plot(kind="barh", ax=axes[0], title="Pokemon by Primary Type", color="skyblue")
# axes[1].scatter(clean_df["Height"], clean_df["Weight"], c=clean_df["Legendary"], cmap="coolwarm", alpha=0.7)
# axes[1].set_xlabel("Height (m)")
# axes[1].set_ylabel("Weight (kg)")
# axes[1].set_title("Height vs Weight (Colored by Legendary)")
# plt.tight_layout()
# plt.show()


# ==============================================================================
# STEP 4: PREPARING THE DATA FOR MACHINE LEARNING
# ==============================================================================
print("\n" + "=" * 60)
print(">>> STEP 4: PREPARING FOR MACHINE LEARNING")
print("=" * 60)

"""
ML Question: Can we predict whether a Pokemon is LEGENDARY (0 or 1)
based on its Height, Weight, and Types?

Why can't we feed 'clean_df' directly to an ML model?
1. Identifiers ('No', 'Name') are unique labels - they don't predict generic patterns and cause leakage.
2. 'Type1' and 'Type2' are text strings - ML algorithms only accept numbers.
3. 'Height' and 'Weight' have vastly different scales (e.g. Height ~1m vs Weight ~100kg).
   Many algorithms (like SVM, KNN, Neural Networks, Logistic Regression) get biased toward large numbers.
"""

# 4.1 Separate Features (X) and Target (y)
target_col = "Legendary"
feature_df = clean_df.drop(columns=["No", "Name", target_col])  # Drop IDs and Target
y = clean_df[target_col]

print(f"Initial Features Shape: {feature_df.shape}")
print("Feature columns before encoding:", feature_df.columns.tolist())

# 4.2 One-Hot Encoding Categorical Variables
# Convert categorical text ('Type1', 'Type2') into binary numeric columns (0 and 1)
X_encoded = pd.get_dummies(feature_df, columns=["Type1", "Type2"], drop_first=True)
print(f"Features Shape after One-Hot Encoding: {X_encoded.shape}")

# 4.3 Train / Test Split
# We split the data BEFORE scaling to prevent Data Leakage!
print("\n--- 4.3 Train / Test Split (How it works conceptually) ---")

# Method A: Pure Pandas / NumPy (Shows you what happens under the hood!)
np.random.seed(42)
test_ratio = 0.2
shuffled_indices = np.random.permutation(len(X_encoded))
test_set_size = int(len(X_encoded) * test_ratio)

test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]

X_train_manual = X_encoded.iloc[train_indices]
X_test_manual = X_encoded.iloc[test_indices]
y_train_manual = y.iloc[train_indices]
y_test_manual = y.iloc[test_indices]

print(f"Manual Split -> Train: {len(X_train_manual)}, Test: {len(X_test_manual)}")

# Manual Feature Scaling: Standardize numeric features (Height, Weight)
# Formula: z = (x - mean) / std
X_train_manual = X_train_manual.copy()
X_test_manual = X_test_manual.copy()

# Crucial concept: Calculate mean & std on TRAIN set only (avoids data leakage)!
train_mean = X_train_manual[["Height", "Weight"]].mean()
train_std = X_train_manual[["Height", "Weight"]].std()

X_train_manual[["Height", "Weight"]] = (X_train_manual[["Height", "Weight"]] - train_mean) / train_std
X_test_manual[["Height", "Weight"]] = (X_test_manual[["Height", "Weight"]] - train_mean) / train_std

print("\nManual Scaled Features (First 3 Train Rows):")
print(X_train_manual[["Height", "Weight"]].head(3))

# Method B: Using scikit-learn (Industry Standard)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    print("\n--- 4.4 Using Scikit-Learn (Industry Standard) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Scikit-Learn Split -> Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Feature Scaling: Standardize numeric features (Height, Weight)
    scaler = StandardScaler()
    numeric_cols = ["Height", "Weight"]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("\nScaled Feature Samples (Train set):")
    print(X_train_scaled[numeric_cols].head(3))

    # 4.5 Training a Machine Learning Model!
    print("\n" + "-" * 50)
    print(">>> BONUS: Training a Machine Learning Model (Random Forest)!")
    print("-" * 50)
    
    model = RandomForestClassifier(random_state=42, n_estimators=50)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

except ImportError:
    print("\n[NOTE] scikit-learn is finishing installation.")
    print("Meanwhile, the manual Pandas/NumPy preparation above succeeded!")

print("\n" + "=" * 60)
print("SUCCESS: Full Data Cleaning, EDA, & ML Preparation Pipeline Completed!")
print("=" * 60)
