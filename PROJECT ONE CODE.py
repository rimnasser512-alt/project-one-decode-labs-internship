# ============================================================
# DATA SCIENCE - PROJECT 1
# Advanced EDA & Feature Engineering
# DecodeLabs Industrial Training Kit - Batch 2026
# ============================================================
# Dataset: Titanic (loaded directly from the internet)
# What this script does:
#   1. Loads the dataset and explores it
#   2. Handles missing values (empty cells)
#   3. Handles outliers (weird extreme numbers)
#   4. Engineers 3 new features (new columns)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

# ── Make plots look nice ──────────────────────────────────
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (10, 5)


# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================
print("=" * 60)
print("STEP 1: Loading the Titanic Dataset")
print("=" * 60)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"\n✅ Dataset loaded successfully!")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nFirst 5 rows:")
print(df.head())


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis (EDA)")
print("=" * 60)

# Basic info
print("\n📊 Dataset Info:")
print(df.info())

print("\n📊 Basic Statistics:")
print(df.describe())

# Check missing values
print("\n📊 Missing Values per Column:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])

# Plot missing values
plt.figure()
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("Missing Values Heatmap\n(Yellow = Missing)", fontsize=14)
plt.tight_layout()
plt.savefig("plot1_missing_values.png", dpi=150)
plt.close()
print("\n✅ Saved: plot1_missing_values.png")


# ============================================================
# STEP 3: HANDLING MISSING VALUES
# ============================================================
# Rule from the project (the Decision Matrix):
#   < 5%  missing → Drop rows
#   5-20% missing → Statistical Imputation (Median or Group-Wise)
#   > 20% missing → KNN Imputation
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Handling Missing Values")
print("=" * 60)

df_clean = df.copy()

# --- AGE column: ~20% missing → use Median imputation ---
age_missing_pct = df_clean["Age"].isnull().mean() * 100
print(f"\n'Age' missing: {age_missing_pct:.1f}% → Using Median Imputation")
age_median = df_clean["Age"].median()
df_clean["Age"] = df_clean["Age"].fillna(age_median)
print(f"   Filled with median age: {age_median}")

# --- EMBARKED column: <1% missing → Drop those rows ---
embarked_missing_pct = df_clean["Embarked"].isnull().mean() * 100
print(f"\n'Embarked' missing: {embarked_missing_pct:.1f}% → Dropping rows")
df_clean = df_clean.dropna(subset=["Embarked"])

# --- CABIN column: ~77% missing → Drop the whole column ---
cabin_missing_pct = df_clean["Cabin"].isnull().mean() * 100
print(f"\n'Cabin' missing: {cabin_missing_pct:.1f}% → Dropping entire column")
df_clean = df_clean.drop(columns=["Cabin"])

# Verify no more missing values in key columns
print("\n✅ Missing values after cleaning:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])
print("   (No output above = all clean!)")


# ============================================================
# STEP 4: HANDLING OUTLIERS USING IQR
# ============================================================
# IQR method:
#   Lower Bound = Q1 - 1.5 * IQR
#   Upper Bound = Q3 + 1.5 * IQR
#   Values outside these bounds = outliers
#   We CAP them (Winsorization) instead of deleting rows
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Handling Outliers (IQR Method + Winsorization)")
print("=" * 60)

def cap_outliers_iqr(dataframe, column):
    """Detects and caps outliers using the IQR method."""
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers_count = ((dataframe[column] < lower) | (dataframe[column] > upper)).sum()
    # Cap the values (Winsorization)
    dataframe[column] = np.clip(dataframe[column], lower, upper)
    print(f"   '{column}': {outliers_count} outliers capped → [{lower:.2f}, {upper:.2f}]")
    return dataframe

# Apply to numeric columns
for col in ["Age", "Fare", "SibSp", "Parch"]:
    # Box plot before capping
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(df_clean[col].dropna())
    axes[0].set_title(f"{col} — Before Capping")
    
    df_clean = cap_outliers_iqr(df_clean, col)
    
    axes[1].boxplot(df_clean[col].dropna())
    axes[1].set_title(f"{col} — After Capping")
    
    plt.suptitle(f"Outlier Treatment: {col}", fontsize=13)
    plt.tight_layout()
    fname = f"plot2_outlier_{col.lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

print(f"\n✅ Saved box plots for each column")


# ============================================================
# STEP 5: FEATURE ENGINEERING (Creating 3 New Features)
# ============================================================
# We derive new columns from existing ones that might
# help a machine learning model predict survival better.
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Feature Engineering — Creating 3 New Features")
print("=" * 60)

# --- Feature 1: FamilySize ---
# A person's family size on the ship = siblings/spouses + parents/children + themselves
df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
print("\n✅ Feature 1 Created: 'FamilySize'")
print(f"   = SibSp + Parch + 1")
print(f"   Sample values: {df_clean['FamilySize'].value_counts().head(3).to_dict()}")

# --- Feature 2: IsAlone ---
# Binary flag: was this person travelling alone?
df_clean["IsAlone"] = (df_clean["FamilySize"] == 1).astype(int)
print("\n✅ Feature 2 Created: 'IsAlone'")
print(f"   = 1 if FamilySize == 1, else 0")
print(f"   Alone: {df_clean['IsAlone'].sum()} passengers | Not alone: {(df_clean['IsAlone']==0).sum()} passengers")

# --- Feature 3: FarePerPerson ---
# Average fare paid per person in the family group
df_clean["FarePerPerson"] = df_clean["Fare"] / df_clean["FamilySize"]
print("\n✅ Feature 3 Created: 'FarePerPerson'")
print(f"   = Fare / FamilySize")
print(f"   Average fare per person: ${df_clean['FarePerPerson'].mean():.2f}")

# Show the new columns
print("\n📊 New features preview:")
print(df_clean[["Name", "FamilySize", "IsAlone", "FarePerPerson"]].head(8))


# ============================================================
# STEP 6: VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Creating Visualizations")
print("=" * 60)

# Survival by Family Size
plt.figure()
df_clean.groupby("FamilySize")["Survived"].mean().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Survival Rate by Family Size", fontsize=14)
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot3_survival_by_family.png", dpi=150)
plt.close()
print("✅ Saved: plot3_survival_by_family.png")

# Survival by IsAlone
plt.figure()
df_clean.groupby("IsAlone")["Survived"].mean().plot(kind="bar", color=["coral", "steelblue"], edgecolor="black")
plt.title("Survival Rate: Alone vs Not Alone", fontsize=14)
plt.xlabel("Is Alone (0=No, 1=Yes)")
plt.ylabel("Survival Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot4_survival_alone.png", dpi=150)
plt.close()
print("✅ Saved: plot4_survival_alone.png")

# Age distribution after cleaning
plt.figure()
df_clean["Age"].hist(bins=30, color="mediumseagreen", edgecolor="black")
plt.title("Age Distribution After Cleaning", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot5_age_distribution.png", dpi=150)
plt.close()
print("✅ Saved: plot5_age_distribution.png")


# ============================================================
# STEP 7: SAVE THE CLEANED DATASET
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Saving the Cleaned Dataset")
print("=" * 60)

df_clean.to_csv("titanic_cleaned.csv", index=False)
print(f"\n✅ Saved: titanic_cleaned.csv")
print(f"   Final shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
print(f"   New columns added: FamilySize, IsAlone, FarePerPerson")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ PROJECT 1 COMPLETE — SUMMARY")
print("=" * 60)
print("""
What we did:
  1. Loaded the Titanic dataset (891 passengers)
  2. Explored it (EDA) — found missing values & distributions
  3. Handled missing values:
       • Age (20%)  → Filled with median
       • Embarked   → Dropped 2 rows
       • Cabin      → Dropped entire column (77% missing)
  4. Handled outliers using IQR + Winsorization (capping)
       • Age, Fare, SibSp, Parch
  5. Engineered 3 new features:
       • FamilySize    = SibSp + Parch + 1
       • IsAlone       = 1 if FamilySize == 1
       • FarePerPerson = Fare / FamilySize
  6. Created 5 visualizations
  7. Saved cleaned dataset as titanic_cleaned.csv
""")
