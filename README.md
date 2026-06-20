# 📊 Data Science Project 1 — Advanced EDA & Feature Engineering

**DecodeLabs Industrial Training Kit | Batch 2026**

---

## 🎯 Project Goal

Transform raw, messy data into a mathematically clean dataset ready for machine learning algorithms — using pure statistical logic.

---

## 📁 Project Structure

```
data_science_project1/
│
├── project1_eda_feature_engineering.py   ← Main Python script (run this!)
├── titanic_cleaned.csv                   ← Output: cleaned dataset
├── plot1_missing_values.png              ← Missing values heatmap
├── plot2_outlier_age.png                 ← Outlier treatment: Age
├── plot2_outlier_fare.png                ← Outlier treatment: Fare
├── plot2_outlier_sibsp.png               ← Outlier treatment: SibSp
├── plot2_outlier_parch.png               ← Outlier treatment: Parch
├── plot3_survival_by_family.png          ← Survival rate by family size
├── plot4_survival_alone.png              ← Survival: alone vs not alone
├── plot5_age_distribution.png            ← Age distribution after cleaning
└── README.md                             ← This file
```

---

## 📦 Dataset

**Titanic Dataset** — 891 passengers, 12 columns  
Source: [datasciencedojo/datasets](https://github.com/datasciencedojo/datasets)

Loaded directly from the internet inside the script — no manual download needed.

---

## ✅ What This Project Does

### Phase 1 — Handle Missing Values (Input Fidelity)

| Column   | Missing % | Strategy Used             |
|----------|-----------|---------------------------|
| Age      | ~20%      | Median Imputation          |
| Embarked | <1%       | Drop rows                 |
| Cabin    | ~77%      | Drop entire column        |

**Decision rule (from the Missing Data Decision Matrix):**
- `< 5%` missing → Drop rows
- `5–20%` missing → Statistical Imputation (Median)
- `> 20%` missing → Drop or KNN (Cabin dropped here)

---

### Phase 2 — Handle Outliers (IQR Method + Winsorization)

For each numeric column (`Age`, `Fare`, `SibSp`, `Parch`):

```
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values outside these bounds are **capped** (not deleted) using `numpy.clip()`.  
This preserves row count and avoids data loss.

---

### Phase 3 — Feature Engineering (3 New Features)

| New Feature    | Formula                        | Why It's Useful                        |
|----------------|--------------------------------|----------------------------------------|
| `FamilySize`   | `SibSp + Parch + 1`            | Larger families may have lower survival|
| `IsAlone`      | `1 if FamilySize == 1 else 0`  | Alone passengers behave differently    |
| `FarePerPerson`| `Fare / FamilySize`            | Fairer cost comparison across groups   |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### 2. Run the script
```bash
python project1_eda_feature_engineering.py
```

### 3. Check outputs
- `titanic_cleaned.csv` — your cleaned dataset
- `plot1_*.png` to `plot5_*.png` — your visualizations

---

## 🛠️ Skills Demonstrated

- `pandas` — data loading, cleaning, grouping
- `numpy` — vectorized math, clipping
- `matplotlib` / `seaborn` — data visualization
- Statistical imputation (Median)
- Outlier detection & Winsorization (IQR)
- Feature engineering from existing columns

---

## 📚 Key Concepts Learned

- **Missing Data Decision Matrix** — choosing the right imputation strategy
- **IQR (Interquartile Range)** — robust, non-parametric outlier detection
- **Winsorization** — capping outliers instead of deleting rows
- **Feature Engineering** — deriving new predictive signals from raw data
- **EDA (Exploratory Data Analysis)** — understanding data before transforming it

---

*Built as part of the DecodeLabs Data Science Industrial Training Program.*
