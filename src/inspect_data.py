"""
Student Performance Analysis and Prediction System
Dataset expected at: data/student-mat.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. PATHS AND GRAPH STYLE
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10

# Keep all figures open and show them together at the end.
figures = []


def finish_plot(filename):
    """Save the current figure and keep it open for the final display."""
    fig = plt.gcf()
    fig.tight_layout()
    
    figures.append(fig)


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions),
    }
    return metrics, predictions


# ------------------------------------------------------------
# 2. LOAD AND INSPECT DATA
# ------------------------------------------------------------
print("\nSTUDENT PERFORMANCE DATASET")
print("-" * 50)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place student-mat.csv inside the project's data folder."
    )

df = pd.read_csv(DATA_PATH, sep=";")

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())
print("\nColumn Names:")
print(df.columns.tolist())
print("\nMissing Values (total):", int(df.isnull().sum().sum()))
print("Duplicate Rows:", int(df.duplicated().sum()))

# Clean duplicate rows and missing values.
df = df.drop_duplicates().copy()

numeric_columns_all = df.select_dtypes(include=np.number).columns
categorical_columns_all = df.select_dtypes(exclude=np.number).columns

for column in numeric_columns_all:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_columns_all:
    if df[column].isnull().any():
        mode_values = df[column].mode()
        if not mode_values.empty:
            df[column] = df[column].fillna(mode_values.iloc[0])

if "G3" not in df.columns:
    raise ValueError("The dataset must contain the final-grade column G3.")

# Total_Grades is for descriptive analysis only. It is NOT used as a model feature.
if {"G1", "G2", "G3"}.issubset(df.columns):
    df["Total_Grades"] = df[["G1", "G2", "G3"]].sum(axis=1)

print("\nFINAL GRADE STATISTICS")
print("-" * 50)
print("Mean:", round(df["G3"].mean(), 2))
print("Median:", round(df["G3"].median(), 2))
print("Standard deviation:", round(df["G3"].std(), 2))
print("\nStudy Time Statistics:")
if "studytime" in df.columns:
    print(df["studytime"].describe())

print(
    "\nDataset note: this UCI dataset contains an absences count, "
    "not an attendance percentage."
)

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS — GRAPH 1
# Final grade distribution
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
sns.histplot(df["G3"], bins=11, kde=True, color="steelblue")
plt.title("Distribution of Students' Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
finish_plot("01_final_grade_distribution.png")

# ------------------------------------------------------------
# GRAPH 2 — Study time vs final grade
# ------------------------------------------------------------
if "studytime" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.stripplot(
        data=df, x="studytime", y="G3",
        jitter=0.22, alpha=0.55, size=5, color="teal"
    )
    plt.title("Study Time vs Final Grade")
    plt.xlabel("Weekly Study Time Category (1 = lowest, 4 = highest)")
    plt.ylabel("Final Grade (G3)")
    finish_plot("02_study_time_vs_final_grade.png")

# ------------------------------------------------------------
# GRAPH 3 — Absences vs final grade
# ------------------------------------------------------------
if "absences" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.regplot(
        data=df, x="absences", y="G3",
        scatter_kws={"alpha": 0.55, "s": 38},
        line_kws={"color": "darkred", "linewidth": 2},
        ci=None
    )
    plt.title("Absences vs Final Grade")
    plt.xlabel("Number of Absences")
    plt.ylabel("Final Grade (G3)")
    finish_plot("03_absences_vs_final_grade.png")

# ------------------------------------------------------------
# GRAPH 4 — Average final grade by study time
# ------------------------------------------------------------
if "studytime" in df.columns:
    study_summary = (
        df.groupby("studytime")["G3"]
        .agg(["mean", "count"])
        .reindex([1, 2, 3, 4])
        .dropna()
    )
    labels = {
        1: "Less than 2 hours",
        2: "2 to 5 hours",
        3: "5 to 10 hours",
        4: "More than 10 hours",
    }
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        x=study_summary.index.astype(str),
        y=study_summary["mean"],
        color="cornflowerblue"
    )
    ax.set_xticklabels([labels.get(int(x), x) for x in study_summary.index])
    for i, (_, row) in enumerate(study_summary.iterrows()):
        ax.text(
            i, row["mean"] + 0.15,
            f'{row["mean"]:.1f} (n={int(row["count"])})',
            ha="center", va="bottom", fontsize=9
        )
    plt.title("Average Final Grade by Weekly Study Time")
    plt.xlabel("Weekly Study Time")
    plt.ylabel("Average Final Grade (G3)")
    finish_plot("04_average_grade_by_study_time.png")

# ------------------------------------------------------------
# GRAPH 5 — Previous grade G2 vs final grade G3
# ------------------------------------------------------------
if "G2" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.regplot(
        data=df, x="G2", y="G3",
        scatter_kws={"alpha": 0.6, "s": 40},
        line_kws={"color": "darkred", "linewidth": 2},
        ci=None
    )
    correlation = df["G2"].corr(df["G3"])
    plt.title(f"Previous Grade (G2) vs Final Grade (G3) | r = {correlation:.2f}")
    plt.xlabel("Second Period Grade (G2)")
    plt.ylabel("Final Grade (G3)")
    finish_plot("05_previous_vs_final_grade.png")

# ------------------------------------------------------------
# GRAPH 6 — Final grade by parental education
# ------------------------------------------------------------
if "Medu" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Medu", y="G3", color="lightsteelblue")
    education_means = df.groupby("Medu")["G3"].mean()
    plt.scatter(
        education_means.index, education_means.values,
        color="darkred", marker="D", s=45,
        label="Average grade", zorder=3
    )
    plt.legend()
    plt.title("Final Grade by Mother's Education Level")
    plt.xlabel("Mother's Education Level (0 = none, 4 = higher education)")
    plt.ylabel("Final Grade (G3)")
    finish_plot("06_grade_by_mothers_education.png")

# ------------------------------------------------------------
# GRAPH 7 — Correlation heatmap
# ------------------------------------------------------------
selected_columns = [
    col for col in
    ["age", "Medu", "Fedu", "studytime", "failures",
     "absences", "G1", "G2", "G3"]
    if col in df.columns
]
corr_matrix = df[selected_columns].corr(numeric_only=True)

plt.figure(figsize=(10, 7))
sns.heatmap(
    corr_matrix, annot=True, cmap="coolwarm",
    fmt=".2f", linewidths=0.5, center=0
)
plt.title("Correlation Heatmap of Academic and Student Features")
finish_plot("07_correlation_heatmap.png")

# ------------------------------------------------------------
# 4. MACHINE LEARNING — LEAKAGE-SAFE FEATURES
# ------------------------------------------------------------
target = "G3"

# Exclude G3 and Total_Grades to prevent target leakage.
X = df.drop(columns=[target, "Total_Grades"], errors="ignore")
y = df[target]

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nMACHINE LEARNING DATA SPLIT")
print("-" * 50)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Input features:", X.shape[1])

# ------------------------------------------------------------
# 5. LINEAR REGRESSION
# ------------------------------------------------------------
linear_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])
linear_pipeline.fit(X_train, y_train)

linear_metrics, linear_predictions = evaluate_model(
    "Linear Regression", linear_pipeline, X_test, y_test
)

print("\nLINEAR REGRESSION RESULTS")
print("-" * 50)
print(f'MAE:  {linear_metrics["MAE"]:.3f}')
print(f'RMSE: {linear_metrics["RMSE"]:.3f}')
print(f'R2:   {linear_metrics["R2 Score"]:.3f}')

print("\nFIRST 10 ACTUAL VS PREDICTED GRADES")
for actual, predicted in zip(y_test.iloc[:10], linear_predictions[:10]):
    print(f"Actual: {actual:>2} | Predicted: {predicted:.2f}")

# ------------------------------------------------------------
# GRAPH 8 — Actual vs predicted
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    y_test, linear_predictions,
    alpha=0.68, color="steelblue",
    edgecolor="white", s=55, label="Predicted Grades"
)
plt.plot(
    [0, 20], [0, 20],
    color="darkred", linestyle="--",
    linewidth=2, label="Perfect Prediction"
)
plt.title("Actual vs Predicted Student Grades")
plt.xlabel("Actual Final Grades")
plt.ylabel("Predicted Final Grades")
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.legend()
finish_plot("08_actual_vs_predicted.png")

# ------------------------------------------------------------
# 6. CROSS-VALIDATION
# ------------------------------------------------------------
cv_scores = cross_val_score(
    linear_pipeline, X, y, cv=5, scoring="r2"
)
print("\nLINEAR REGRESSION CROSS-VALIDATION")
print("-" * 50)
print("Individual R2 scores:", np.round(cv_scores, 3))
print("Average R2:", round(cv_scores.mean(), 3))
print("Standard deviation:", round(cv_scores.std(), 3))

# ------------------------------------------------------------
# 7. RIDGE REGRESSION + HYPERPARAMETER TUNING
# ------------------------------------------------------------
ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge()),
])
ridge_pipeline.fit(X_train, y_train)

ridge_metrics, _ = evaluate_model(
    "Ridge Regression (alpha=1)", ridge_pipeline, X_test, y_test
)

ridge_search_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge()),
])

parameter_grid = {
    "model__alpha": [0.1, 1, 10, 100]
}
grid_search = GridSearchCV(
    ridge_search_pipeline,
    parameter_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_ridge = grid_search.best_estimator_
tuned_metrics, _ = evaluate_model(
    "Tuned Ridge Regression", best_ridge, X_test, y_test
)

print("\nRIDGE REGRESSION RESULTS")
print("-" * 50)
print(f'Baseline Ridge MAE:  {ridge_metrics["MAE"]:.3f}')
print(f'Baseline Ridge RMSE: {ridge_metrics["RMSE"]:.3f}')
print(f'Baseline Ridge R2:   {ridge_metrics["R2 Score"]:.3f}')
print("\nRIDGE HYPERPARAMETER TUNING")
print("-" * 50)
print("Best alpha:", grid_search.best_params_["model__alpha"])
print("Best CV R2:", round(grid_search.best_score_, 3))
print("\nTUNED RIDGE RESULTS")
print("-" * 50)
print(f'MAE:  {tuned_metrics["MAE"]:.3f}')
print(f'RMSE: {tuned_metrics["RMSE"]:.3f}')
print(f'R2:   {tuned_metrics["R2 Score"]:.3f}')

# ------------------------------------------------------------
# GRAPH 9 — Model error comparison
# ------------------------------------------------------------
comparison = pd.DataFrame([
    linear_metrics,
    ridge_metrics,
    tuned_metrics,
])

print("\nFINAL MODEL COMPARISON")
print("-" * 60)
print(comparison.round(3).to_string(index=False))

ax = comparison.set_index("Model")[["MAE", "RMSE"]].plot(
    kind="bar", figsize=(10, 6),
    edgecolor="white", width=0.7
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
ax.set_title("Model Error Comparison (Lower is Better)")
ax.set_ylabel("Error (grade points)")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=12)
plt.tight_layout()
finish_plot("09_model_error_comparison.png")

# ------------------------------------------------------------
# 8. SHOW ALL GRAPHS TOGETHER
# ------------------------------------------------------------
print("\nGRAPH OUTPUT")
print("-" * 50)
print("Number of graphs created:", len(figures))
print("Saved graph folder:", OUTPUT_DIR)
print("Each graph is also saved as a PNG file in that folder.")

plt.show()
