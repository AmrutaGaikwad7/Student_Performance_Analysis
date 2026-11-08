# Student Performance Analysis and Prediction System

A Python-based Machine Learning project that analyzes students' academic performance, explores factors associated with final grades, and predicts final examination marks using Linear Regression and Ridge Regression.

---

## 1. Project Overview and Objectives

This project uses the UCI Student Performance dataset (Mathematics course) to analyze students' academic performance and predict their final grades (`G3`).

### Objectives

- Load, inspect, and clean the student performance dataset.
- Perform Exploratory Data Analysis (EDA) using descriptive statistics and visualizations.
- Analyze relationships between study time, absences, previous grades (`G1`, `G2`), and final grades (`G3`).
- Build and evaluate a Linear Regression model.
- Apply cross-validation and hyperparameter tuning using Ridge Regression.
- Prevent data leakage to ensure reliable model evaluation.

## 2. Dataset Description and Source

| Item | Details |
|---|---|
| Dataset | Student Performance Dataset (Mathematics: `student-mat.csv`) |
| Source | UCI Machine Learning Repository |
| Authors | P. Cortez and A. Silva (2008) |
| Records | 395 students |
| Original attributes | 33 |
| Target variable | `G3` – final grade (0–20) |
| File format | CSV, semicolon-separated |

**Important features:**

- `studytime`: Weekly study time category.
- `absences`: Number of school absences.
- `failures`: Number of past class failures.
- `G1`: First-period grade.
- `G2`: Second-period grade.
- `Medu` and `Fedu`: Mother's and father's education levels.

**Dataset note:** The dataset records the number of absences rather than an attendance percentage. Therefore, `absences` is used as an attendance-related feature.

**Dataset citation:**

Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.* Proceedings of the 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008).

Dataset link: https://archive.ics.uci.edu/dataset/320/student+performance

## 3. Technologies and Libraries Used

- Python 3.11
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 4. Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/AmrutaGaikwad7/Student_Performance_Analysis.git
cd Student_Performance_Analysis
```

### Step 2: Create a virtual environment (optional)

```bash
python -m venv .venv
```

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## 5. How to Run

1. Ensure the dataset is available at `data/student-mat.csv`.
2. Open the terminal in the project root directory.
3. Run the following command:

```bash
python src/inspect_data.py
```

The script performs data analysis, preprocessing, model training, evaluation, and visualization.

It displays the generated graphs and prints the model evaluation results in the terminal. The nine graphs are also saved as PNG files in `outputs/visualizations/`.

## 6. Methodology

### Data Cleaning and Feature Engineering

- Inspected the dataset for missing values and duplicate records.
- Checked and handled duplicate records where applicable.
- Used preprocessing pipelines with appropriate imputation techniques.
- Created a `Total_Grades` feature for descriptive analysis only.
- Excluded `G3` and `Total_Grades` from model inputs to prevent target leakage.

### Data Preprocessing

A Scikit-learn preprocessing pipeline is used.

- **Numerical features:** Median imputation and StandardScaler.
- **Categorical features:** Most-frequent imputation and OneHotEncoder.

Preprocessing is fitted on training data to prevent data leakage.

### Machine Learning Models

| Model | Purpose |
|---|---|
| Linear Regression | Primary prediction model |
| Ridge Regression (alpha = 1) | Regularized baseline model |
| Tuned Ridge Regression | Hyperparameter-optimized model |

### Training and Testing

- Training data: 80% (316 records)
- Testing data: 20% (79 records)
- Random state: 42
- Input features: 32
- Cross-validation: 5-fold
- Hyperparameter tuning: GridSearchCV

Ridge Regression hyperparameter tuning is performed using alpha values `[0.1, 1, 10, 100]`.

### Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors.
- **R² Score:** Measures how much variation in final grades is explained by the model.

## 7. Visualizations

The project generates nine visualizations:

| No. | Visualization | Chart Type |
|---|---|---|
| 1 | Distribution of Final Grades | Histogram |
| 2 | Study Time vs Final Grade | Strip Plot |
| 3 | Absences vs Final Grade | Scatter Plot with Trend Line |
| 4 | Average Final Grade by Study Time | Bar Chart |
| 5 | Previous Grade (G2) vs Final Grade (G3) | Scatter Plot with Trend Line |
| 6 | Final Grade by Mother's Education | Box Plot |
| 7 | Correlation Heatmap | Heatmap |
| 8 | Actual vs Predicted Grades | Scatter Plot |
| 9 | Model Error Comparison | Bar Chart |

All nine visualizations are generated and saved as PNG files in the `outputs/visualizations/` folder when the Python script runs.

## 8. Results and Model Evaluation

The models were evaluated using MAE, RMSE, and R² score.

### Model Performance Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.647 | 2.378 | 0.724 |
| Ridge Regression (alpha = 1) | 1.639 | 2.372 | 0.726 |
| Tuned Ridge Regression | **1.589** | **2.328** | **0.736** |

### Cross-Validation Results

Linear Regression was evaluated using 5-fold cross-validation.

- Individual R² scores: `[0.801, 0.883, 0.768, 0.774, 0.696]`
- Average R²: **0.784**
- Standard deviation: **0.061**

### Ridge Hyperparameter Tuning

- Best alpha: **10**
- Best cross-validation R²: **0.83**

### Key Findings

- The dataset contains 395 student records and 33 original attributes.
- No missing values or duplicate records were found.
- The average final grade was 10.42, with a median of 11.0.
- The standard deviation of final grades was 4.58.
- The average weekly study-time category was approximately 2.04.
- Tuned Ridge Regression achieved the best performance among the three evaluated models.
- The tuned model achieved an R² score of 0.736 and an MAE of 1.589.
- Hyperparameter tuning improved Ridge Regression performance compared with the baseline model.

## 9. Project Structure

```text
Student_Performance_Analysis/
│
├── data/
│   └── student-mat.csv
│
├── src/
│   └── inspect_data.py
│
├── outputs/
│   └── visualizations/
│       ├── (nine generated PNG files)
│
├── screenshots/
│
├── Project_Report.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

## 10. Limitations

- The dataset contains 395 student records from two schools and focuses on one subject.
- An attendance percentage is not available; the number of absences is used instead.
- Previous grades (`G1` and `G2`) are strong predictors of final grades, making the model more suitable for predicting final performance than very early intervention.
- Some students have a final grade of zero, which may affect prediction performance.
- Relationships identified through the analysis do not necessarily establish causation.

## 11. Future Improvements

- Implement an optional pass/fail classification model.
- Compare additional algorithms such as Random Forest and Gradient Boosting.
- Develop a prediction model that does not rely on previous grades.
- Identify students who may need additional academic support.
- Explore additional datasets to improve model generalization.

## Author

**Amruta Dipak Gaikwad**

B.Tech – Artificial Intelligence and Machine Learning

September 2026 Batch – AI/ML with Python Training Program (InternsElite)
