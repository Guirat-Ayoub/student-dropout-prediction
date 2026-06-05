[README (1).md](https://github.com/user-attachments/files/28643742/README.1.md)
# Student Dropout Prediction using Machine Learning

---

## Project Overview

This project predicts whether a university student is likely to drop out or successfully graduate based on academic, demographic, and financial information. Early identification of at-risk students allows educational institutions to provide timely support and improve student retention rates.

---

## Objective

Develop a machine learning pipeline capable of:

- Predicting student dropout risk with high accuracy
- Identifying the most important factors influencing dropout
- Generating individual risk scores with tiered intervention recommendations
- Supporting data-driven retention strategies for academic advisors

---

## Dataset

The dataset contains student records across four feature categories:

**Demographic Features**
- Age at enrollment, gender, nationality, marital status

**Academic Features**
- Curricular units approved and enrolled, academic grades, performance trends

**Financial Features**
- Tuition fees payment status, scholarship holder status, financial stress indicators

**Economic Indicators**
- GDP, inflation rate, unemployment rate

**Target Variable**

| Class | Description |
|---|---|
| 0 | Graduate |
| 1 | Dropout |

The Enrolled class was removed to formulate a binary classification problem focused on dropout detection.

---

## Feature Engineering

Six engineered features were created to capture information not present in the raw columns:

| Feature | Formula | Rationale |
|---|---|---|
| `approval_rate_1st` | approved_1st / enrolled_1st | Academic success rate in semester 1 |
| `approval_rate_2nd` | approved_2nd / enrolled_2nd | Academic success rate in semester 2 |
| `grade_ratio` | grade_1st / (grade_2nd + 1) | Relative academic achievement across semesters |
| `performance_trend` | approval_rate_2nd - approval_rate_1st | Captures improvement or decline between semesters |
| `financial_stress` | tuition_overdue AND no scholarship | Combined financial vulnerability indicator |
| `economic_pressure` | unemployment_rate × inflation_rate | Macroeconomic burden at time of enrollment |

---

## Data Preprocessing

**Numerical Features**
- Missing value imputation using median strategy
- Standardisation using StandardScaler

**Categorical Features**
- Missing value imputation using most frequent value
- Encoding using OneHotEncoder

All preprocessing was implemented inside a scikit-learn Pipeline with ColumnTransformer to prevent data leakage.

---

## Models Evaluated

The following algorithms were compared using Stratified 5-Fold Cross Validation with ROC-AUC as the primary metric:

- Logistic Regression
- Gradient Boosting
- Random Forest
- XGBoost

ROC-AUC was chosen over accuracy due to class imbalance in the dataset.

---

## Results

**Cross-Validation ROC-AUC**

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.9125 |
| Gradient Boosting | 0.9148 |
| Random Forest | 0.9073 |
| XGBoost | 0.9145 |

**Final Test Performance**

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.9263 |
| XGBoost | 0.9382 |

XGBoost achieved the best overall performance and was selected as the final model.

---

## Confusion Matrix

**Logistic Regression**

|  | Predicted Graduate | Predicted Dropout |
|---|---|---|
| Actual Graduate | 567 | 34 |
| Actual Dropout | 72 | 212 |

**XGBoost**

|  | Predicted Graduate | Predicted Dropout |
|---|---|---|
| Actual Graduate | 563 | 38 |
| Actual Dropout | 60 | 224 |

XGBoost demonstrated superior detection of students at risk of dropping out, catching 224 out of 284 dropout cases.

---

## Model Explainability

SHAP (SHapley Additive Explanations) was used to interpret model predictions at both global and individual student level.

**Most Influential Features**

1. Approval rate — 2nd semester
2. Tuition fees up to date
3. Financial stress
4. Course
5. Approval rate — 1st semester

This analysis provides actionable insight into why the model flags a specific student as high risk, making the system interpretable for academic advisors.

---

## Risk Scoring System

Each student is assigned a dropout risk tier based on predicted probability:

| Probability | Risk Tier | Recommended Action |
|---|---|---|
| Below 40% | Low Risk | Standard monitoring |
| 40% to 70% | Medium Risk | Schedule advisor check-in within 2 weeks |
| Above 70% | High Risk | Immediate intervention recommended |

**Example output for a single student:**

```
Dropout Probability : 87%
Risk Tier           : High Risk
Key Factors         : Low approval rate, tuition fees overdue, financial stress
Recommendation      : Immediate advisor intervention recommended
```

---

## Technologies Used

- Python 3.12
- pandas, NumPy
- scikit-learn (Pipeline, ColumnTransformer, StratifiedKFold)
- XGBoost
- Matplotlib, Seaborn
- SHAP

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/Guirat-Ayoub/student-dropout-prediction.git
cd student-dropout-prediction
pip install -r requirements.txt
```

Open the notebook in Google Colab:

https://colab.research.google.com/github/Guirat-Ayoub/student-dropout-prediction/blob/main/notebook.ipynb

The full dataset is available on Kaggle:
https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention

---

## Future Improvements

- Hyperparameter optimisation using Optuna
- Stacking ensemble combining XGBoost and Logistic Regression
- Interactive Streamlit dashboard for academic advisors
- Real-time risk monitoring with data drift detection
- Personalised student intervention recommendation system
