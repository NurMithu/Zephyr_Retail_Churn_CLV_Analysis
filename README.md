# Zephyr Retail Solutions — Customer Churn Prediction & Customer Lifetime Value Analysis

End-to-end data science project analyzing ~5,000 customer records to predict churn, estimate customer lifetime value (CLV), and translate the findings into prioritized business recommendations.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.0+-150458?logo=pandas&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## Overview

Zephyr Retail Solutions is a subscription-based retail business facing volatile monthly churn and inconsistent customer spending. This project builds a complete, reproducible analytics pipeline — from raw data quality assessment through statistical testing to two production-ready machine learning models — that answers six core business questions:

1. Which customers are most likely to churn?
2. Which factors influence churn?
3. Which customers generate the highest lifetime value?
4. Which variables significantly affect customer spending?
5. How can Zephyr reduce churn?
6. How can Zephyr increase customer lifetime value?

The deliverable is a single, fully-executable Jupyter notebook structured as 15 modules, written to consulting/portfolio standard — every visualization, statistical test, and model output is paired with a plain-English business interpretation.

## Repository Structure

```
zephyr-retail-churn-clv-analysis/
├── notebooks/
│   └── Zephyr_Retail_Churn_CLV_Analysis.ipynb   # Full analysis notebook (run top to bottom)
├── data/
│   └── zephyr_customer_data.csv                 # Raw dataset (~5,000 customer records)
├── requirements.txt                             # Python dependencies
├── LICENSE
└── README.md
```

## Dataset

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Age` | Customer age (years) |
| `Gender` | Customer gender |
| `Region` | North / South / East / West |
| `Income` | Annual income |
| `Months_Subscribed` | Subscription tenure in months |
| `Total_Spend` | Total spend to date (used as the CLV proxy) |
| `Customer_Service_Calls` | Number of support contacts |
| `Rating_Score` | Customer satisfaction rating (0–5) |
| `Payment_Method` | PayPal / Credit Card / Bank Transfer |
| `Churn` | 1 = churned, 0 = retained (target variable) |

The raw file contains realistic data quality issues — duplicate records, inconsistent category labels, invalid negative income values, and missing satisfaction ratings — all of which are detected and resolved transparently in the notebook (Modules 3–4) rather than assumed away.

## Notebook Structure (15 Modules)

| Module | Contents |
|---|---|
| 1 | Business Understanding — problem, objectives, scope, expected outcomes |
| 2 | Data Understanding — load, inspect, describe |
| 3 | Data Quality Assessment — missingness, duplicates, invalid values, outliers |
| 4 | Data Preprocessing — cleaning, standardization, imputation, validation |
| 5 | Exploratory Data Analysis — distributions, correlations, skewness/kurtosis |
| 6 | Bivariate & Multivariate Analysis — every predictor vs. Churn |
| 7 | Statistical Analysis — Shapiro-Wilk, Levene, t-test, Mann-Whitney U, ANOVA, Chi-square, Cramer's V |
| 8 | Feature Engineering — tenure buckets, value segments, risk flags |
| 9 | Feature Selection — correlation, VIF, mutual information, RFE, Random Forest importance |
| 10 | Data Transformation — train/test split, encoding, scaling pipeline |
| 11 | Churn Prediction — Logistic Regression (class-balanced), ROC/AUC, odds ratios |
| 12 | CLV Prediction — Multiple Linear Regression, residual diagnostics |
| 13 | Model Evaluation — learning curves, misclassification analysis, limitations |
| 14 | Business Insights Dashboard — executive summary visuals |
| 15 | Business Recommendations — retention, revenue, marketing, service, pricing |

## Key Results

**Churn Model — Logistic Regression (class-balanced)**

| Metric | Value |
|---|---|
| ROC AUC | 0.789 |
| Recall | 0.61 |
| Precision | 0.11 |
| Accuracy | 0.76 |

> Churn occurs in only ~4–5% of customers in this dataset. Because of this imbalance, the model is trained with `class_weight='balanced'` and is deliberately tuned to prioritize **Recall** over raw Accuracy — for an early-warning system, missing a true churner is far costlier than an unnecessary retention offer. The notebook documents this trade-off explicitly and shows how the decision threshold can be adjusted if Zephyr's retention outreach cost structure changes.

**CLV Model — Multiple Linear Regression**

| Metric | Value |
|---|---|
| R² | 0.455 |
| RMSE | 483.4 |
| MAE | 384.0 |

**Top drivers identified across correlation, mutual information, RFE, and Random Forest importance:** `Customer_Service_Calls`, `Rating_Score`, `Months_Subscribed`, and `Total_Spend` consistently emerge as the strongest predictors of both churn and spend.

## How to Run

### Option 1 — Google Colab (recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `notebooks/Zephyr_Retail_Churn_CLV_Analysis.ipynb`.
3. Upload `data/zephyr_customer_data.csv` to the Colab file browser (or mount Google Drive and update `DATA_PATH` in Module 2).
4. Run all cells (`Runtime > Run all`).

### Option 2 — Local Jupyter
```bash
git clone https://github.com/<your-username>/zephyr-retail-churn-clv-analysis.git
cd zephyr-retail-churn-clv-analysis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/Zephyr_Retail_Churn_CLV_Analysis.ipynb
```
Run all cells top to bottom — no manual edits are required beyond confirming the dataset path in Module 2 if you relocate the CSV.

## Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scipy` · `statsmodels` · `scikit-learn` · `Jupyter`

## Business Recommendations (Summary)

- **Deploy the churn model as a monthly scoring system**, routing high-probability customers to a proactive customer-success queue.
- **Treat customer service friction as a leading indicator**: 2+ service calls in a short window should auto-trigger a satisfaction check-in.
- **Invest in early-tenure retention** (first 6–12 months), since tenure is the strongest protective factor against churn and the strongest positive driver of spend.
- **Prioritize retention spend using a churn-probability × customer-value matrix** rather than blanket discounting.

Full detail, including marketing, pricing, and future-work recommendations, is in Module 15 of the notebook.

## Limitations & Future Work

- Logistic/Linear Regression assume linear relationships; non-linear model classes (gradient boosting, random forests) are natural next candidates and are noted as future work in the notebook.
- The dataset does not include marketing-exposure, product-usage, or macroeconomic variables that likely also influence churn and spend.
- This analysis is observational; a controlled A/B test is recommended before rolling out retention interventions at scale.
- Models should be retrained periodically to account for concept drift in customer behavior.

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center"><i>Built as an end-to-end analytics case study — from raw, messy customer data to validated, business-ready predictive models.</i></p>
