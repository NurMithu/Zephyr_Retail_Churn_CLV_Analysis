# README update — copy these sections into your repo's README.md

Replace the **Repository Structure**, **Key Results** and **Business Recommendations** sections with the text below, and fix the `git clone` URL (replace `<your-username>/zephyr-retail-churn-clv-analysis` with `NurMithu/Zephyr_Retail_Churn_CLV_Analysis`).

---

## Repository Structure

```
Zephyr_Retail_Churn_CLV_Analysis/
├── notebooks/
│   ├── Zephyr_Retail_Churn_CLV_Analysis.ipynb    # v1: 15-module baseline analysis
│   └── Zephyr_Churn_CLV_Advanced_v2.ipynb        # v2: model comparison, cost-benefit decisions, SHAP, forward-looking CLV
├── data/
│   └── zephyr_customer_data.csv                  # Raw dataset (5,015 rows, 15 duplicates)
├── requirements.txt
├── LICENSE
└── README.md
```

Add to `requirements.txt`: `xgboost`, `lightgbm`, `shap`

## What v2 adds (Modules 16–23)

| Module | Question answered |
|---|---|
| 17 | Validity audit: do the v1 claims survive scrutiny? |
| 18 | Logistic Regression vs XGBoost vs LightGBM (5-fold CV repeated 3×, mean ± std) |
| 19 | Calibration: are the predicted probabilities trustworthy? |
| 20 | Cost-benefit decision rule: whom should Zephyr actually contact? |
| 21 | SHAP: why was a customer flagged? |
| 22 | Forward-looking, risk-adjusted 12-month CLV and a risk × value priority matrix |
| 23 | Summary, corrections, limitations |

## Key Results (v2, base churn rate 4.4%)

**Model comparison** (5-fold stratified CV × 3 repeats)

| Model | ROC AUC | PR AUC (random = 0.044) |
|---|---|---|
| Logistic Regression (class-balanced) | 0.803 ± 0.050 | 0.270 ± 0.059 |
| XGBoost | 0.806 ± 0.046 | 0.236 ± 0.049 |
| LightGBM | 0.809 ± 0.045 | 0.265 ± 0.054 |

Boosted trees do **not** meaningfully beat Logistic Regression on this data: differences are within fold-to-fold variation. Logistic Regression is retained as the production candidate because it is simpler, auditable and well calibrated.

**Decision value** (assumptions: $15 per retention offer, 30% offer success rate, 30% gross margin, 12-month horizon — all editable in Module 20)

| Policy | Customers contacted | Expected net profit | Realized net profit (actual churn labels) |
|---|---|---|---|
| Contact everyone | 5,000 | −$51,947 | −$52,966 |
| Contact if p ≥ 0.10 | 500 | $3,777 | $3,571 |
| Expected-value rule | 310 | $7,198 | $6,939 |

The expected-value rule reaches 28% of churners with 20% precision, and the realized profit (computed from true labels) agrees closely with the expected profit, supporting the calibration of the probabilities. These dollar figures depend on the stated assumptions; the offer success rate must be measured with an A/B test.

**What drives churn**

- **Customer service calls:** churn ≈ 1% with 0 calls, ≈ 17% with 4 calls, ≈ 39% with 5+ calls
- **Rating:** churn ≈ 16% at rating ≤ 2, ≈ 3% at rating > 4
- Tenure, total spend, region and payment method add almost nothing. Income and Age show small effects that need confirmation on more data.

## Corrections to v1 (kept transparent on purpose)

- v1 stated tenure is the strongest protective factor against churn. In this dataset churn is roughly flat across tenure buckets (≈ 4–5%), so that recommendation is **not supported**.
- v1 reported CLV R² = 0.455 with `Total_Spend` as the target. Removing `Months_Subscribed` drops CV R² to 0.06: the model was mostly recovering *spend rate × months*, not forecasting value. v2 replaces it with a risk-adjusted 12-month CLV.

## Business Recommendations (updated)

- Route customers to retention by **expected net gain per contact**, not by a fixed score cutoff.
- Treat **repeated support contacts** (3+) and **low ratings** (≤ 3) as the leading churn signals; trigger a proactive service check-in.
- Run an **A/B test** of the retention offer to measure true uplift before scaling.
- Prioritize the **high-risk × high-value** segment (≈ 5% of customers, ≈ 17% actual churn vs 4.4% overall).

## Limitations

- ~5,000 rows and 218 churners: wide confidence intervals; results will fluctuate across samples.
- Dataset appears synthetic; the churn horizon is undocumented (v2 assumes 12 months).
- Random CV does not test temporal drift; use time-based validation in production.
- Correlation is not causation: service calls predict churn, but reducing calls may not reduce churn.
