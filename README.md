# Predicting User Churn for Waze with Random Forest and XGBoost

## Overview

The goal of this project was to predict which Waze users will churn (stop using the app) in a given month, and to identify the behaviors most associated with churn. I built, tuned and compared two tree-based classifiers, a random forest and XGBoost, on a dataset of 14,999 users (14,299 after removing rows with no label). The final XGBoost model reached a **recall of 18%, precision of 42% and F1 of 0.25** on held-out test data. It finds about 1 in 5 churned users, so it is best used as a decision-support tool to prioritize retention efforts rather than as a standalone decision-maker.

*This is a course project from the Google Advanced Data Analytics Certificate. The dataset is synthetic and was created for the course in partnership with Waze.*

**The repository has two parts, using the same Waze dataset:**

1. **Part 1, regression course:** binomial logistic regression (data cleaning, outliers, feature engineering, multicollinearity). Reported accuracy: 82.6%. Accuracy alone is a weak measure here because about 82% of users are retained, so a model that always predicts "retained" scores almost the same.
2. **Part 2, machine learning course (main part of this README):** random forest and XGBoost, evaluated with recall, precision and F1 instead of accuracy.

## Business Understanding

Waze leadership wants to reduce user churn, improve retention and grow the business. The stakeholders are the finance and administration team, operations, and the data analytics team. Two errors matter:

- **False negative** (the model says a user will stay, but the user churns): Waze loses the user without any attempt to retain them. This is the costlier error.
- **False positive** (the model says a user will churn, but the user stays): a retention offer is sent to someone who would have stayed. This cost is small.

Because churned users are a minority, I evaluated models with **recall, precision and F1** instead of accuracy.

## Data Understanding

- **Source:** `waze_dataset.csv`, 14,999 rows (one per user) and 12 columns, including usage (sessions, drives, kilometers, driving time, active days), days since onboarding, navigations to favorite places and device type.
- **Target:** `label` (retained / churned). 700 rows had no label and were removed, leaving 14,299 users: **82.3% retained and 17.7% churned**.
- **Data issues found:** many variables are heavily skewed with extreme values; some ratios produced infinite values when dividing by zero (replaced with 0); the estimated `total_sessions` made one ratio exceed 100% for some users. Tree-based models are robust to these, so the values were kept.
- **Feature engineering:** I created 7 features: `km_per_driving_day`, `percent_sessions_in_last_month`, `professional_driver`, `total_sessions_per_day`, `km_per_hour`, `km_per_drive` and `percent_of_sessions_to_favorite`. `device` was encoded as `device2` (Android = 0, iPhone = 1).

## Modeling and Evaluation

Data was split 60% train / 20% validation / 20% test (stratified). Models were tuned with `GridSearchCV` (4-fold cross-validation, scored on recall), compared on the validation set, and only the champion was evaluated once on the test set.

| Model | Dataset | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Random forest | Cross-validation | 0.457 | 0.127 | 0.198 | 0.819 |
| XGBoost | Cross-validation | 0.426 | 0.171 | 0.244 | 0.812 |
| Random forest | Validation | 0.445 | 0.120 | 0.189 | 0.817 |
| XGBoost | Validation | 0.423 | 0.162 | 0.234 | 0.812 |
| **XGBoost (champion)** | **Test** | **0.424** | **0.181** | **0.254** | **0.811** |

Scores were stable across cross-validation, validation and test, so there is no sign of overfitting. On the test set (2,860 users) the champion found 92 of the 507 churned users and flagged 125 users who stayed.

**Most used features (XGBoost):** `km_per_hour`, `n_days_after_onboarding`, `percent_sessions_in_last_month`, `total_navigations_fav1` and `percent_of_sessions_to_favorite`. Six of the top ten features were engineered by me, while `professional_driver` and `device2` added almost nothing. Feature importance shows which variables the model uses, not the direction or cause of the effect.

## Conclusion

The data contains some signal about churn, but not enough for strong predictions. Recommendations:

1. Use the model to shortlist at-risk users and test retention offers with an A/B test before any wider rollout.
2. Collect richer data: monthly usage over time instead of a single snapshot, app performance, user feedback and reasons for leaving.
3. Improve the model with a wider hyperparameter search, more engineered features and a lower decision threshold to raise recall, and monitor it every month.

**Next questions:** when after sign-up do users churn, and which user segments churn most?

## Repository Contents

```
waze-churn-prediction/
├── README.md
├── notebook/
│   ├── 01_logistic_regression.ipynb       # part 1: binomial logistic regression
│   └── 02_random_forest_xgboost.ipynb     # part 2: tree-based models, evaluation
├── data/
│   └── waze_dataset.csv                   # synthetic dataset created for the course
└── docs/
    ├── PACE_strategy_document.docx        # planning and reflection (Plan, Analyze, Construct, Execute)
    └── executive_summary.pptx             # one-page summary for stakeholders
```

To reproduce the results, install the tools below, open a notebook in Jupyter and run all cells (the notebooks read `data/waze_dataset.csv`; adjust the file path if needed).

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, XGBoost, Jupyter

## Author

Adama Sidy Sane, aspiring data analyst, Google Advanced Data Analytics Professional Certificate (2026).
