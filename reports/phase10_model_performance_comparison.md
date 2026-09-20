# Phase 10.2 — Model Performance Comparison

Project: Data Science Framework for Early Detection of Mental Health Risk

Framework Version: 0.1.0

## Standardized Comparison

| dataset | modality | model | evaluation | accuracy | precision | recall | f1 | roc_auc | pr_auc | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dreaddit | Text | TF-IDF + Logistic Regression | Final holdout | 0.7175 | 0.7103 | 0.7642 | 0.7363 | 0.8207 | 0.8412 |  |  |  |
| Dreaddit | Text + engineered | TF-IDF + engineered features + Logistic Regression | Validation | 0.7766 | 0.7771 | 0.8026 | 0.7896 | 0.8503 | 0.8478 |  |  |  |
| EATD | Text | Character TF-IDF + Logistic Regression | Official validation | 0.7848 | 0.0000 | 0.0000 | 0.0000 | 0.5094 | 0.2105 |  |  |  |
| EATD | Text + Voice | OOF Text + Voice Logistic Regression Fusion | Official validation | 0.7468 | 0.2222 | 0.1333 | 0.1667 | 0.5167 | 0.2152 |  |  |  |
| EATD | Voice | Logistic Regression | Official validation | 0.7722 | 0.2857 | 0.1333 | 0.1818 | 0.5531 | 0.2388 |  |  |  |
| StudentLife | Behavior | Random Forest Regression | 5-fold participant-level CV |  |  |  |  |  |  | 3.8488 | 4.7222 | -0.1510 |

## Interpretation Notes

- Classification models are compared using multiple metrics rather than accuracy alone.
- ROC-AUC and PR-AUC are retained because class imbalance can make accuracy misleading.
- StudentLife is a regression experiment and therefore uses MAE, RMSE and R².
- Results come from different datasets and evaluation protocols and must not be interpreted as a single head-to-head benchmark.
- The comparison is descriptive and does not establish clinical validity.

## Phase 10.2 Status

COMPLETE