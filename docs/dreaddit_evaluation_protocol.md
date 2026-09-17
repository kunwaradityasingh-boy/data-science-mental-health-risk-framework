# Dreaddit Evaluation Protocol

## Purpose

This document defines the evaluation strategy used for Dreaddit-based
experiments in the Data Science Framework for Early Detection of Mental
Health Risk.

The objective is to prevent inappropriate data leakage and make model
comparisons reproducible.

---

## 1. Dataset Structure

The acquired Dreaddit dataset contains:

- `dreaddit-train.csv`
- `dreaddit-test.csv`

The inspected training split contains:

- 2,838 segments
- 2,343 unique posts
- 380 posts with more than one segment
- 1,963 posts with exactly one segment
- Maximum of 6 segments from a single post

The inspected test split contains:

- 715 segments
- 586 unique posts
- 92 posts with more than one segment
- 494 posts with exactly one segment
- Maximum of 6 segments from a single post

---

## 2. Original Train/Test Split

The supplied Dreaddit train/test split will be preserved unchanged.

The audit found:

- 0 overlapping `id` values between train and test
- 0 overlapping `post_id` values between train and test
- 3 exact text records appearing in both train and test

Therefore, the supplied split will not be described as completely
leakage-free.

The three exact text overlaps will be documented as an evaluation
limitation.

---

## 3. Grouping Unit

The primary grouping unit for newly generated evaluation partitions
will be:

`post_id`

This is necessary because a single Reddit post may generate multiple
sentence-level segments.

All segments belonging to the same `post_id` must remain within the
same partition.

---

## 4. Research Validation Split

The supplied training split will be divided into training and validation
sets using group-aware splitting based on `post_id`.

The exact validation proportion and random seed will be recorded in the
experiment configuration.

No row-level random split should be used when constructing the research
validation set.

---

## 5. Test Set

The supplied Dreaddit test set will remain untouched during model
development.

It will only be used for final evaluation after model and preprocessing
decisions have been finalized.

No hyperparameter tuning should be performed using the test set.

---

## 6. Evaluation Experiments

The framework will evaluate multiple feature configurations.

### Experiment A — Text-only

Input:

- Raw Reddit text

Purpose:

Establish a baseline for NLP-based stress classification.

---

### Experiment B — Engineered Features

Input:

- Linguistic features
- Readability features
- Social features
- Sentiment-related features

Purpose:

Evaluate the contribution of structured features independently from
raw text.

---

### Experiment C — Combined Text + Engineered Features

Input:

- Raw text representation
- Selected engineered features

Purpose:

Evaluate whether combining textual representation with structured
features provides additional predictive information.

---

## 7. Primary Metrics

Because the dataset contains two classes, evaluation will include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

The confusion matrix will also be reported.

Class-wise precision, recall and F1-score will be reported to avoid
relying only on overall accuracy.

---

## 8. Reproducibility

Every experiment must record:

- Dataset version/source
- Dataset checksum where available
- Random seed
- Grouping strategy
- Train/validation/test definition
- Feature configuration
- Preprocessing configuration
- Model configuration
- Hyperparameters
- Evaluation metrics
- Model version

---

## 9. Leakage Controls

The following controls are required:

1. No `post_id` may appear in both research training and validation sets.
2. Test data must remain isolated during model development.
3. Text preprocessing must be fitted only on training data where applicable.
4. Feature scaling must be fitted only on training data.
5. Imputation parameters must be fitted only on training data.
6. Hyperparameter tuning must not use the test set.
7. Duplicate and near-duplicate records should be investigated.
8. Exact train/test text overlap must be reported.
9. Any additional grouping identifier discovered during analysis must
   be evaluated for leakage risk.

---

## 10. Raw Data Preservation

The original Dreaddit CSV files must not be modified.

All preprocessing and derived datasets must be written to separate
locations under:

`data/processed/`

The raw source remains immutable.

---

## 11. Research Interpretation

Dreaddit labels represent a stress-identification research task.

They must not automatically be interpreted as clinical psychiatric
diagnoses.

Model outputs from this dataset are therefore research screening
signals and not clinical diagnoses.

---

## 12. Current Audit Status

| Audit Item                      | Result        |
| ------------------------------- | ------------- |
| Missing values                  | None detected |
| Duplicate complete rows         | None detected |
| Duplicate IDs within splits     | None detected |
| Train/test ID overlap           | 0             |
| Train/test post overlap         | 0             |
| Exact train/test text overlap   | 3             |
| Multiple segments per post      | Present       |
| Group-aware evaluation required | Yes           |
| Raw dataset modification        | Not permitted |

---

## 13. Decision

The supplied Dreaddit train/test partition will be preserved as the
official benchmark split.

New validation partitions will use `post_id`-grouped splitting.

The three exact text overlaps between the supplied train and test
partitions will be retained in the raw data and reported as a dataset
evaluation limitation rather than manually deleting or modifying the
source files.
