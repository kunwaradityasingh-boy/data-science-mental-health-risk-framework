from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "phase10_cross_modality_findings.md"


report = """# Phase 10.4 — Cross-Modality Findings

Project: Data Science Framework for Early Detection of Mental Health Risk

Framework Version: 0.1.0

## 1. Purpose

This report synthesizes findings from the text, voice, behavioral,
and dataset-specific multimodal experiments conducted in the framework.

The comparison is descriptive. The datasets use different populations,
label definitions, collection protocols, modalities, and evaluation
procedures. Therefore, the reported metrics must not be interpreted as
a single head-to-head benchmark.

---

## 2. Text Modality Findings

### Dreaddit

The Dreaddit experiments provided the strongest observed classification
signal among the evaluated text experiments.

The final supplied holdout results were:

| Configuration | Accuracy | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Text only | 0.7175 | 0.7363 | 0.8207 | 0.8412 |
| Text + social/engineered features | 0.7413 | 0.7601 | 0.8237 | 0.8416 |
| Text + all engineered features | 0.7594 | 0.7731 | 0.8422 | 0.8513 |

The validation experiment for the combined text + engineered-feature
model produced accuracy = 0.7766, F1 = 0.7896, ROC-AUC = 0.8503 and
PR-AUC = 0.8478.

These results indicate that textual information and engineered
features contained measurable predictive signal within the Dreaddit
evaluation protocol.

However, three exact cross-split text overlaps were identified and
retained as an evaluation limitation. The results therefore should not
be interpreted as evidence of clinical validity or generalization to
clinical populations.

### EATD Text

The EATD text baseline behaved differently.

The official validation result was:

- Accuracy = 0.7848
- Precision = 0.0000
- Recall = 0.0000
- F1 = 0.0000
- ROC-AUC = 0.5094
- PR-AUC = 0.2105

The confusion matrix was:

| | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 62 | 2 |
| Actual 1 | 15 | 0 |

The experiment therefore demonstrates that moderate accuracy can coexist
with very poor positive-class detection when the class distribution is
imbalanced.

---

## 3. Voice Modality Findings

The framework evaluated EATD acoustic features using participant-level
aggregation of response-level acoustic features.

The Logistic Regression voice baseline produced:

- Accuracy = 0.7722
- Precision = 0.2857
- Recall = 0.1333
- F1 = 0.1818
- ROC-AUC = 0.5531
- PR-AUC = 0.2388

Its confusion matrix was:

| | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 59 | 5 |
| Actual 1 | 13 | 2 |

The Random Forest baseline produced:

- Accuracy = 0.7848
- Precision = 0.0000
- Recall = 0.0000
- F1 = 0.0000
- ROC-AUC = 0.4240
- PR-AUC = 0.2096

Its confusion matrix was:

| | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 62 | 2 |
| Actual 1 | 15 | 0 |

The voice experiments therefore provide evidence that acoustic features
can be processed and evaluated through the framework, but the tested
baselines showed limited positive-class discrimination on the EATD
evaluation set.

---

## 4. Behavioral Modality Findings

The StudentLife experiment used participant-level behavioral features
derived from sensing data and evaluated them against PHQ-9 scores.

The Random Forest regression baseline produced:

- MAE = 3.8488
- RMSE = 4.7222
- R² = -0.1510

The Ridge regression baseline produced:

- MAE = 5.3253
- RMSE = 6.7992
- R² = -2.1558

The behavioral experiment demonstrates an end-to-end behavioral
feature engineering and participant-level modeling pipeline.

However, the StudentLife assessment lacked the exact temporal alignment
required for a true prospective early-prediction experiment. The
behavioral result is therefore treated as a retrospective participant-
level association baseline rather than as validated temporal early
prediction.

---

## 5. Dataset-Specific Multimodal Findings

The EATD dataset was also evaluated using text + voice fusion.

The final corrected fusion result was:

- Accuracy = 0.7468
- Precision = 0.2222
- Recall = 0.1333
- F1 = 0.1667
- ROC-AUC = 0.5167
- PR-AUC = 0.2152

Confusion matrix:

| | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 57 | 7 |
| Actual 1 | 13 | 2 |

The fusion experiment used train-only out-of-fold predictions and
preserved the final validation set as an untouched evaluation set.

The result shows that adding another modality did not automatically
produce strong discrimination under this dataset-specific protocol.

The EATD fusion result should therefore be interpreted as an empirical
finding about the tested models and dataset rather than as evidence
against multimodal learning in general.

---

## 6. Cross-Modality Observations

Several observations can be supported across the experiments.

### 6.1 Different modalities expose different data characteristics

Text experiments operate on linguistic and engineered textual signals.

Voice experiments operate on acoustic characteristics extracted from
recorded responses.

Behavioral experiments operate on participant-level sensing-derived
features.

These modalities therefore represent different measurement processes
and should not be assumed to encode the same signal.

### 6.2 Dataset effects are substantial

Dreaddit and EATD produced substantially different text-model behavior.

Dreaddit produced measurable discrimination under its evaluation
protocol, whereas EATD text produced ROC-AUC close to chance and zero
positive-class recall at the evaluated operating threshold.

This difference demonstrates why a result from one dataset cannot
automatically be generalized to another population or collection
protocol.

### 6.3 Multimodal fusion requires validation

The EATD text + voice experiment did not demonstrate strong
discrimination.

This indicates that combining modalities is not sufficient by itself.
Successful multimodal learning would require appropriate alignment,
sufficient sample size, compatible labels, robust representations and
independent validation.

### 6.4 Behavioral data require temporal alignment

StudentLife contains rich sensing information, but the available
assessment structure did not provide exact temporal alignment between
behavioral observations and the PHQ-9 assessment.

Consequently, the current experiment cannot establish that the
behavioral features predicted future assessment outcomes.

### 6.5 Class imbalance affects interpretation

EATD contains substantially more non-depression than depression
examples.

This makes accuracy alone insufficient for evaluating positive-class
detection.

Precision, recall, F1, ROC-AUC and PR-AUC therefore remain important
for interpreting the classification experiments.

---

## 7. Cross-Modality Comparison Table

| Modality | Dataset | Task | Primary observation | Major limitation |
| --- | --- | --- | --- | --- |
| Text | Dreaddit | Classification | Measurable discrimination under evaluated protocol | Cross-split text overlap limitation and dataset specificity |
| Text | EATD | Classification | Poor positive-class detection | Class imbalance and weak discrimination |
| Voice | EATD | Classification | Acoustic pipeline successfully evaluated | Weak positive-class discrimination |
| Behavior | StudentLife | Regression | Behavioral association baseline established | No exact temporal alignment |
| Text + Voice | EATD | Dataset-specific fusion | Fusion pipeline successfully evaluated | Weak discrimination on evaluation set |

---

## 8. What the Current Framework Demonstrates

The current framework demonstrates an end-to-end research pipeline
covering:

1. Dataset provenance and acquisition tracking.
2. Data schema and quality validation.
3. Text feature engineering.
4. Acoustic feature engineering.
5. Behavioral feature engineering.
6. Dataset-specific baseline modeling.
7. Participant-aware evaluation.
8. Out-of-fold multimodal fusion.
9. Error analysis.
10. Model registry and governance.
11. Unified prediction schema.
12. Risk-policy integration.
13. FastAPI framework integration.
14. Automated regression testing.

The experiments demonstrate the feasibility of implementing a common
research framework across heterogeneous modalities while preserving
dataset-specific evaluation boundaries.

---

## 9. What the Current Framework Does Not Demonstrate

The experiments do not establish:

- Clinical diagnostic validity.
- Clinical screening validity.
- Generalization to real-world clinical populations.
- Prospective early-warning performance.
- Cross-dataset probability comparability.
- Clinical probability calibration.
- Causal relationships between behavioral/acoustic/textual features
  and mental-health outcomes.
- Safety for consequential automated decision-making.

The framework is therefore explicitly research-only.

---

## 10. Research Implications

The cross-modality experiments support the need for modality-specific
evaluation combined with strict dataset and governance boundaries.

Future research should investigate:

- Larger and more diverse participant populations.
- Stronger temporal alignment between observations and outcomes.
- External validation on independent datasets.
- Better handling of class imbalance.
- Calibration and uncertainty estimation.
- Robust multimodal representation learning.
- Prospective evaluation.
- Reproducible and auditable model monitoring.

These directions are research opportunities rather than claims that the
current framework already satisfies those requirements.

---

## Phase 10.4 Status

COMPLETE
"""

OUTPUT.write_text(report, encoding="utf-8")

print("=" * 60)
print("PHASE 10.4 COMPLETE")
print("=" * 60)
print(f"Created: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size} bytes")
print("=" * 60)