from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "phase10_limitations_threats_to_validity.md"


report = """# Phase 10.5 — Limitations & Threats to Validity

Project: Data Science Framework for Early Detection of Mental Health Risk

Framework Version: 0.1.0

## 1. Purpose

This document identifies the principal limitations and threats to
validity associated with the current research framework, datasets,
feature pipelines, baseline models, evaluation protocols, and
integration design.

The purpose is to establish clear boundaries around what the current
experiments support and what they do not support.

---

## 2. Dataset Limitations

### 2.1 Dataset heterogeneity

The framework combines evidence from Dreaddit, EATD, StudentLife and
MODMA provenance work.

These datasets differ in:

- participant populations,
- collection environments,
- modality,
- label definitions,
- sampling procedures,
- recording protocols,
- assessment procedures,
- dataset size,
- and evaluation protocol.

Consequently, results obtained from one dataset cannot automatically be
generalized to another dataset or population.

### 2.2 Dataset-specific targets

The experiments do not use one universal ground-truth target across all
modalities.

Dreaddit uses a stress-related text classification setting.

EATD uses a depression-related classification target.

StudentLife uses PHQ-9-derived behavioral regression.

MODMA was used for provenance, access and governance analysis rather
than a completed voice ML benchmark.

These targets should therefore not be treated as interchangeable
measurements.

### 2.3 Sample-size limitations

EATD contains 162 participants, with 83 participants in the training
split and 79 in the validation split.

The relatively small participant population limits the statistical
certainty and generalization of the resulting baseline estimates.

The StudentLife behavioral experiment also operates at participant
level, which reduces the effective sample size compared with the large
number of raw sensing records.

---

## 3. Class-Imbalance Limitations

Class imbalance is a major limitation of the EATD classification
experiments.

The EATD text dataset contains substantially more non-depression than
depression examples.

This creates an important distinction between overall accuracy and
positive-class detection.

For example, the EATD text baseline produced:

- Accuracy = 0.7848
- Precision = 0.0000
- Recall = 0.0000
- F1 = 0.0000
- ROC-AUC = 0.5094
- PR-AUC = 0.2105

The corresponding confusion matrix was:

| | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 62 | 2 |
| Actual 1 | 15 | 0 |

This demonstrates that accuracy alone can provide an incomplete view of
positive-class detection.

The same issue is visible in the EATD voice and multimodal experiments.

---

## 4. Label Limitations

### 4.1 Label definitions differ across datasets

The framework does not assume that a stress-related label, depression
label, and PHQ-9-derived score represent the same construct.

Each dataset is evaluated according to its own documented target
definition.

### 4.2 Proxy and questionnaire-derived targets

The StudentLife behavioral experiment uses PHQ-9-derived scores.

The current implementation uses the available questionnaire structure
and explicitly excludes the Q10 functional-difficulty item from the
computed PHQ-9 score.

The behavioral result therefore represents an association with the
available assessment target rather than an independently established
clinical diagnosis.

### 4.3 Threshold-defined classification

The EATD classification target uses the documented threshold applied to
the dataset's `new_label`.

Changing the threshold would change class membership and consequently
change model metrics.

Therefore, the current classification results are conditional on the
specified target construction.

---

## 5. Temporal Alignment Limitations

Temporal alignment is a major limitation of the StudentLife behavioral
experiment.

The available StudentLife assessment structure did not provide an exact
assessment timestamp that could be reliably aligned with all behavioral
features.

Therefore, the behavioral model should not be described as a validated
prospective early-warning model.

Instead, it is treated as a retrospective participant-level behavioral
association baseline.

A true early-prediction experiment would require:

1. a clearly defined prediction time,
2. a future outcome window,
3. features restricted to information available before that time,
4. participant-level temporal splitting,
5. and an independent future evaluation set.

---

## 6. Data Leakage and Split Limitations

### 6.1 Dreaddit text overlap

Three exact cross-split text overlaps were identified in Dreaddit.

These overlaps are retained and explicitly documented as an evaluation
limitation.

They do not justify discarding the complete experiment, but they reduce
confidence in interpreting the result as fully independent
generalization.

### 6.2 Participant-aware evaluation

Where participant-level data were available, evaluation was designed to
avoid mixing the same participant's observations across incompatible
training and evaluation boundaries.

EATD multimodal fusion additionally used train-only out-of-fold
predictions while preserving the final validation set.

### 6.3 EATD directory identifiers

Numeric participant-folder identifiers overlap between the EATD train
and validation directory structures.

The source directory structure was preserved, and the identifiers were
not automatically interpreted as participant leakage because the source
organization treats the split directories separately.

This remains a dataset-structure consideration that should be checked
again if additional EATD experiments are performed.

---

## 7. Evaluation Limitations

### 7.1 Different evaluation protocols

The framework contains:

- Dreaddit group-aware validation and supplied holdout evaluation,
- EATD official train/validation evaluation,
- EATD train-only OOF predictions for fusion,
- StudentLife participant-level cross-validation,
- and MODMA provenance/access analysis without ML benchmarking.

These protocols are not directly interchangeable.

### 7.2 No single universal benchmark

The reported metrics should not be combined into a single ranking across
modalities or datasets.

The evaluation is intended to describe the behavior of each experiment
under its own protocol.

### 7.3 Accuracy limitations

Accuracy is insufficient as the sole classification metric, especially
under class imbalance.

The framework therefore reports precision, recall, F1, ROC-AUC and
PR-AUC where appropriate.

For StudentLife regression, MAE, RMSE and R² are used instead.

---

## 8. Model Limitations

### 8.1 Baseline model scope

The current experiments primarily use interpretable and established
baseline methods, including:

- Logistic Regression,
- Random Forest,
- Ridge Regression,
- TF-IDF representations,
- and engineered acoustic/behavioral features.

These models establish reproducible baselines but do not represent the
full range of modern multimodal or foundation-model approaches.

### 8.2 EATD discrimination

The EATD text, voice and text + voice experiments showed limited
positive-class discrimination.

The final text + voice fusion result was:

- Accuracy = 0.7468
- Precision = 0.2222
- Recall = 0.1333
- F1 = 0.1667
- ROC-AUC = 0.5167
- PR-AUC = 0.2152

Therefore, the current EATD experiments do not establish strong
discriminative performance.

### 8.3 StudentLife regression

The StudentLife Random Forest baseline produced:

- MAE = 3.8488
- RMSE = 4.7222
- R² = -0.1510

The negative R² indicates that the current model does not provide a
strong explanatory/predictive fit under the evaluated protocol.

This result must be interpreted together with the temporal-alignment
limitation.

---

## 9. Calibration and Probability Limitations

A model score or class probability generated by these baselines should
not automatically be interpreted as a clinical probability.

The framework explicitly treats model scores as model-generated outputs.

Calibration analysis was performed for relevant experiments, but this
does not establish clinical calibration.

Clinical probability calibration would require appropriate clinical
reference populations, independent validation, calibration assessment,
and prospective evaluation.

---

## 10. External Validity

External validity is currently limited.

The framework has not established that the observed experimental
patterns generalize to:

- different countries,
- different demographic groups,
- different languages,
- different clinical settings,
- different recording devices,
- different social-media communities,
- or prospective real-world populations.

Dataset-specific performance should therefore not be presented as
population-level clinical performance.

---

## 11. Construct Validity

The framework uses computational features as measurable proxies for
complex human phenomena.

Examples include:

- linguistic characteristics,
- acoustic characteristics,
- behavioral sensing features,
- and questionnaire-derived scores.

These measurements do not directly represent the complete underlying
mental-health construct.

Therefore, model performance should not be interpreted as proof that a
specific feature independently represents a clinical condition.

---

## 12. Causal Interpretation

The experiments are predictive/associational rather than causal.

The framework does not establish that:

- a linguistic feature causes mental-health risk,
- an acoustic feature causes a change in mental-health status,
- a behavioral pattern causes a questionnaire score,
- or a model feature represents a causal mechanism.

Feature importance and explainability outputs should therefore be
interpreted as model behavior rather than causal evidence.

---

## 13. Explainability Limitations

The project includes feature importance and explainability artifacts.

However, feature importance indicates how a model uses available
features under a particular dataset and model specification.

It does not prove:

- causality,
- clinical relevance,
- universality,
- or individual-level clinical meaning.

Explainability outputs should therefore remain within the scope of
model interpretation.

---

## 14. MODMA Access Limitation

MODMA was incorporated into the research framework for provenance,
access, ethics, licensing and governance analysis.

The controlled raw-audio access required for a complete ML benchmark was
not available within the project workflow.

Therefore, no MODMA voice ML performance result is claimed.

This is preferable to constructing an unsupported benchmark from
incomplete or unavailable source data.

---

## 15. Privacy and Ethical Limitations

Mental-health-related text, voice and behavioral data can contain highly
sensitive information.

The framework therefore includes:

- pseudonymous identifiers,
- consent metadata,
- provenance tracking,
- no raw text/audio logging in the integration configuration,
- research-only governance,
- and explicit restrictions on consequential use.

However, these controls do not by themselves establish that the system
is ready for deployment with real-world sensitive populations.

Additional requirements would be necessary for any future operational
deployment.

---

## 16. Clinical Validity Boundary

The current project does not establish:

- clinical diagnostic validity,
- clinical screening validity,
- treatment recommendation validity,
- emergency triage validity,
- or clinical decision-support validity.

The system should therefore be presented as a research framework for
experimental multimodal risk-signal modeling.

It should not be presented as a diagnostic or autonomous clinical
decision system.

---

## 17. Threats to Internal Validity

Important internal-validity threats include:

1. Dataset-specific sampling procedures.
2. Class imbalance.
3. Small participant populations in some experiments.
4. Cross-split text overlap in Dreaddit.
5. Limited temporal alignment in StudentLife.
6. Dataset-specific label construction.
7. Threshold-dependent classification behavior.
8. Differences between train and validation populations.
9. Potential sensitivity to feature engineering choices.
10. Limited baseline model diversity.

These factors can affect the measured performance of individual
experiments.

---

## 18. Threats to External Validity

Important external-validity threats include:

1. Differences between benchmark datasets and real-world populations.
2. Differences in language and communication style.
3. Differences in recording environments.
4. Differences in demographic composition.
5. Differences in sensing devices and data collection procedures.
6. Differences in questionnaire and label definitions.
7. Lack of prospective external validation.
8. Lack of independent clinical-site validation.

Consequently, benchmark performance should not be generalized beyond
the evaluated datasets without additional evidence.

---

## 19. Threats to Construct Validity

Construct-validity threats include:

- treating behavioral measurements as proxies for complex psychological
  states,
- treating acoustic measurements as proxies for mental-health status,
- treating textual patterns as proxies for stress-related signals,
- using questionnaire-derived targets as operational labels,
- and interpreting model-generated scores beyond their validated
  experimental meaning.

These issues reinforce the need for careful terminology and explicit
research boundaries.

---

## 20. Mitigation Strategies

The current project already implements several mitigation strategies:

| Threat | Current mitigation |
| --- | --- |
| Data-quality problems | Schema and quality validation |
| Participant leakage | Participant-aware evaluation where applicable |
| Multimodal leakage | Train-only OOF fusion |
| Class imbalance | Multiple classification metrics and threshold analysis |
| Dataset heterogeneity | Dataset-specific evaluation |
| Unsupported probability fusion | Cross-dataset fusion disabled |
| Privacy risk | Pseudonymous IDs and restricted logging |
| Unsupported clinical interpretation | Research-only governance |
| Model/version ambiguity | Model registry |
| Integration inconsistency | Unified prediction schema |
| API regression risk | Automated API and full regression tests |

---

## 21. Recommended Future Validation

Future work should prioritize:

1. Larger participant cohorts.
2. Independent external validation.
3. Prospective temporal evaluation.
4. Better class-balance strategies.
5. Calibration on representative validation populations.
6. Robust multimodal representation learning.
7. Replication across multiple datasets.
8. Demographic and subgroup robustness analysis.
9. Controlled clinical validation where ethically and legally
   appropriate.
10. Longitudinal evaluation of prediction stability.

These are future research requirements, not capabilities established by
the current project.

---

## 22. Overall Validity Statement

The current framework provides a reproducible research infrastructure
for experimenting with text, voice and behavioral data under explicit
dataset-specific evaluation and governance boundaries.

The experiments demonstrate measurable predictive signal in some
settings and substantial limitations in others.

The evidence is insufficient to claim clinical validity, prospective
early-warning validity, or generalization to real-world clinical
populations.

The appropriate interpretation is therefore that the project establishes
a research and engineering framework with documented experimental
baselines, error analysis, governance and integration, while leaving
external, prospective and clinical validation as future work.

---

## Phase 10.5 Status

COMPLETE
"""


OUTPUT.write_text(report, encoding="utf-8")

print("=" * 60)
print("PHASE 10.5 COMPLETE")
print("=" * 60)
print(f"Created: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size} bytes")
print("=" * 60)