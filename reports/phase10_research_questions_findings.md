# Phase 10.6 — Research Questions & Findings

Project: Data Science Framework for Early Detection of Mental Health Risk

Framework Version: 0.1.0

## 1. Research Objective

The project investigates whether a unified data-science framework can
support reproducible experimentation with text, voice and behavioral
signals associated with mental-health-related research targets.

The study focuses on:

- reproducible data preparation,
- modality-specific feature engineering,
- baseline predictive modeling,
- multimodal experimentation,
- participant-aware evaluation,
- error analysis,
- governance,
- and research-only system integration.

The project does not attempt to establish clinical diagnosis or clinical
validity.

---

# 2. Research Questions

## RQ1 — Can heterogeneous mental-health-related datasets be processed
through a common research framework?

### Finding

Yes, the project demonstrates an end-to-end framework capable of
processing heterogeneous text, voice and behavioral datasets while
maintaining dataset-specific schemas, feature pipelines, evaluation
protocols and governance boundaries.

The implemented framework includes:

- data schemas,
- data-quality validation,
- provenance tracking,
- modality-specific feature engineering,
- model baselines,
- model registry,
- unified prediction schema,
- risk-policy integration,
- FastAPI integration,
- and automated regression testing.

The datasets are not treated as one interchangeable dataset. Their
different targets and evaluation protocols remain explicitly separated.

### Evidence

The framework successfully integrated:

- Dreaddit text,
- EATD text,
- EATD voice,
- EATD text + voice,
- StudentLife behavioral data,
- and MODMA provenance/access information.

---

## RQ2 — Does textual information provide measurable predictive signal
under the evaluated datasets?

### Finding

The answer differs by dataset.

Dreaddit produced measurable classification performance under its
evaluation protocol.

The final supplied Dreaddit holdout results were:

| Configuration | Accuracy | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Text only | 0.7175 | 0.7363 | 0.8207 | 0.8412 |
| Text + social/engineered | 0.7413 | 0.7601 | 0.8237 | 0.8416 |
| Text + all engineered | 0.7594 | 0.7731 | 0.8422 | 0.8513 |

The validation experiment for the combined text + engineered model
produced:

- Accuracy = 0.7766
- F1 = 0.7896
- ROC-AUC = 0.8503
- PR-AUC = 0.8478

However, three exact cross-split text overlaps were identified and
retained as an evaluation limitation.

EATD text behaved differently:

- Accuracy = 0.7848
- Precision = 0.0000
- Recall = 0.0000
- F1 = 0.0000
- ROC-AUC = 0.5094
- PR-AUC = 0.2105

Therefore, textual predictive signal is dataset-dependent.

### Conclusion

The experiments support the existence of measurable textual signal in
some benchmark settings, but do not establish universal or clinical
text-based prediction.

---

## RQ3 — Can acoustic voice features provide useful predictive signal?

### Finding

The EATD acoustic pipeline successfully transformed participant voice
responses into participant-level acoustic features and evaluated
baseline classifiers.

The Logistic Regression baseline produced:

- Accuracy = 0.7722
- Precision = 0.2857
- Recall = 0.1333
- F1 = 0.1818
- ROC-AUC = 0.5531
- PR-AUC = 0.2388

The Random Forest baseline produced:

- Accuracy = 0.7848
- Precision = 0.0000
- Recall = 0.0000
- F1 = 0.0000
- ROC-AUC = 0.4240
- PR-AUC = 0.2096

### Conclusion

The experiments demonstrate a reproducible acoustic-feature pipeline,
but the tested EATD voice baselines showed limited positive-class
discrimination.

The results do not establish clinical usefulness of voice-based
prediction.

---

## RQ4 — Can behavioral sensing features predict questionnaire-derived
outcomes?

### Finding

The StudentLife experiment established a participant-level behavioral
feature engineering and regression pipeline.

The Random Forest baseline produced:

- MAE = 3.8488
- RMSE = 4.7222
- R² = -0.1510

The Ridge baseline produced:

- MAE = 5.3253
- RMSE = 6.7992
- R² = -2.1558

The behavioral experiment therefore provides a reproducible baseline
for studying associations between sensing-derived behavioral features
and PHQ-9-derived scores.

However, exact temporal alignment between behavioral observations and
assessment outcomes was unavailable.

### Conclusion

The current experiment supports retrospective participant-level
behavioral association analysis, not validated prospective early
prediction.

---

## RQ5 — Does combining text and voice automatically improve predictive
performance?

### Finding

The EATD text + voice fusion experiment did not demonstrate strong
discrimination under the tested protocol.

Final result:

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

The fusion system used train-only out-of-fold predictions and preserved
the final validation set for evaluation.

### Conclusion

Multimodal fusion cannot be assumed to improve performance simply
because additional modalities are added.

Appropriate alignment, representation learning, sample size,
compatible targets and independent validation remain necessary.

The result is a dataset-specific finding and should not be generalized
as evidence that multimodal learning is ineffective in general.

---

## RQ6 — How important is evaluation design for interpreting model
performance?

### Finding

Evaluation design is critical.

The project uses different protocols according to dataset structure:

- group-aware validation for Dreaddit,
- supplied holdout evaluation for Dreaddit,
- official train/validation evaluation for EATD,
- train-only OOF predictions for EATD fusion,
- participant-level cross-validation for StudentLife.

The experiments also explicitly report multiple metrics rather than
accuracy alone.

This is particularly important for EATD, where moderate accuracy
coexists with poor positive-class recall.

### Conclusion

Model performance must be interpreted together with class distribution,
evaluation protocol, split design, threshold behavior and the underlying
target definition.

---

## RQ7 — Can the framework provide a common integration layer without
performing unsupported cross-dataset probability fusion?

### Finding

Yes.

The framework implements:

- a model registry,
- unified prediction schema,
- modality-specific predictions,
- risk-policy mapping,
- integration service,
- framework prediction service,
- and FastAPI endpoints.

Cross-dataset probability fusion is explicitly disabled because the
underlying datasets differ in population, target, modality and
evaluation protocol.

The framework can therefore integrate modality outputs while preserving
their dataset-specific boundaries.

### Conclusion

A common software architecture is feasible without assuming that
probabilities from heterogeneous datasets are directly comparable.

---

## RQ8 — Can governance boundaries be incorporated into the technical
architecture?

### Finding

Yes.

The framework includes explicit research-only governance.

The implemented architecture includes:

- pseudonymous identifiers,
- consent metadata,
- provenance tracking,
- restricted raw-data logging,
- model version tracking,
- risk-policy configuration,
- UNKNOWN handling,
- human-review signaling,
- and restrictions against clinical diagnosis and consequential
  automated decision-making.

### Conclusion

Governance can be implemented as part of the technical architecture
rather than being treated solely as external documentation.

---

# 3. Consolidated Findings

## Finding 1 — A unified research infrastructure is feasible

The project successfully combines data validation, modality-specific
processing, model experimentation, evaluation, governance and API
integration within one framework.

## Finding 2 — Predictive behavior is dataset-dependent

Dreaddit and EATD produced substantially different text-model behavior.
Therefore, benchmark performance cannot automatically be generalized
across datasets.

## Finding 3 — Positive-class detection remains a major limitation

The EATD text, voice and multimodal experiments showed limited
positive-class recall.

This demonstrates why accuracy must not be interpreted independently
of precision, recall, F1, ROC-AUC and PR-AUC.

## Finding 4 — Multimodal fusion requires empirical validation

The EATD fusion experiment did not demonstrate strong discrimination.
Adding modalities alone does not guarantee improved predictive
performance.

## Finding 5 — Behavioral early prediction requires temporal alignment

The StudentLife experiment provides a behavioral association baseline,
but its current temporal structure does not support a validated
prospective early-warning claim.

## Finding 6 — Dataset boundaries are essential

The framework intentionally avoids treating heterogeneous datasets as a
single benchmark and disables unsupported cross-dataset probability
fusion.

## Finding 7 — Governance can be integrated into system design

Research-only constraints, privacy controls, model versioning,
uncertainty/UNKNOWN handling and human-review signaling are represented
within the technical architecture.

---

# 4. Negative and Null Findings

A rigorous research project must also report findings that did not
support the intended hypothesis.

The current experiments provide several such findings:

1. EATD text did not demonstrate useful positive-class discrimination
   under the evaluated baseline.
2. EATD voice baselines did not demonstrate strong positive-class
   discrimination.
3. EATD text + voice fusion did not demonstrate strong discrimination.
4. StudentLife behavioral regression did not establish strong
   predictive fit under the evaluated protocol.
5. MODMA could not provide a completed ML benchmark because controlled
   raw-audio access was unavailable.
6. The framework cannot currently claim prospective clinical early
   prediction.
7. Cross-dataset probability fusion is not scientifically justified by
   the current evidence.

These are research findings and limitations rather than failures of the
engineering framework.

---

# 5. Research Contribution

The primary contribution of the project is the construction of a
reproducible, governance-aware research framework for heterogeneous
mental-health-related data.

The contribution consists of:

- unified data contracts,
- modality-specific feature pipelines,
- dataset-specific baseline experiments,
- participant-aware evaluation,
- leakage-aware multimodal fusion,
- structured error analysis,
- model registry,
- unified prediction schema,
- risk-policy integration,
- research-only API integration,
- and explicit validity boundaries.

The framework emphasizes reproducibility and responsible interpretation
rather than claiming clinical deployment readiness.

---

# 6. Answer to the Overall Research Objective

The project demonstrates that a common engineering and research
framework can support experimentation across text, voice and behavioral
data while maintaining explicit dataset-specific boundaries.

The experimental results show that predictive signal is not uniform
across datasets or modalities.

Some experiments demonstrate measurable discrimination, while other
experiments show weak or near-chance discrimination and substantial
positive-class detection limitations.

Therefore, the current evidence supports the framework as a research
infrastructure and experimental baseline system, but does not establish
clinical validity, prospective early-warning validity or generalization
to real-world clinical populations.

---

# 7. Future Research Questions

The completed experiments motivate the following future questions:

1. How does performance change with larger and more diverse participant
   populations?

2. Can temporally aligned longitudinal behavioral data improve
   prospective prediction?

3. Can independent external datasets reproduce the observed findings?

4. Can calibrated multimodal models improve positive-class detection?

5. How robust are the models across demographic and linguistic
   subgroups?

6. Can stronger multimodal representations improve over the current
   baseline fusion approach?

7. How stable are model predictions over time?

8. What evaluation protocol best measures early-warning performance
   without introducing temporal leakage?

These questions are proposed for future research and are not answered
by the current experiments.

---

# 8. Final Research Findings Table

| Research Question | Evidence | Finding |
| --- | --- | --- |
| RQ1 — Common framework feasibility | Multiple modality pipelines and integration services | Demonstrated |
| RQ2 — Textual signal | Dreaddit and EATD experiments | Dataset-dependent |
| RQ3 — Voice signal | EATD acoustic experiments | Limited positive-class discrimination |
| RQ4 — Behavioral signal | StudentLife regression | Retrospective association baseline |
| RQ5 — Multimodal fusion | EATD text + voice | No strong discrimination demonstrated |
| RQ6 — Evaluation design | Multiple split and metric strategies | Critical for interpretation |
| RQ7 — Unified integration | Registry, schema, service and API | Demonstrated |
| RQ8 — Governance integration | Risk policy, privacy and research-only controls | Demonstrated |

---

## Phase 10.6 Status

COMPLETE
