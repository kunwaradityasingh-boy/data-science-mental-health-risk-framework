# Model Governance and Risk Output Specification

## 1. Purpose

This document defines the intended use, output boundaries,
safety constraints, explainability requirements, and governance
rules for the Data Science Framework for Early Detection of
Mental Health Risk.

The framework is a research prototype for early risk-signal
screening and referral-support research.

It is not a diagnostic system.

---

## 2. Intended Use

The system is intended to:

1. Process permitted research data modalities.
2. Generate machine-learning-based risk signals.
3. Provide model scores and supporting evidence.
4. Surface potentially higher-risk observations for human review.
5. Support research into multimodal early-warning systems.
6. Provide reproducible evaluation and monitoring artifacts.

The system is not intended to independently diagnose a
mental-health disorder.

---

## 3. Non-Intended Uses

The system must not be used as the sole basis for:

- Medical diagnosis.
- Psychiatric diagnosis.
- Emergency triage.
- Academic punishment.
- Student disciplinary action.
- Employment decisions.
- Insurance decisions.
- Financial eligibility decisions.
- Legal decisions.
- Automated denial of services.
- Automated treatment decisions.

---

## 4. Risk Output Semantics

The primary model output is a research risk signal.

Example:

    Risk signal: Elevated

The numerical model score must not automatically be described
as a clinical probability.

Example:

    Model score: 0.78

This should be interpreted as a model-generated score under the
specific trained model and evaluation protocol.

It must not automatically be interpreted as:

    "78% probability of depression."

---

## 5. Proposed Output Structure

Each prediction should contain:

- prediction identifier
- pseudonymous subject identifier
- timestamp
- risk signal
- model score
- model version
- data modalities available
- data quality indicators
- missing modality indicators
- contributing evidence
- human review recommendation

Example:

    Prediction ID:
        PRED-000001

    Risk signal:
        Elevated

    Model score:
        0.78

    Model version:
        text-engineered-logistic-v1

    Available modalities:
        text
        behavior

    Missing modalities:
        voice

    Data quality:
        acceptable

    Contributing evidence:
        text-derived features
        linguistic feature groups
        behavioral feature groups

    Human review:
        recommended

---

## 6. Risk Bands

Risk bands are a communication layer and must not be
interpreted as medical diagnoses.

Initial research categories:

    LOW
    MODERATE
    HIGH
    UNKNOWN

The thresholds for these bands must not be selected using
the final holdout test set.

Threshold development must use training/validation data
only.

The final holdout remains reserved for final evaluation.

---

## 7. UNKNOWN State

The system must support an UNKNOWN state.

UNKNOWN may be returned when:

- Required data is missing.
- Data quality is insufficient.
- Consent does not permit processing.
- The model does not support the supplied modality.
- Input is outside the validated data distribution.
- A required preprocessing step fails.
- Model inference fails.
- Confidence or calibration information is unavailable.

The system must not force a LOW/MODERATE/HIGH classification
when evidence is insufficient.

---

## 8. Human Review

A high-risk research signal should be treated as a trigger
for human review rather than an automated diagnosis.

Human reviewers should have access to:

- model score
- model version
- data quality
- available modalities
- missing modalities
- contributing model evidence
- limitations of the underlying dataset
- relevant consent status

Human review remains responsible for contextual interpretation.

---

## 9. Explainability

The system should provide model-level and
instance-level explanations.

Model-level explanations may include:

- feature coefficients
- SHAP feature importance
- feature-group summaries
- model performance metrics

Instance-level explanations may include:

- positive contributors
- negative contributors
- model score
- prediction
- data-quality information

Model explanations describe associations learned by the
model.

They must not be presented as proof that a feature causes
mental-health risk.

---

## 10. Dataset Limitations

The current NLP implementation uses Dreaddit.

The Dreaddit task represents a stress-related research
classification task.

Its labels must not automatically be interpreted as
clinical psychiatric diagnoses.

The dataset also contains multiple segments from the same
posts and three exact-text overlaps between the supplied
training and test files were identified during auditing.

These limitations must be reported in final evaluation.

---

## 11. Evaluation Boundary

The supplied Dreaddit test set is treated as a final
holdout evaluation dataset.

Model selection, threshold tuning, and feature engineering
decisions should not be based on final holdout performance.

Holdout results are descriptive evidence for the evaluated
dataset and protocol.

They do not establish clinical validity.

---

## 12. Calibration

Calibration metrics should be reported separately from
discrimination metrics.

Current evaluation includes:

- Brier score
- Log loss
- Mean absolute calibration error
- Calibration-bin analysis

A calibrated model score does not by itself establish
clinical probability.

---

## 13. Privacy

The system should use:

- pseudonymous identifiers
- explicit consent references
- minimum necessary data
- controlled data access
- secure storage
- audit logging
- data retention policies
- revocation handling

Public availability of data does not automatically establish
permission for individual mental-health profiling.

---

## 14. Consent

Processing must respect the applicable consent record.

The system should verify:

- consent identifier
- purpose
- permitted modalities
- consent status
- expiration
- revocation

Revoked or expired consent must prevent processing for
purposes requiring active consent.

---

## 15. Security

The implementation should consider:

- authentication
- authorization
- encryption in transit
- encryption at rest
- secret management
- audit logs
- access controls
- secure API validation
- rate limiting
- input validation

Sensitive data must not be written to logs unnecessarily.

---

## 16. Model Versioning

Every prediction should be traceable to:

- model version
- feature pipeline version
- preprocessing version
- dataset version
- evaluation protocol version

Model artifacts should be stored separately from raw data.

---

## 17. Monitoring

Future production-oriented evaluation should monitor:

- data drift
- feature drift
- prediction drift
- missing-data rates
- calibration drift
- subgroup performance
- false-positive rates
- false-negative rates
- model failures

Monitoring results should be retained as auditable artifacts.

---

## 18. Regulatory Boundary

Before any real-world deployment, intended use and
functionality must be reviewed against applicable laws,
regulations, institutional requirements, and medical-device
or clinical-decision-support requirements.

The research prototype should not be represented as a
clinically validated medical device.

---

## 19. Current Research Status

Current status:

    Research prototype

Validated capabilities:

- Data schema validation
- Dataset auditing
- Leakage auditing
- Post-aware validation
- Text baseline
- Engineered-feature baseline
- Combined model
- Ablation analysis
- Repeated validation
- Statistical comparison
- Final holdout evaluation
- Error analysis
- Explainability
- Calibration analysis
- Governance specification

Not yet established:

- Clinical validity
- Prospective clinical utility
- External clinical validation
- Real-world deployment safety
- Generalization across populations
- Generalization across institutions
- Generalization across languages
- Clinical diagnostic accuracy

---

## 20. Final Safety Principle

The system should assist qualified human decision-makers
rather than replace them.

A machine-learning prediction is evidence generated under
a particular dataset, model, preprocessing pipeline, and
evaluation protocol.

It is not a diagnosis.
