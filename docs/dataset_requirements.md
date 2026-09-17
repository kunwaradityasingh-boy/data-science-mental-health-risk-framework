# Dataset Requirements

## 1. Research Framework

Project:
Data Science Framework for Early Detection of Mental Health Risk

Primary purpose:
Research-oriented early-risk screening and referral-support framework.

The system is not intended to provide a clinical diagnosis.

---

## 2. Required Modalities

The framework considers three major data modalities:

1. Text / NLP
2. Voice / Speech
3. Behavioral / Longitudinal data

---

## 3. Dataset Requirements

### 3.1 Text Dataset

Required properties:

- Text observations must be available.
- Dataset must provide a documented target label or annotation.
- Label definition must be recorded.
- Source and licensing information must be documented.
- Data preprocessing must be reproducible.
- Personally identifying information must not be intentionally introduced into the project.
- Dataset should be suitable for NLP experimentation.

Potential features:

- TF-IDF
- n-grams
- Word embeddings
- Transformer embeddings

---

### 3.2 Voice Dataset

Required properties:

- Audio recordings must be available.
- Audio format and sampling information should be documented.
- Dataset must provide usable labels or annotations.
- Speaker/subject identifiers should be available where permitted.
- Train/test separation must prevent speaker leakage.
- Audio preprocessing must be reproducible.

Potential features:

- MFCC
- Spectral features
- Prosodic features
- Voice quality features

---

### 3.3 Behavioral / Longitudinal Dataset

Required properties:

- Repeated observations over time are preferred.
- Subject identifiers must allow longitudinal grouping where permitted.
- Temporal information should be available.
- Behavioral variables should be documented.
- Target labels or outcome definitions must be documented.
- Train/test splitting must be subject-aware and time-aware where appropriate.

Potential features:

- Event frequency
- Time-of-day patterns
- Activity duration
- Attendance
- Submission timing
- Screen/activity patterns

---

## 4. Label Requirements

For every dataset, document:

- Original label
- Normalized label
- Label source
- Labeling methodology
- Whether the label represents a clinical diagnosis
- Whether the label is a proxy or research annotation

The framework must not automatically interpret a research label as a clinical diagnosis.

---

## 5. Data Quality Requirements

Each dataset must be checked for:

- Missing values
- Duplicate records
- Invalid timestamps
- Invalid numeric values
- Inconsistent labels
- Outliers
- Data leakage
- Subject overlap between splits
- Class imbalance

---

## 6. Privacy Requirements

The project should use:

- Pseudonymous identifiers
- Minimum necessary data
- Documented data provenance
- Documented consent/licensing where applicable
- No unnecessary direct identifiers

Public availability of data must not automatically be treated as equivalent to informed consent for individual mental-health profiling.

---

## 7. Dataset Documentation

For every selected dataset, maintain:

- Dataset name
- Dataset source
- Dataset URL/reference
- Population
- Sample size
- Modalities
- Features
- Label definition
- Collection period, if available
- License/access conditions
- Known limitations
- Intended research use
