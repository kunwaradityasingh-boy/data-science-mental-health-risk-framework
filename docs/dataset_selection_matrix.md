# Dataset Selection Matrix

## Purpose

This document records the evaluation of candidate datasets for the
Data Science Framework for Early Detection of Mental Health Risk.

Dataset selection must consider research relevance, label quality,
data quality, privacy, licensing/access conditions, reproducibility,
and potential data leakage.

---

## Evaluation Criteria

| Criterion            | Description                                                              |
| -------------------- | ------------------------------------------------------------------------ |
| Modality             | Text, voice, or behavioral data                                          |
| Population           | Population represented by the dataset                                    |
| Sample Size          | Number of subjects/records                                               |
| Labels               | Available target labels or annotations                                   |
| Label Meaning        | What the labels actually represent                                       |
| Clinical Status      | Whether labels represent clinical diagnosis or proxy/research annotation |
| Temporal Information | Whether observations have meaningful timestamps                          |
| Subject IDs          | Whether observations can be grouped by subject                           |
| Data Quality         | Missingness, duplicates, invalid values, etc.                            |
| Class Balance        | Distribution of target classes                                           |
| Leakage Risk         | Risk of subject/data overlap between splits                              |
| License/Access       | Conditions for research use                                              |
| Reproducibility      | Ease of reproducing experiments                                          |
| Framework Fit        | Relevance to the proposed system                                         |
| Limitations          | Important known limitations                                              |

---

## Candidate Datasets

Candidate datasets will be added only after source verification.

| Dataset     | Modality   | Population                                           |                                                                         Size | Labels                                        | Label Meaning                                                              | Access/License                                        | Leakage Risk                               | Framework Fit              | Decision                         |
| ----------- | ---------- | ---------------------------------------------------- | ---------------------------------------------------------------------------: | --------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------ | -------------------------- | -------------------------------- |
| Dreaddit    | Text       | Reddit social-media users/posts                      |                                         ~190K posts; ~3.5K labelled segments | Stress / non-stress                           | Crowd-annotated stress signal, not clinical diagnosis                      | Verify dataset distribution terms before acquisition  | Post/user overlap and text duplication     | High for NLP baseline      | Candidate - first implementation |
| MODMA       | Voice      | Research participants with mental-health assessments |                                Verify exact usable audio cohort after access | Psychological/mental-health assessment labels | Preserve original study definition; not automatically a clinical diagnosis | EULA / controlled research access conditions          | Multiple recordings from same participant  | High for voice module      | Candidate                        |
| StudentLife | Behavioral | Dartmouth students                                   | Original study cohort 48; usable sample depends on selected streams/outcomes | Surveys / EMA / behavioral outcomes           | Questionnaire/assessment-derived outcomes, depending on task               | Public research dataset; exact terms to be documented | Same subject across time; temporal leakage | High for behavioral module | Candidate                        |

---

## Selection Rules

A dataset should not be selected only because it produces high model accuracy.

Before selection, verify:

1. Dataset provenance.
2. Population represented.
3. Label definition.
4. Labeling methodology.
5. Access and licensing conditions.
6. Subject/speaker identifiers where applicable.
7. Temporal structure where applicable.
8. Missing-data characteristics.
9. Class distribution.
10. Potential train/test leakage.
11. Reproducibility.
12. Relevance to the research question.

---

## Research Safety Rule

A dataset label must not automatically be interpreted as a clinical diagnosis.

The project will distinguish between:

- Clinical diagnosis
- Clinician-rated assessment
- Research annotation
- Questionnaire-derived label
- Proxy label
- Self-reported status
- Machine-generated label

---

## Dataset Decision Log

| Date | Dataset | Decision | Reason                | Evidence |
| ---- | ------- | -------- | --------------------- | -------- |
| TBD  | TBD     | Pending  | Awaiting verification | TBD      |
