# Dataset Provenance and Access Registry

## Purpose

This registry records the provenance, access conditions, label origin,
and research-use considerations for datasets considered by the framework.

Dataset access must be verified before downloading or using data in experiments.

---

## 1. Dreaddit

### Basic Information

- Dataset: Dreaddit
- Modality: Text
- Domain: Social-media text
- Primary research task: Stress identification
- Paper: Turcan and McKeown, 2019
- Venue: LOUHI 2019
- DOI: 10.18653/v1/D19-6213

### Dataset Description

The published dataset contains approximately 190K Reddit posts from
five categories of Reddit communities.

Approximately 3.5K labelled segments were taken from 3K posts.

### Label Provenance

The labelled segments were annotated using Amazon Mechanical Turk.

The labels represent a stress-identification research task.

They must not automatically be interpreted as clinical psychiatric diagnoses.

### Access

Publication source:

https://aclanthology.org/D19-6213/

Raw dataset distribution source:

https://www.cs.columbia.edu/~eturcan/data/dreaddit.zip

The official ACL Anthology attachment was acquired and inspected.

The raw Dreaddit dataset was subsequently acquired from the
author-hosted distribution source and extracted locally.

### Official Attachment

- Archive: D19-6213.Attachment.zip
- Download date: 2026-09-17
- Archive size: 331,618 bytes
- SHA-256: 05EF5F84B9B22C054D37C02C1C2B86002B6BFF6989FB4380A105EFFFED09F0B5
- Archive contents:
  - Dreaddit_Appendix.pdf

### Appendix Contents

The official appendix contains:

- Full sample Reddit posts with identifying information removed
- Full Mechanical Turk annotation guidelines
- Model parameter settings
- Additional error-analysis examples

The annotation instructions describe three possible worker selections:

- Stress
- Not Stress
- Can't Tell

The appendix is supporting research documentation and is not a substitute
for the complete raw labelled dataset.

### Raw Dataset Acquisition

- Complete raw dataset acquired: Yes
- Download date: 2026-09-17
- Archive: dreaddit.zip
- Archive size: 1,348,791 bytes
- SHA-256: 6C7D8859764231CCA47410D6995D8941AC45B08A446A561923FD73B34F3F61CB
- Extracted files:
  - dreaddit-train.csv
  - dreaddit-test.csv
- Extracted location:
  - data/raw/dreaddit/extracted/

### Dataset Version

The downloaded distribution does not explicitly specify a dataset
version or release identifier.

The downloaded archive checksum is recorded to support reproducibility.

### License / Terms

The raw dataset was acquired from the author-hosted distribution source.

License and redistribution terms require authoritative verification
before redistribution or publication of the raw data.

The public availability of the publication does not by itself establish
permission to redistribute the underlying Reddit-derived data.

### Leakage Considerations

Potential risks:

- Multiple segments originating from the same post
- Multiple posts from the same user
- Near-duplicate text
- Community/source effects
- Exact text overlap across supplied train/test data

The supplied train/test split was audited.

Results:

- Overlapping record IDs: 0
- Overlapping post IDs: 0
- Exact text overlaps across train/test: 3

The three exact text overlaps are retained in the raw dataset and documented
as an evaluation limitation.

For newly generated validation partitions, post-level group-aware splitting
must be used.

### Data Quality Audit

The acquired dataset was inspected for:

- Dataset dimensions
- Missing values
- Duplicate rows
- Duplicate post IDs
- Subreddit distribution
- Text length
- Label distribution
- Cross-split ID overlap
- Cross-split post overlap
- Exact text overlap
- Multiple segments per post

The detailed audit results are documented in the Dreaddit evaluation
protocol and audit scripts.

### Local Storage

Official attachment:

data/raw/dreaddit/D19-6213.Attachment.zip

Inspection copy:

data/raw/dreaddit/inspection/Dreaddit_Appendix.pdf

Downloaded raw dataset:

data/raw/dreaddit/dreaddit.zip

Extracted dataset:

data/raw/dreaddit/extracted/

### Git Tracking

Raw dataset archives, extracted CSV files, and inspection artifacts are
excluded from Git tracking.

Dataset documentation and audit scripts are tracked by Git.

### Current Status

Dreaddit raw dataset acquired and extracted successfully.

Dataset quality and leakage audits completed.

Evaluation protocol established.

License and redistribution terms remain pending authoritative verification.

---

## 2. MODMA

### Basic Information

- Dataset: MODMA
- Full name: A Multi-modal Open Dataset for Mental-disorder Analysis
- Modality: Voice / spoken language
- Domain: Mental-health research
- Primary use: Mental-health-related speech research

### Audio Dataset Information

The official MODMA access page currently describes an audio package
containing:

- 23 Major Depressive Disorder subjects
- 29 Healthy Control subjects
- Age range: 18–52 years
- Audio data
- Demographic information
- Psychological assessment information

### Label Provenance

Labels are derived from the original research participant
classification and psychological assessment framework.

The original study definition must be preserved.

The framework must not silently convert these research labels into
a new clinical diagnosis.

### Access

MODMA requires:

1. Organization-email registration
2. Account activation
3. Dataset selection
4. Downloading the EULA
5. Signing and scanning the EULA
6. Uploading the EULA
7. Approval by dataset administrators

### License / Terms

Dataset access is governed by the MODMA End User License Agreement (EULA).

The EULA must be reviewed before acquisition and use.

### Leakage Considerations

Potential risks:

- Multiple recordings from the same participant
- Participant overlap across files
- Subject-level information leakage

Train/validation/test splitting must be participant-aware.

### Current Status

Candidate — access process verified; EULA approval required.

---

## 3. StudentLife

### Basic Information

- Dataset: StudentLife
- Modality: Behavioral / longitudinal
- Population: Dartmouth college students
- Study duration: Approximately 10 weeks
- Original cohort: 48 students

### Data Types

The dataset contains passive and automatic smartphone sensing data,
ecological momentary assessment data, and pre/post-study mental-health
survey information.

Potential behavioral streams include:

- Activity
- Sleep
- Conversation
- Location-related sensing
- Phone-derived behavioral measurements
- EMA responses

### Label / Outcome Provenance

Mental-health-related outcomes originate from study assessments and
survey instruments.

The exact outcome definition must be documented for every experiment.

These outcomes must not automatically be interpreted as clinical diagnoses.

### Access

The original StudentLife dataset is available through the
Dartmouth StudentLife project.

A smaller test dataset is also documented by the StudentLife R package.

### License / Terms

The raw dataset access terms must be documented from the original
dataset source before redistribution or publication of derived data.

Do not assume that a third-party mirror has the same rights as the
original dataset.

### Leakage Considerations

High importance because the dataset is longitudinal.

Potential risks:

- Same student appearing in multiple splits
- Future observations entering training data
- Temporal leakage
- Repeated measurements from the same participant

Subject-aware and, where appropriate, time-aware splitting must be used.

### Current Status

Candidate — provenance identified; original access terms require verification.

---

## 4. General Data Governance Rules

For every selected dataset:

1. Record the original source.
2. Record the dataset paper or DOI.
3. Record access date.
4. Record access conditions.
5. Record license or EULA information.
6. Preserve the original label definition.
7. Record population and cohort characteristics.
8. Record known limitations.
9. Do not redistribute restricted raw data.
10. Do not expose raw personal or sensitive information in GitHub.
11. Store raw datasets outside Git tracking.
12. Keep dataset manifests and preprocessing metadata under version control.
13. Record file checksums where practical.
14. Document known leakage and evaluation limitations.

---

## 5. Acquisition Status

| Dataset     | Provenance | Access Verified | Terms Verified | Downloaded | Status                     |
| ----------- | ---------- | --------------- | -------------- | ---------- | -------------------------- |
| Dreaddit    | Yes        | Yes             | Pending        | Yes        | Acquired; audits completed |
| MODMA       | Yes        | Yes             | EULA required  | No         | Candidate                  |
| StudentLife | Yes        | Partial         | Pending        | No         | Candidate                  |
