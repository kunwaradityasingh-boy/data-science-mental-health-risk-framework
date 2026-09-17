# Dataset Acquisition Log

## Purpose

This document records the acquisition status of datasets used by the
research framework.

No dataset should be downloaded into the project until its source,
access conditions, and applicable usage terms have been reviewed.

---

## 1. Dreaddit

### Status

Official ACL Anthology attachment acquired and inspected.

Complete raw Dreaddit dataset also acquired and extracted locally.

### Dataset

Dreaddit

### Modality

Text

### Primary Research Task

Stress identification from Reddit-derived text.

### Primary Publication

Turcan, E. and McKeown, K. (2019).

Dreaddit: A Reddit Dataset for Stress Analysis in Social Media.

LOUHI 2019.

DOI:
10.18653/v1/D19-6213

### Sources

Publication:

https://aclanthology.org/D19-6213/

Raw dataset distribution source:

https://www.cs.columbia.edu/~eturcan/data/dreaddit.zip

### Official Attachment Acquired

- Publication verified: Yes
- Official attachment identified: Yes
- Official attachment downloaded: Yes
- Download date: 2026-09-17
- Archive: D19-6213.Attachment.zip
- Archive size: 331,618 bytes
- SHA-256: 05EF5F84B9B22C054D37C02C1C2B86002B6BFF6989FB4380A105EFFFED09F0B5
- Archive contents:
  - Dreaddit_Appendix.pdf

### Raw Dataset Acquisition

- Complete raw dataset downloaded: Yes
- Download date: 2026-09-17
- Archive: dreaddit.zip
- Archive size: 1,348,791 bytes
- SHA-256: 6C7D8859764231CCA47410D6995D8941AC45B08A446A561923FD73B34F3F61CB
- Extracted files:
  - dreaddit-train.csv
  - dreaddit-test.csv
- Extracted location:
  - data/raw/dreaddit/extracted/

### Appendix Inspection

The official appendix contains:

- Full anonymized sample posts
- Full Mechanical Turk annotation guidelines
- Model parameter settings
- Additional error-analysis examples

The annotation instructions describe three possible worker selections:

- Stress
- Not Stress
- Can't Tell

The appendix is supporting research documentation and is not a
substitute for the complete raw labelled dataset.

### Research Label

Stress-related annotation.

The label must be treated as a research annotation and not automatically
as a clinical psychiatric diagnosis.

### License / Redistribution Terms

The raw dataset was acquired from the author-hosted distribution source.

License and redistribution terms still require authoritative verification
before redistribution or publication of the raw data.

### Leakage Checks Required

Before model training:

- Check duplicate text.
- Check duplicate or near-duplicate records.
- Check whether multiple records originate from the same source/user,
  where identifiers permit.
- Prevent inappropriate overlap between training and test data.

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

Raw Dreaddit dataset acquired and extracted successfully.

Dataset quality and leakage audits completed.

License and redistribution terms remain pending authoritative verification.

---

## 2. MODMA

### Status

Pending acquisition.

### Dataset

MODMA

### Modality

Voice

### Access

Official access process requires registration and EULA-related steps.

### Downloaded

No

### Local Copy

No

### Leakage Checks Required

- Participant-aware splitting
- Multiple recordings per participant
- Subject overlap between train/validation/test

### Local Storage

data/raw/modma/

### Git Tracking

Raw dataset files are excluded by .gitignore.

---

## 3. StudentLife

### Status

Pending acquisition.

### Dataset

StudentLife

### Modality

Behavioral / longitudinal

### Population

Dartmouth college students.

### Study Duration

Approximately 10 weeks.

### Original Cohort

48 students.

### Downloaded

No

### Local Copy

No

### Leakage Checks Required

- Subject-level splitting
- Temporal leakage
- Repeated observations
- Future-to-past information leakage
- Outcome leakage

### Local Storage

data/raw/studentlife/

### Git Tracking

Raw dataset files are excluded by .gitignore.

---

## Acquisition Rules

1. Verify the original or authoritative source.
2. Record access date.
3. Record applicable license/EULA.
4. Do not commit raw datasets to Git.
5. Do not publish restricted raw data.
6. Preserve original labels.
7. Record preprocessing steps.
8. Record dataset version or release information when available.
9. Record SHA-256 checksums for downloaded files where practical.
10. Maintain a reproducible acquisition record.

---

## Current Acquisition Status

| Dataset     | Modality   | Downloaded | Extracted | Status                     |
| ----------- | ---------- | ---------- | --------- | -------------------------- |
| Dreaddit    | Text       | Yes        | Yes       | Acquired; audits completed |
| StudentLife | Behavioral | No         | No        | Pending verification       |
| MODMA       | Voice      | No         | No        | Pending verification       |
