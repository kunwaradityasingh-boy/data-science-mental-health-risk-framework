# Dataset Acquisition Log

This document records the acquisition history and provenance of datasets used
in the Data Science Framework for Early Detection of Mental Health Risk.

---

## 1. Dreaddit

### Dataset
Dreaddit

### Modality
Text

### Purpose
Initial NLP and multimodal-framework text baseline.

### Acquisition Status
ACQUIRED

### Acquisition Source
Official research distribution source identified during Phase 3.

### Acquisition Details

Archive:
dreaddit.zip

Train file:
dreaddit-train.csv

Test file:
dreaddit-test.csv

Archive size:
1,348,791 bytes

SHA-256:
6C7D8859764231CCA47410D6995D8941AC45B08A446A561923FD73B34F3F61CB

Train rows:
2,838

Test rows:
715

### Data Handling

Raw dataset files are excluded from Git version control.

The supplied train/test split was preserved.

Post-aware validation was created from the supplied training set.

Cross-split duplicate and exact-text overlap audits were performed.

### Research Label

Dreaddit labels represent a stress-related research classification task.

They are not treated as clinical mental-health diagnoses.

---

# 2. MODMA

## Dataset

Multi-modal Open Dataset for Mental-disorder Analysis (MODMA)

## Relevant Subset

audio_lanzhou_2015

## Modality

Voice / Speech

## Purpose

Development and evaluation of the voice modality of the research framework.

## Acquisition Status

NOT YET ACQUIRED

## Official Source

MODMA Official Dataset Portal:

https://modma.lzu.edu.cn/

## Access Status

Official access procedure identified.

Dataset access requires registration and acceptance/submission of the
applicable End User License Agreement (EULA).

No unofficial copy will be treated as the authoritative research acquisition.

## Expected Published Cohort

Major Depressive Disorder:
23 subjects

Healthy Control:
29 subjects

Total:
52 subjects

Reported age range:
18–52 years

Reported dataset size:
approximately 2.5 GB

These values must be verified against the actual acquired files.

## Acquisition Record

Acquisition date:
NOT YET ACQUIRED

Archive filename:
NOT YET ACQUIRED

Archive size:
NOT YET ACQUIRED

SHA-256:
NOT YET ACQUIRED

Download source:
NOT YET ACQUIRED

EULA/access record:
PENDING

## Post-Acquisition Audit

The following must be verified after acquisition:

- [ ] Archive integrity
- [ ] SHA-256 checksum
- [ ] Number of audio files
- [ ] Number of unique participants
- [ ] Participant ID structure
- [ ] Label distribution
- [ ] Audio file format
- [ ] Sampling rate
- [ ] Bit depth
- [ ] Number of channels
- [ ] Duration distribution
- [ ] Recording/task distribution
- [ ] Missing metadata
- [ ] Corrupt/unreadable files
- [ ] Duplicate files
- [ ] Duplicate recordings
- [ ] Multiple recordings per participant
- [ ] Participant-level train/validation/test split feasibility

## Data Protection

Raw MODMA audio must not be committed to Git.

Raw audio will remain under the project's protected raw-data directory.

Only permitted metadata, preprocessing code, derived research artifacts, and
results will be considered for version control.

## Participant Leakage Prevention

All recordings from a participant must remain within a single evaluation
partition.

File-level random splitting is prohibited.

The planned strategy is participant/group-aware splitting.

## Research Label Boundary

The original MODMA participant labels will be preserved.

The project will not reinterpret the dataset labels as an individual user's
clinical diagnosis.

The resulting model will generate a research signal only.

---

# 3. Acquisition Hash Policy

For every newly acquired research dataset, record:

1. Source URL
2. Acquisition date
3. Archive filename
4. Archive size
5. SHA-256 checksum
6. Extraction status
7. Dataset version, if available
8. Access/EULA status
9. Data license/usage restrictions

---

# 4. Current Acquisition Status

Dreaddit:
ACQUIRED AND AUDITED

MODMA:
RESEARCHED — ACCESS PENDING — NOT YET ACQUIRED

Next action:

Obtain MODMA access through the official dataset procedure and perform the
Phase 7.2 audio acquisition and audit.

---

# 5. Research Reproducibility Note

Dataset acquisition records are maintained separately from raw research data.

Raw datasets may be subject to access restrictions and must not be committed
to the public source repository unless the applicable license explicitly
permits redistribution.

Dataset checksums are used to establish the identity of acquired research
archives without storing the raw archive in Git.