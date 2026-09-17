## 1. Dreaddit

### Status

Official ACL Anthology attachment acquired and inspected.

The acquired attachment contains the research appendix PDF,
not the complete raw labelled dataset.

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

### Source

ACL Anthology:

https://aclanthology.org/D19-6213/

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

- Complete raw dataset downloaded: No
- Complete labelled dataset available in the acquired archive: No
- Raw dataset source: Requires separate verification
- Raw dataset license/redistribution terms: Pending verification

### Appendix Inspection

The official appendix contains:

- Full anonymized sample posts
- Full Mechanical Turk annotation guidelines
- Model parameter settings
- Additional error-analysis examples

The appendix confirms that workers could assign:

- Stress
- Not Stress
- Can't Tell

The appendix is supporting research documentation and is not a
substitute for the complete raw dataset.

### Research Label

Stress-related annotation.

The label must be treated as a research annotation and not automatically
as a clinical psychiatric diagnosis.

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

### Git Tracking

Raw dataset files and archives are excluded by .gitignore.
Dataset documentation is tracked by Git.

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

## Current Acquisition Queue

| Priority | Dataset     | Modality   | Status                                          |
| -------: | ----------- | ---------- | ----------------------------------------------- |
|        1 | Dreaddit    | Text       | Official appendix acquired; raw dataset pending |
|        2 | StudentLife | Behavioral | Pending verification                            |
|        3 | MODMA       | Voice      | Pending verification                            |
