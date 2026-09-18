\# MODMA Voice Dataset Provenance



\## 1. Purpose



This document records the provenance, access requirements, label interpretation,

privacy considerations, and acquisition protocol for the voice modality of the

"Data Science Framework for Early Detection of Mental Health Risk" research project.



The dataset is intended for research experimentation and model development.

It is not being used to establish a clinical diagnostic system.



\---



\## 2. Dataset Identity



Dataset name:



Multi-modal Open Dataset for Mental-disorder Analysis (MODMA)



Relevant modality:



Voice / Speech



Relevant subset:



audio\_lanzhou\_2015



Official dataset portal:



https://modma.lzu.edu.cn/



Primary dataset publication:



Cai et al., "A multi-modal open dataset for mental-disorder analysis",

Scientific Data.



DOI:



10.1038/s41597-022-01211-x



\---



\## 3. Official Dataset Information



According to the official MODMA dataset information, the relevant audio

cohort contains:



\- 23 subjects with Major Depressive Disorder (MDD)

\- 29 healthy control subjects

\- Total: 52 subjects

\- Reported age range: 18–52 years

\- Dataset size: approximately 2.5 GB



These values describe the published dataset information and will be verified

against the actually obtained files during Phase 7.2.



\---



\## 4. Research Labels



The principal group distinction relevant to the initial voice experiment is:



\- MDD

\- Healthy Control



The published MODMA dataset describes participant assessment and clinical

selection procedures.



For this project, these labels will remain explicitly tied to the original

dataset definition.



The model output in this project will NOT be interpreted as a clinical

diagnosis.



\---



\## 5. Voice Data Collection



The MODMA documentation describes speech data collected using speech-related

tasks such as:



\- Interview

\- Reading

\- Picture description



Published research using the MODMA voice database reports multiple voice

samples per participant.



The exact number, duration, format, sampling rate, and task distribution of

the files actually acquired will be verified during the Phase 7.2 audio audit.



No assumptions will be made about the downloaded files until the actual

directory and audio metadata have been inspected.



\---



\## 6. Participant-Level Data Policy



Participant identity is treated as a grouping variable.



The same participant must never appear across training and validation/test

partitions.



The planned evaluation strategy is therefore participant-aware:



Participant ID

&#x20;   ->

Group-aware split

&#x20;   ->

Training participants

Validation participants

Test participants



A random file-level split is prohibited because multiple recordings from the

same participant could otherwise appear in different partitions and cause

subject leakage.



\---



\## 7. Data Acquisition Status



Current status:



NOT YET ACQUIRED



The dataset will only be obtained through the official MODMA access process.



No unofficial mirror, random GitHub repository, or unverified third-party

copy will be treated as the authoritative acquisition source.



After acquisition, the following information will be recorded:



\- Acquisition date

\- Official download source

\- Dataset/subset name

\- Archive filename

\- File size

\- SHA-256 checksum

\- Extraction status

\- Number of audio files

\- Participant count

\- Available metadata

\- Label distribution

\- Audio formats

\- Sampling rates

\- Corrupt/unreadable files

\- Access/EULA restrictions



\---



\## 8. Access and EULA



The official MODMA access procedure requires registration and acceptance of

the applicable End User License Agreement (EULA) before access to controlled

dataset material.



The project will comply with the dataset's applicable access and usage

conditions.



Raw MODMA audio will NOT be committed to Git.



Raw research data will remain outside version-controlled source files and

will be protected using the project's data exclusion rules.



\---



\## 9. Privacy Considerations



Voice recordings are potentially identifying biometric information.



Therefore:



\- Raw voice recordings must be treated as sensitive research data.

\- Raw audio must not be uploaded to a public Git repository.

\- Direct identifiers must not be included in model features.

\- Participant IDs must be treated as pseudonymous research identifiers.

\- Dataset access conditions must be respected.

\- Model artifacts and reports must not expose unnecessary participant-level

&#x20; information.

\- Any future deployment requires a separate privacy, security, consent, and

&#x20; regulatory assessment.



\---



\## 10. Planned Voice Processing Pipeline



The planned research pipeline is:



Raw Audio

&#x20;   ->

File Integrity Check

&#x20;   ->

Audio Metadata Audit

&#x20;   ->

Participant/Label Mapping

&#x20;   ->

Participant-Aware Split

&#x20;   ->

Audio Quality Validation

&#x20;   ->

Preprocessing

&#x20;   ->

Acoustic Feature Extraction

&#x20;   ->

MFCC / Additional Features

&#x20;   ->

Voice Baseline Model

&#x20;   ->

Validation

&#x20;   ->

Final Holdout Evaluation

&#x20;   ->

Error Analysis

&#x20;   ->

Model Artifact

&#x20;   ->

API Integration



\---



\## 11. Planned Acoustic Features



The initial voice experiment will investigate conventional acoustic features,

including:



\- MFCC

\- Delta MFCC

\- Delta-delta MFCC

\- Spectral characteristics

\- Zero-crossing rate

\- Chroma-related features where appropriate

\- Energy-related features

\- Fundamental-frequency-related features where reliably extractable



Feature selection will be determined after the Phase 7.2 data and quality

audit.



No feature will be included solely because it is commonly used in another

paper.



\---



\## 12. Model Development Strategy



The first voice model will be a reproducible research baseline.



The exact algorithm will be selected after:



1\. Dataset audit

2\. Feature extraction

3\. Class distribution analysis

4\. Participant-level split design



Candidate baseline models may include:



\- Logistic Regression

\- Random Forest

\- Support Vector Machine



Model selection will be based on documented validation results rather than

an assumed best-performing algorithm.



\---



\## 13. Evaluation Strategy



Evaluation must be participant-aware.



The following metrics will be considered:



\- Accuracy

\- Precision

\- Recall

\- F1

\- ROC-AUC

\- PR-AUC

\- Confusion matrix

\- Class-wise performance



Because the expected cohort is small, confidence intervals and the limitations

of the evaluation design will be documented where feasible.



The final test/holdout data will not be used for model selection or threshold

tuning.



\---



\## 14. Known Dataset Limitations



The expected limitations include:



\- Small participant count

\- Multiple recordings per participant

\- Potential class imbalance

\- Potential recording/task differences

\- Limited demographic diversity

\- Dataset-specific collection conditions

\- Controlled research setting

\- Limited external generalizability



These limitations prevent interpreting a successful benchmark result as proof

of clinical effectiveness.



\---



\## 15. Safety Boundary



This project is a research prototype for mental-health risk-signal screening

and referral-support research.



The voice model must NOT:



\- Diagnose depression

\- Diagnose another mental-health disorder

\- Replace a qualified professional

\- Make emergency decisions autonomously

\- Make academic disciplinary decisions

\- Make employment decisions

\- Make insurance decisions

\- Make legal decisions

\- Automatically prescribe or recommend treatment



A model-generated score is a research signal, not automatically a clinical

probability.



\---



\## 16. Phase 7.1 Completion Criteria



Phase 7.1 will be considered complete when:



\- \[x] MODMA identified as the voice dataset candidate

\- \[x] Official dataset source identified

\- \[x] Primary publication identified

\- \[x] Initial cohort information documented

\- \[x] Access/EULA requirement documented

\- \[x] Participant-aware split requirement documented

\- \[x] Privacy requirements documented

\- \[x] Planned acquisition metadata documented

\- \[ ] Dataset actually acquired

\- \[ ] Dataset hash recorded

\- \[ ] Audio files audited

\- \[ ] Participant metadata audited



The remaining unchecked items belong to Phase 7.2 and later phases.



\---



\## 17. References



1\. MODMA Official Dataset Portal:

&#x20;  https://modma.lzu.edu.cn/



2\. MODMA Data Access:

&#x20;  https://modma.lzu.edu.cn/data/application/



3\. Cai et al., "A multi-modal open dataset for mental-disorder analysis",

&#x20;  Scientific Data:

&#x20;  https://www.nature.com/articles/s41597-022-01211-x



4\. MODMA Publications:

&#x20;  https://modma.lzu.edu.cn/data/publications/



\---



\## 18. Research Status



Current phase:



Phase 7.1 — Dataset Research and Provenance



Status:



DOCUMENTED — DATASET NOT YET ACQUIRED



Next phase:



Phase 7.2 — MODMA Audio Acquisition and Dataset Audit

