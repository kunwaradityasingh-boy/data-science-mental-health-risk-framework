\# Phase 4 — MODMA Voice Dataset

\## Research, Provenance, Access and Governance Completion Report



\### Project

Data Science Framework for Early Detection of Mental Health Risk



\### Phase

Phase 4 — MODMA Voice Dataset



\### Status

COMPLETE — Research / Provenance / Access Track



\---



\## 1. Objective



The objective of Phase 4 was to evaluate the MODMA speech dataset as a

candidate voice modality for the mental-health risk research framework.



The phase covered:



1\. Dataset identification

2\. Dataset provenance

3\. Population and label documentation

4\. Access and licensing review

5\. Ethical and privacy review

6\. Dataset integrity requirements

7\. Reproducibility requirements

8\. Voice-feature pipeline planning

9\. Risk and limitation documentation

10\. Final dataset decision



\---



\## 2. Dataset Identity



Dataset:



MODMA — Multi-modal Open Dataset for Mental-disorder Analysis



Voice dataset:



audio\_lanzhou\_2015



Dataset provider:



Lanzhou University / MODMA



Repository:



MODMA Dataset



UK Data Service / ReShare collection:



Multi-modal Open Dataset for Mental-disorder Analysis,

Experimental Data 2014–2016



\---



\## 3. Dataset Population



According to the official MODMA dataset documentation:



Total voice participants:



52



Major Depressive Disorder:



23



Healthy Controls:



29



Age range:



18–52 years



The dataset contains demographic information and psychological

assessment information associated with the audio experiment.



\---



\## 4. Audio Dataset Characteristics



Dataset package:



audio\_lanzhou\_2015



Approximate package size:



2.5 GB



Official MODMA MD5:



5550552006ec4ae5a89ecb86b7a147ac



The audio experiment contains spoken-language recordings collected

through tasks including interviewing, reading and picture description.



The dataset contains speech recordings rather than text-only records.



\---



\## 5. Dataset Provenance



Primary dataset publication:



Cai et al. (2020)



"A multi-modal open dataset for mental-disorder analysis."



Additional audio-related publication:



Liu, Z., Wang, D., Zhang, L., \& Hu, B. (2020).



"A Novel Decision Tree for Depression Recognition in Speech."



The dataset has a documented provenance through the MODMA project and

the UK Data Service / ReShare archive.



\---



\## 6. Access and Licensing



MODMA provides controlled access to the audio dataset.



The official MODMA workflow requires:



1\. Account registration

2\. Organization email verification

3\. Dataset selection

4\. End User Licence Agreement (EULA)

5\. EULA submission

6\. Administrator approval

7\. Download through the authorized access interface



The UK Data Service records the audio package as safeguarded data

available to registered users under the applicable End User Licence.



Therefore, the raw voice recordings must not be treated as unrestricted

public files.



\---



\## 7. Ethical and Privacy Assessment



The MODMA documentation states that written informed consent was

obtained from participants.



The study design and consent procedures were approved by the local

Ethics Committee for Biomedical Research at Lanzhou University Second

Hospital.



Voice recordings are potentially identifiable biometric information.



Therefore:



\- Raw audio must not be committed to Git.

\- Raw audio must not be redistributed.

\- Raw audio must remain in an authorized local or controlled storage

&#x20; location.

\- Derived features should be treated as research data.

\- Participant identifiers must remain pseudonymous.

\- Access restrictions must be respected.



\---



\## 8. Dataset Integrity Requirements



Before any future modeling using MODMA audio, the following checks

must be performed:



1\. Verify the downloaded archive hash.

2\. Verify archive integrity.

3\. Enumerate all participant records.

4\. Verify participant counts.

5\. Verify class counts.

6\. Verify audio file formats.

7\. Verify sample rates.

8\. Verify channel configuration.

9\. Detect corrupted files.

10\. Detect zero-duration files.

11\. Detect duplicate recordings.

12\. Verify participant-level label consistency.

13\. Preserve participant-level separation during evaluation.



No raw MODMA audio should be modified in place.



\---



\## 9. Planned Voice Feature Pipeline



If MODMA audio becomes available for authorized research use, the

planned baseline pipeline is:



Raw WAV

&#x20;   ↓

Audio quality validation

&#x20;   ↓

Resampling / channel normalization

&#x20;   ↓

Silence and invalid-segment handling

&#x20;   ↓

Acoustic feature extraction

&#x20;   ↓

Participant-level aggregation

&#x20;   ↓

Participant-level train/validation split

&#x20;   ↓

Baseline classifier

&#x20;   ↓

ROC-AUC / PR-AUC / Precision / Recall / F1

&#x20;   ↓

Threshold analysis

&#x20;   ↓

Error analysis

&#x20;   ↓

Calibration / model interpretation



Potential acoustic features include:



\- Duration

\- RMS energy

\- Zero-crossing rate

\- Spectral centroid

\- Spectral bandwidth

\- Spectral rolloff

\- Spectral contrast

\- Chroma

\- MFCC

\- Delta MFCC

\- Delta-delta MFCC



\---



\## 10. Leakage Prevention



The unit of evaluation must be the participant rather than an individual

audio segment.



Audio recordings from the same participant must never be distributed

across training and validation/test sets.



If multiple recordings exist for one participant, all recordings must

remain inside the same split.



Feature aggregation must occur without using validation/test information

during model training.



\---



\## 11. Current Acquisition Status



The MODMA voice dataset was investigated through the official MODMA

source and the UK Data Service archive.



The dataset is documented and provenance-verified.



However, raw audio modeling is not claimed as completed because access

to the safeguarded audio package is controlled.



This is an access constraint, not a modeling result.



\---



\## 12. Project Decision



MODMA remains a documented candidate voice dataset in the research

framework.



For the actual implemented voice-machine-learning pipeline, the project

uses EATD-Corpus.



The EATD voice pipeline is documented separately under Phase 6.



This separation prevents the project from claiming experiments that were

not actually performed on MODMA audio.



\---



\## 13. Phase 4 Deliverables



Completed:



\[PASS] MODMA dataset identification



\[PASS] Audio dataset identification



\[PASS] Participant population documentation



\[PASS] Class distribution documentation



\[PASS] Dataset provenance



\[PASS] Primary publication identification



\[PASS] Audio-related publication identification



\[PASS] Access procedure documentation



\[PASS] Licensing / EULA review



\[PASS] Ethical and privacy review



\[PASS] Raw-data handling policy



\[PASS] Data integrity checklist



\[PASS] Voice feature pipeline specification



\[PASS] Leakage-prevention protocol



\[PASS] Acquisition limitation documented



\[PASS] Alternative implemented voice dataset identified



\---



\## 14. Final Phase Status



PHASE 4 STATUS:



100% COMPLETE



Completion type:



MODMA Voice Research, Provenance, Access and Governance Track



Important limitation:



A MODMA voice ML benchmark is not claimed because the safeguarded raw

audio was not successfully obtained for processing.



Actual implemented voice modeling is covered by:



PHASE 6 — EATD Voice Modality



\---



\## 15. Research Integrity Statement



This phase intentionally distinguishes between:



1\. Dataset research

2\. Dataset access

3\. Dataset acquisition

4\. Dataset processing

5\. Model training

6\. Model evaluation



Only activities actually performed are marked as completed.



No performance result is reported for MODMA without running the model

on the authorized MODMA audio data.

