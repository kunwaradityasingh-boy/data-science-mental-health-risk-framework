\# Phase 10.1 — Dataset \& Experiment Summary



\## Project



Data Science Framework for Early Detection of Mental Health Risk



Framework Version: 0.1.0



Phase: 10.1



Status: COMPLETE





\---



\# 1. Research Scope



The project investigates a multimodal data-science framework for

research-oriented detection of mental-health-related risk signals

using:



\- text

\- voice

\- behavioral sensing data



The framework is research-only and is not a clinical diagnostic

system.





\---



\# 2. Dataset Overview



The project uses multiple datasets for modality-specific experiments.



| Dataset | Modality | Primary Task | Role |

|---|---|---|---|

| Dreaddit | Text | Stress-related classification | Text baseline |

| EATD | Voice | Depression-related classification | Voice baseline |

| EATD | Text | Depression-related classification | Text baseline |

| EATD | Text + Voice | Multimodal classification | Dataset-specific fusion |

| StudentLife | Behavior | PHQ-9 score regression | Behavioral baseline |

| MODMA | Voice | Depression-related research | Provenance/access track |



The datasets are not treated as a single homogeneous population.





\---



\# 3. Dreaddit — Text



\## Dataset Role



Dreaddit was used for the primary text-based stress-related

classification experiments.



\## Dataset Characteristics



Training records:



2838



Test records:



715



Engineered numeric features:



109



\## Target



The target represents the dataset's stress-related label.



\## Validation Strategy



The project used post-aware/group-aware validation procedures.



An important dataset limitation was identified:



3 exact cross-split text overlaps were found.



These were retained and reported as an evaluation limitation rather

than silently removed.



\## Baselines



\### Text-only



TF-IDF + Logistic Regression



Validation:



Accuracy: 0.7526

Precision: 0.7395

Recall: 0.8125

F1: 0.7743

ROC-AUC: 0.8432

PR-AUC: 0.8462



\### Engineered-only



Accuracy: 0.7766

Precision: 0.7788

Recall: 0.7993

F1: 0.7890

ROC-AUC: 0.8429

PR-AUC: 0.8408



\### Text + Engineered



Accuracy: 0.7766

Precision: 0.7771

Recall: 0.8026

F1: 0.7896

ROC-AUC: 0.8503

PR-AUC: 0.8478



The supplied holdout evaluation was also retained separately from

validation results.





\---



\# 4. EATD — Voice



\## Dataset Role



EATD was used as the accessible voice dataset for acoustic-feature

experimentation after MODMA raw-audio access was unavailable for

actual modeling.



\## Dataset



Participants:



162



Train:



83



Validation:



79



Audio responses:



3 response types per participant



\## Acoustic Features



The voice pipeline extracted 142 acoustic features including:



\- duration

\- RMS

\- zero-crossing rate

\- spectral centroid

\- spectral bandwidth

\- spectral rolloff

\- spectral contrast

\- chroma

\- MFCC

\- MFCC delta

\- MFCC delta-delta



Participant-level aggregation was performed across usable responses.



One zero-duration processed audio file was excluded from feature

extraction as a data-quality issue.



Raw source data was not modified.





\## Target



The classification target was derived from:



new\_label > 52



The official train/validation split was preserved.



\## Logistic Regression Baseline



Accuracy: 0.772152

Precision: 0.285714

Recall: 0.133333

F1: 0.181818

ROC-AUC: 0.553125

PR-AUC: 0.238817



The result indicates weak discrimination despite relatively high

overall accuracy.



\## Random Forest Baseline



Accuracy: 0.784810

Precision: 0

Recall: 0

F1: 0

ROC-AUC: 0.423958

PR-AUC: 0.209629



The model predicted no positive validation cases at the default

threshold.



Therefore accuracy alone is not treated as evidence of useful

depression-risk discrimination.





\---



\# 5. EATD — Text



\## Dataset Role



Participant-level text classification experiment.



The negative, neutral and positive response texts were concatenated

into a participant-level document.



\## Model



Character-level TF-IDF + Logistic Regression.



Vocabulary size:



4648



\## Validation Results



Accuracy: 0.784810

Precision: 0

Recall: 0

F1: 0

ROC-AUC: 0.509375

PR-AUC: 0.210533



Confusion matrix:



TN = 62

FP = 2

FN = 15

TP = 0



The model therefore showed weak discrimination on the official

validation split.



The high accuracy is primarily associated with the class imbalance

and should not be interpreted independently of recall, F1, ROC-AUC

and PR-AUC.





\---



\# 6. EATD — Text + Voice Fusion



\## Objective



Evaluate whether participant-level text and voice model outputs

could provide a dataset-specific multimodal baseline.



\## Leakage Prevention



The fusion model was trained using train-only out-of-fold

predictions.



The official validation split remained untouched during fusion

training.



Participant keys were split-aware.



\## OOF Results



Text OOF:



ROC-AUC: 0.546200

PR-AUC: 0.301912



Voice OOF:



ROC-AUC: 0.488823

PR-AUC: 0.274443



\## Final Validation Results



Accuracy: 0.746835

Precision: 0.222222

Recall: 0.133333

F1: 0.166667

ROC-AUC: 0.516667

PR-AUC: 0.215210



Confusion matrix:



TN = 57

FP = 7

FN = 13

TP = 2



The current multimodal baseline therefore does not demonstrate

strong discrimination.





\---



\# 7. StudentLife — Behavioral



\## Dataset Role



StudentLife was used to construct a participant-level behavioral

association baseline.



\## Source



The project used the Zenodo-hosted StudentLife RDS archive because

the original Dartmouth server was unavailable during acquisition.



Downloaded archive:



dataset\_rds (1).zip



MD5:



353AE79F157097A57BFF1C004634BA1B



SHA-256:



C7592667708785D9345A4B67454DCDC7BC3A1271023B460B6DC9DECA3E678A73



The documented MD5 matched the expected Zenodo value.





\## PHQ-9



PHQ-9 records:



84



Unique participants:



46



Pre assessments:



46



Post assessments:



38



The project used the PHQ-9 PRE score as the primary behavioral

baseline target.



\## Behavioral Features



Features were derived from:



\- activity

\- phone-lock intervals

\- conversation metadata

\- GPS mobility

\- audio metadata



Raw GPS coordinates were not retained in the final feature table.



\## Important Temporal Limitation



The source PHQ-9 table did not provide the exact assessment timestamp

required for reliable temporal alignment.



Therefore this experiment is explicitly classified as:



retrospective participant-level behavioral association baseline



It is not claimed to be a true temporal early-prediction experiment.





\## Baseline Results



\### Ridge Regression



MAE: 5.3253

RMSE: 6.7992

R²: -2.1558



\### Random Forest Regression



MAE: 3.8488

RMSE: 4.7222

R²: -0.1510



The Random Forest had lower MAE and RMSE in the evaluated

participant-level cross-validation, but its negative mean R² indicates

that the behavioral feature set did not provide strong predictive

explanatory power under this evaluation.





\---



\# 8. MODMA — Voice Provenance Track



MODMA was investigated as the primary controlled-access voice dataset.



Dataset:



audio\_lanzhou\_2015



Participants:



23 MDD

29 healthy controls



Total:



52



The official dataset portal and provenance were documented.



The raw audio could not be used for actual model training because

the safeguarded download/access path was not available during the

project phase.



Therefore:



No MODMA voice ML benchmark is claimed.



MODMA remains a documented provenance, access, ethics and governance

track.





\---



\# 9. Model Registry



The project registry contains six registered models:



1\. dreaddit\_text\_baseline

2\. dreaddit\_combined\_baseline

3\. studentlife\_behavior\_baseline

4\. eatd\_voice\_logistic\_regression

5\. eatd\_text\_baseline

6\. eatd\_text\_voice\_fusion



All models have associated model identifiers, versions, metrics and

limitations documented in:



models/registry/model\_registry.yaml





\---



\# 10. Evaluation Principles



The project uses multiple metrics because class imbalance makes

accuracy alone insufficient.



Primary classification metrics include:



\- Accuracy

\- Precision

\- Recall

\- F1

\- ROC-AUC

\- PR-AUC



Regression metrics include:



\- MAE

\- RMSE

\- R²



Confusion matrices and threshold/error analyses were also performed

for EATD classification experiments.





\---



\# 11. Key Experimental Observations



The experiments demonstrate that:



1\. Text-based Dreaddit experiments produced substantially stronger

&#x20;  research benchmark performance than the current EATD

&#x20;  text/voice baselines.



2\. EATD validation accuracy is not sufficient to characterize model

&#x20;  quality because the positive class is relatively small.



3\. EATD voice and text baselines showed weak discrimination.



4\. The current EATD text + voice fusion baseline did not materially

&#x20;  demonstrate strong discrimination.



5\. StudentLife behavioral features produced a valid baseline but

&#x20;  negative R² values indicate limited predictive performance under

&#x20;  the current evaluation.



6\. Dataset-specific modeling is preferable to unsupported

&#x20;  cross-dataset probability fusion under the current experimental

&#x20;  design.



7\. The framework should therefore be interpreted as a research

&#x20;  engineering framework and experimental methodology, not as a

&#x20;  clinically validated screening system.





\---



\# 12. Reproducibility



Major experiment outputs are stored under:



data/processed/

models/

reports/



Source code is organized under:



src/



API and dashboard applications are under:



apps/



Documentation is under:



docs/



Model registry:



models/registry/model\_registry.yaml





\---



\# 13. Phase 10.1 Status



Phase 10.1 — Dataset \& Experiment Summary



STATUS: COMPLETE

