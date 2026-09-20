# Phase 10.3 — Error Analysis Summary

Project: Data Science Framework for Early Detection of Mental Health Risk

Framework Version: 0.1.0

This report summarizes observed prediction errors, threshold behavior, class imbalance, and dataset-specific limitations across the evaluated experiments. The analysis is descriptive and does not establish clinical validity.

## 1. EATD Text Error Analysis

Available false-negative records: 15 records.

The EATD text baseline showed high overall accuracy but failed to identify the positive/depression class reliably. The recorded validation result had precision = 0, recall = 0 and F1 = 0 for the positive class, while ROC-AUC was close to chance level. Therefore, the observed accuracy should not be interpreted as evidence of useful discrimination.

The false-negative records are retained as an explicit error-analysis artifact rather than being treated as successful predictions.

## 2. EATD Voice Error Analysis

Available false-negative records: 28 records.

The EATD voice Logistic Regression baseline produced limited positive-class detection. Validation recall and F1 for the positive class were low, and the ROC-AUC result was only modestly above or around chance depending on the evaluation view.

The Random Forest baseline produced no positive predictions under the evaluated default threshold, resulting in zero positive-class recall and F1. This indicates that the classification threshold and class imbalance materially affect the observed operating behavior.

## 3. EATD Text + Voice Multimodal Error Analysis

Available error records: 79 records.

The final EATD text + voice fusion experiment achieved accuracy = 0.7468, precision = 0.2222, recall = 0.1333, F1 = 0.1667, ROC-AUC = 0.5167 and PR-AUC = 0.2152.

The confusion matrix was:

| | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 57 | 7 |
| Actual 1 | 13 | 2 |

The error pattern shows that most positive examples remained undetected at the evaluated operating threshold. Therefore, multimodal fusion did not demonstrate strong discrimination on this dataset under the tested protocol.

## 4. Dreaddit Error Analysis

Dreaddit provides the strongest classification evidence among the evaluated text experiments, but its errors still need to be interpreted in the context of the dataset and evaluation protocol.

The final supplied holdout results for the text-only model were accuracy = 0.7175, F1 = 0.7363, ROC-AUC = 0.8207 and PR-AUC = 0.8412. Adding engineered features improved the final holdout accuracy to 0.7594 and F1 to 0.7731, with ROC-AUC = 0.8422 and PR-AUC = 0.8513.

Existing error-analysis files examine error counts, score bands, subreddit patterns and text-length patterns. These analyses should be interpreted as dataset-specific patterns rather than general clinical relationships.

Summary table records available: 3.
Error-count records available: 12.
Score-band records available: 15.
Subreddit-analysis records available: 30.
Text-length-analysis records available: 15.

## 5. Class Imbalance

Class imbalance is particularly important for the EATD experiments. The EATD text dataset contains substantially more non-depression examples than depression examples. Consequently, a model can obtain relatively high accuracy while producing poor positive-class recall.

This is directly reflected in the EATD text and voice experiments, where positive-class recall and F1 remained very low despite accuracy values around the mid-to-high 70% range.

For this reason, accuracy is not used as the sole indicator of model quality in this framework.

## 6. Threshold Behavior

Threshold-analysis records available across the EATD experiments: 36.

Threshold analysis demonstrates that changing the classification threshold changes the precision/recall trade-off. However, threshold selection alone cannot convert a weakly discriminative model into a validated screening system.

The framework therefore treats threshold selection as an evaluation and operating-point decision rather than as evidence of clinical performance.

## 7. Observed Error Patterns

Existing Dreaddit error-pattern reports contain 66 combined records across the available pattern summaries.

The available analyses cover overall error behavior, model score bands, threshold behavior, subreddit-level patterns and text-length patterns. These are useful for understanding where the model makes mistakes within the dataset.

They should not be interpreted as evidence that a particular subreddit, writing style, or text length causes mental-health risk.

## 8. Cross-Experiment Error Observations

Several consistent observations emerge:

1. **Accuracy alone is insufficient.** EATD experiments illustrate how a model can obtain moderate accuracy while missing most positive examples.

2. **Positive-class recall is a major limitation.** EATD text, voice and multimodal experiments all showed limited detection of the positive class.

3. **Multimodal fusion did not automatically improve discrimination.** Combining text and voice under the tested EATD protocol produced weak ROC-AUC and PR-AUC.

4. **Dataset-specific evaluation matters.** Results from Dreaddit, EATD and StudentLife use different populations, targets and evaluation protocols and therefore should not be interpreted as one common benchmark.

5. **Threshold changes affect operating characteristics but do not establish clinical validity.**

## 9. Error-Analysis Limitations

- EATD has a relatively small participant count and substantial class imbalance.
- EATD text and voice experiments showed weak positive-class discrimination.
- StudentLife behavioral modeling does not constitute true temporal early prediction because exact assessment-time alignment was unavailable.
- Dreaddit contained three exact cross-split text overlaps, which are retained as an evaluation limitation.
- The datasets differ in population, collection protocol, label definition and modality.
- MODMA raw-audio ML benchmarking was not completed because controlled audio access was unavailable.
- Error patterns are dataset-specific and cannot be assumed to generalize to clinical populations.
- The framework is research-only and does not provide a clinical diagnosis or clinical risk determination.

## 10. Research Implications

The error analysis supports a cautious research conclusion: different modalities provide experimentally measurable signals, but the current datasets and baselines do not justify treating the framework as a clinically validated early-detection system.

The strongest experimental evidence in the current project comes from the Dreaddit text experiments, while the EATD experiments demonstrate important limitations in positive-class discrimination. StudentLife provides a behavioral association baseline but lacks the temporal alignment required for a true early-warning evaluation.

These findings justify further work on larger datasets, better temporal alignment, class-imbalance handling, external validation, calibration and prospective evaluation before any consequential use could be considered.

## Phase 10.3 Status

COMPLETE

