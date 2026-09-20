# Multimodal Fusion Design and Compatibility Audit

## 1. Objective

The purpose of this document is to determine whether the currently
available text, behavioral, and voice datasets can be directly combined
into a participant-level multimodal machine-learning dataset.

The audit is performed before implementing multimodal fusion in order to
prevent participant mismatch, target mismatch, and data leakage.

---

## 2. Available Modalities

| Modality | Dataset     | Unit              | Current Target                     |
| -------- | ----------- | ----------------- | ---------------------------------- |
| Text     | Dreaddit    | Reddit post       | Stress-related label               |
| Behavior | StudentLife | Student           | PHQ-9 assessment score             |
| Voice    | MODMA       | Participant/audio | MDD/control-related research label |

---

## 3. Participant Compatibility

### Text vs Behavior

Dreaddit contains Reddit-based records and does not provide the same
participant identifiers as StudentLife.

Therefore, a participant-level join between Dreaddit and StudentLife
cannot be established.

Status:

**NOT DIRECTLY COMPATIBLE**

---

### Text vs Voice

Dreaddit and MODMA do not share participant identifiers or a common
participant cohort.

Therefore, participant-level text-voice fusion cannot currently be
performed.

Status:

**NOT DIRECTLY COMPATIBLE**

---

### Behavior vs Voice

StudentLife and MODMA represent different participant cohorts and do not
provide a shared participant identifier.

Therefore, direct participant-level behavioral-voice fusion cannot
currently be established.

Status:

**NOT DIRECTLY COMPATIBLE**

---

## 4. Target Compatibility

The current datasets do not use an identical target definition.

Dreaddit provides a stress-related research label.

StudentLife provides PHQ-9 assessment-derived scores.

MODMA contains a depression/MDD-related research cohort.

These targets should not be treated as interchangeable clinical labels.

Therefore, direct supervised fusion using a single shared target is not
currently justified.

---

## 5. Leakage Risk

Artificially joining records from different datasets based only on
feature similarity, row order, demographics, or aggregate statistics
would create invalid participant associations.

Such a procedure is prohibited for the current research implementation.

---

## 6. Current Fusion Decision

Direct participant-level multimodal fusion is currently:

**BLOCKED**

because:

1. There is no shared participant identifier.
2. The datasets represent different cohorts.
3. The target definitions are different.
4. Temporal alignment cannot be established across datasets.

---

## 7. Recommended Architecture

The project will therefore use a modular multimodal framework.

Each modality will maintain an independent processing and modelling
pipeline:

    Text
      |
      v
    Text Model
      |
      v
    Text Risk Signal


    Behavior
      |
      v
    Behavioral Model
      |
      v
    Behavioral Risk Signal


    Voice
      |
      v
    Voice Model
      |
      v
    Voice Risk Signal

The outputs can only be combined into a common multimodal risk layer
when the input data provide a justified common cohort and compatible
target definition.

---

## 8. Future Fusion Requirement

A future multimodal fusion dataset must contain, at minimum:

- common pseudonymous participant ID
- common target definition
- modality-specific timestamps
- consent metadata
- participant-level split strategy
- temporal leakage controls
- modality quality indicators
- missing-modality handling
- provenance for every modality

---

## 9. Research Position

The current implementation prioritizes methodological validity over
artificially creating a multimodal result.

The absence of direct participant-level compatibility is documented as a
dataset limitation rather than being hidden through synthetic joins.

The resulting system is therefore treated as a modular data-science
framework with independently validated modality pipelines.

---

## 10. Phase 8 Status

Phase 8.1 — Multimodal compatibility audit:

**COMPLETE**

Phase 8.2 — Common risk-output schema:

**NEXT**
