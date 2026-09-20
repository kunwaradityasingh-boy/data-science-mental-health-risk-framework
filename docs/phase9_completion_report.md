\# Phase 9 Completion Report

\## Production-Oriented Integration, Governance \& API Validation



Project:

Data Science Framework for Early Detection of Mental Health Risk



Framework Version:

0.1.0



Status:

COMPLETE



Date:

2026-09-20





\---



\## 1. Objective



Phase 9 integrated the independently developed text, voice and

behavior research components into a unified research-oriented

framework.



The phase focused on:



\- model registry and versioning

\- unified prediction schema

\- multimodal integration configuration

\- integration service

\- risk policy integration

\- framework prediction service

\- FastAPI integration

\- automated API validation

\- full regression testing





\---



\## 2. Completed Components



\### 2.1 Model Registry



File:



models/registry/model\_registry.yaml



Registered models:



1\. dreaddit\_text\_baseline

2\. dreaddit\_combined\_baseline

3\. studentlife\_behavior\_baseline

4\. eatd\_voice\_logistic\_regression

5\. eatd\_text\_baseline

6\. eatd\_text\_voice\_fusion



Status:



COMPLETE





\### 2.2 Unified Prediction Schema



Files:



src/data/prediction/unified\_prediction\_schema.py

src/data/prediction/prediction\_schema.yaml



The schema standardizes:



\- subject identifier

\- modality

\- model identifier

\- model version

\- score

\- prediction

\- data quality

\- prediction status

\- timestamp

\- risk signal

\- human review requirement

\- framework version



Status:



COMPLETE





\### 2.3 Integration Configuration



File:



configs/integration/multimodal\_integration.yaml



Configured modalities:



\- text

\- voice

\- behavior



Cross-dataset fusion:



Disabled



Reason:



The datasets use different populations, targets, labeling procedures

and validation protocols. Therefore, their scores are not treated as

directly interchangeable for a cross-dataset fused clinical-style

prediction.



EATD text + voice fusion remains available as a dataset-specific

research experiment.





\### 2.4 Multimodal Integration Service



File:



src/models/multimodal\_integration\_service.py



The service verifies:



\- configured modalities

\- registered models

\- model artifacts

\- fusion artifact

\- integration health

\- unknown prediction policy



Validation result:



PHASE 9.3B PASSED





\### 2.5 Risk Policy



File:



src/models/risk\_policy\_service.py



Configured research thresholds:



\- LOW: 0.00–0.39

\- MODERATE: 0.40–0.69

\- HIGH: 0.70–1.00

\- UNKNOWN: invalid, missing or unavailable score



Human review:



\- moderate: required/recommended according to research policy

\- high: required

\- unknown: human review required by framework-level handling



The system is research-only and does not produce a clinical diagnosis.





\### 2.6 Framework Prediction Service



File:



src/models/framework\_prediction\_service.py



The service combines available modality-level signals into a unified

framework response.



It does not perform unsupported cross-dataset probability fusion.



It supports:



\- text

\- voice

\- behavior

\- missing-modality handling

\- UNKNOWN handling

\- human-review indication

\- model-version reporting





\### 2.7 FastAPI Integration



File:



apps/api/framework\_api.py



Endpoints:



GET /

GET /health

POST /framework/predict



The API returns structured JSON based on the unified prediction schema.



Research-only status is explicitly exposed by the API.





\---



\## 3. API Validation



\### Root Endpoint



GET /



Verified:



\- service name

\- framework version

\- research-only status

\- clinical diagnosis disabled





\### Health Endpoint



GET /health



Verified:



\- status = healthy

\- framework version = 0.1.0

\- text modality configured

\- voice modality configured

\- behavior modality configured

\- risk policy loaded

\- cross-dataset fusion disabled





\### Framework Prediction



POST /framework/predict



Test input:



subject\_id = demo\_subject\_001

text\_score = 0.62

voice\_score = 0.31



Observed result:



overall\_status = success

risk\_signal = moderate

human\_review\_required = true



Available modalities:



text

voice





\### Unknown Prediction



Test input:



subject\_id = demo\_subject\_002



with no modality scores.



Observed result:



overall\_status = unknown

risk\_signal = unknown

human\_review\_required = true





\---



\## 4. Automated API Tests



Test file:



tests/api/test\_framework\_api.py



Tests:



1\. root endpoint

2\. health endpoint

3\. text + voice framework prediction

4\. unknown prediction

5\. invalid score validation

6\. missing subject ID validation



Result:



6 passed





\---



\## 5. Full Regression Test



Command:



pytest -v



Result:



49 passed

2 warnings

3.55 seconds



No test failures occurred.



The warnings were dependency deprecation warnings and did not

cause test failures.





\---



\## 6. Governance Boundaries



The framework is research-only.



It must not be used for:



\- clinical diagnosis

\- emergency triage

\- automated treatment decisions

\- academic punishment

\- employment decisions

\- insurance decisions

\- legal decisions

\- automated denial of services



The generated signal is a model-generated research signal and should

not automatically be interpreted as a clinical probability.



Human review is required before consequential interpretation.





\---



\## 7. Data and Privacy Principles



The framework uses:



\- pseudonymous subject identifiers

\- explicit modality definitions

\- data-quality checks

\- model version tracking

\- no raw text logging in the integration configuration

\- no raw audio logging in the integration configuration

\- research-only dataset usage





\---



\## 8. Important Research Limitations



The current framework does not establish clinical validity.



Important limitations include:



1\. The datasets originate from different populations and study

&#x20;  protocols.



2\. Cross-dataset fusion is intentionally disabled.



3\. EATD text and voice standalone baselines showed weak

&#x20;  discrimination on the official validation split.



4\. EATD text + voice fusion also showed weak discrimination and

&#x20;  should not be interpreted as a clinically useful predictor.



5\. StudentLife behavioral modeling is a retrospective

&#x20;  participant-level association baseline and is not a true

&#x20;  temporal early-prediction model because the available PHQ-9

&#x20;  source table did not provide the exact assessment timestamp

&#x20;  required for temporal alignment.



6\. MODMA raw audio could not be used for actual voice modeling

&#x20;  because safeguarded raw dataset access/download was unavailable

&#x20;  during the project phase.



7\. The framework therefore demonstrates an engineering and research

&#x20;  methodology rather than a clinically validated screening system.





\---



\## 9. Final Phase 9 Status



Phase 9 is COMPLETE.



Completed steps:



9.1  Model Registry \& Versioning              COMPLETE

9.2  Unified Prediction Schema                COMPLETE

9.3  Integration Configuration                COMPLETE

9.3B Integration Service                      COMPLETE

9.4  Risk Policy Integration                  COMPLETE

9.5  Framework Prediction Service             COMPLETE

9.6  FastAPI Integration                      COMPLETE

9.7  API Automated Tests                      COMPLETE

9.8  Full Regression Test                     COMPLETE





Final validation:



49/49 tests passed.





\---



\## 10. Conclusion



Phase 9 successfully transformed the independent research

components into a versioned, governed and test-validated

framework-level architecture.



The framework can expose structured research-only modality signals

through FastAPI while preserving:



\- model provenance

\- modality identity

\- model versioning

\- data-quality status

\- risk policy

\- human-review requirements

\- UNKNOWN handling

\- research-use boundaries



Phase 9 is therefore formally closed.

