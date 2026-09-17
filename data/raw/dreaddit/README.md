# Dreaddit Raw Dataset

## Dataset

Dreaddit: A Reddit Dataset for Stress Analysis in Social Media

## Authors

Elsbeth Turcan and Kathy McKeown

## Publication

LOUHI 2019

## DOI

10.18653/v1/D19-6213

## Authoritative Publication Source

https://aclanthology.org/D19-6213/

## Dataset Description

The published paper describes approximately 190K Reddit posts from
five categories of Reddit communities.

Approximately 3.5K labelled segments were taken from 3K posts.

The labelled segments were annotated using Amazon Mechanical Turk.

## Research Interpretation

The labels represent a stress-identification research task.

They must not automatically be interpreted as clinical psychiatric
diagnoses.

## Official Attachment Acquisition

The official ACL Anthology attachment was downloaded and inspected.

Archive:

D19-6213.Attachment.zip

The archive contains:

Dreaddit_Appendix.pdf

The appendix contains sample posts, annotation instructions,
parameter settings, and error-analysis examples.

It does not contain the complete raw labelled dataset.

## Acquisition Status

Publication verified: Yes

Official attachment acquired: Yes

Raw dataset acquisition: Pending

Local raw dataset download: No

Download date: 2026-09-17

Dataset version/release: TBD

SHA-256 of official attachment:

05EF5F84B9B22C054D37C02C1C2B86002B6BFF6989FB4380A105EFFFED09F0B5

## Storage

Raw data location:

data/raw/dreaddit/

Official attachment:

data/raw/dreaddit/D19-6213.Attachment.zip

Inspection copy:

data/raw/dreaddit/inspection/Dreaddit_Appendix.pdf

Raw data must not be committed to Git.

## Leakage Checks

Before model training:

- Check duplicate records.
- Check duplicate or near-duplicate text.
- Check source/user overlap where identifiers permit.
- Keep related records from leaking across evaluation splits.

## Status

Official research appendix acquired.

Complete raw Dreaddit dataset is still pending authoritative
acquisition and terms verification.
