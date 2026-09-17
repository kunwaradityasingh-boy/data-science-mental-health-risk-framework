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

Complete raw dataset downloaded: Yes

Raw dataset source:

https://www.cs.columbia.edu/~eturcan/data/dreaddit.zip

Download date: 2026-09-17

Downloaded archive:

dreaddit.zip

Archive size:

1,348,791 bytes

SHA-256:

6C7D8859764231CCA47410D6995D8941AC45B08A446A561923FD73B34F3F61CB

Extracted files:

- dreaddit-train.csv
- dreaddit-test.csv

Extracted location:

data/raw/dreaddit/extracted/

Dataset version/release:

Not explicitly specified by the distribution source.

## Storage

Raw data location:

data/raw/dreaddit/

Downloaded archive:

data/raw/dreaddit/dreaddit.zip

Extracted files:

data/raw/dreaddit/extracted/dreaddit-train.csv

data/raw/dreaddit/extracted/dreaddit-test.csv

Official research appendix:

data/raw/dreaddit/inspection/Dreaddit_Appendix.pdf

Raw data must not be committed to Git.

## Leakage Checks

Before model training:

- Check duplicate records.
- Check duplicate or near-duplicate text.
- Check source/user overlap where identifiers permit.
- Keep related records from leaking across evaluation splits.

## Status

Official research appendix acquired and inspected.

Complete raw Dreaddit dataset has been acquired and extracted locally.

License and redistribution terms should still be verified from the
authoritative distribution source before any redistribution or publication
of the raw data.
