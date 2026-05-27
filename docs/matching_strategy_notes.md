# Matching Strategy Notes

This document explains the matching strategies used in the benchmark.

## True-pair matching

True-pair matching uses the known synthetic ground-truth pair identity.

This is not meant to be a practical method. It is an oracle upper bound used to check whether the retrieval model can learn when the pair labels are correct.

## Random matching

Random matching shuffles modality-B samples before pairing them with modality-A samples.

This is a lower bound. If a learned model performs close to random matching, the pair-construction strategy is not providing useful cross-modal supervision.

## Metadata-similarity matching

Metadata-similarity matching compares samples using metadata features such as:

- age
- severity score
- binary condition indicators
- sex

The nearest metadata match is selected as the pseudo-pair.

This strategy is simple and transparent, but it can fail when metadata is noisy or weakly related to the true cross-modal structure.

## Propensity-weighted matching

The propensity-weighted strategy learns a metadata-based score for whether two samples are likely to match.

In this repository, the propensity model is intentionally simple. It uses metadata-difference features such as:

- age difference
- severity-score difference
- condition matches
- sex match

The purpose is not to build the strongest possible matcher. The purpose is to test whether a learned metadata-based matching score can improve over raw metadata similarity.

## Connection to propensity-score alignment

The idea is inspired by propensity-style matching and recent work on propensity-score alignment for unpaired multimodal data.

In the uploaded paper, propensity scores are used as a common space for matching samples across modalities when direct pairs are unavailable. This repository uses a smaller controlled benchmark to test a related idea: whether learned metadata-based pair scores can improve pseudo-pair construction for retrieval learning.

## Practical lesson

Pairing strategy should be evaluated before blaming the encoder or loss function.

If the pairs are weak, a stronger model may simply learn the wrong alignment more confidently.