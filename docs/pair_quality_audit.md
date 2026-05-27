# Pair Quality Audit

This repository treats pair construction as a measurable part of the multimodal retrieval pipeline.

The central idea is simple:

> A retrieval model can only learn useful cross-modal alignment if the training pairs contain useful signal.

For this reason, the benchmark reports both pair-quality metrics and retrieval metrics.

## Why pair quality matters

In weakly paired multimodal learning, exact cross-modal pairs may be unavailable. A common workaround is to create pseudo-pairs using metadata, similarity scores, or matching models.

However, pseudo-pairs can be wrong. If many constructed pairs are incorrect, the contrastive model receives weak or misleading supervision.

This benchmark measures how different pairing strategies affect retrieval learning.

## Pairing strategies

| Strategy | Purpose |
|---|---|
| True pair | Oracle upper bound |
| Random | Failure baseline |
| Metadata similarity | Simple metadata-based pseudo-pairing |
| Propensity weighted | Learned metadata-based pseudo-pairing |

The true-pair strategy shows what is possible when pair supervision is correct. The random strategy shows what happens when pair supervision is arbitrary.

Metadata similarity and propensity weighting are the realistic pseudo-pairing methods.

## Pair-quality metrics

Two pair-quality metrics are reported:

| Metric | Meaning |
|---|---|
| Pair true precision | Fraction of constructed pairs that are exact true pairs |
| Pair group precision | Fraction of constructed pairs that share the same latent group |

Exact true-pair precision is strict. Same-group precision is softer and measures whether a pseudo-pair at least comes from a related semantic group.

## Main observation

The benchmark shows that pseudo-pairing can still provide retrieval signal even when exact true-pair precision is low.

However, when metadata becomes too noisy, both metadata similarity and propensity weighting approach the random baseline.