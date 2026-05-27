# Pair Construction Benchmark for Multimodal Retrieval

## Executive Snapshot

This experiment studies how the quality of constructed multimodal pairs affects retrieval learning.

Instead of comparing model architectures or loss functions, this repository focuses on the pairing step itself:

> If the training pairs are weak, noisy, or only approximately matched, how much retrieval signal can still be learned?

The benchmark compares four pairing strategies across clean, moderate-noise, and high-noise synthetic multimodal settings.

## Benchmark Role of Each Pairing Strategy

| Strategy | Role in the benchmark |
|---|---|
| True pair | Oracle upper bound: what happens when pair supervision is correct |
| Random | Lower bound: what happens when pair supervision is arbitrary |
| Metadata similarity | Simple heuristic baseline using metadata closeness |
| Propensity weighted | Learned metadata-based matching score for pseudo-pair selection |

This setup separates two questions:

1. Can the retrieval model learn when the pairs are reliable?
2. Can pseudo-pair construction recover useful signal when true pairs are unavailable?

## Data Conditions

The synthetic benchmark contains two modalities, metadata, group labels, and known true-pair identities.

| Condition | Meaning |
|---|---|
| Clean | Metadata and modality features are relatively aligned |
| Moderate noise | Metadata and modality features become less reliable |
| High noise | Metadata becomes weak and pair ambiguity becomes severe |

Two sample sizes were used:

| Total samples | Held-out retrieval pool |
|---:|---:|
| 8000 | 1600 |
| 24000 | 4800 |

All training runs used 50 epochs.

## What Was Measured

The benchmark tracks both pair-construction quality and retrieval quality.

| Metric group | Metrics |
|---|---|
| Pair quality | exact true-pair precision, same-group precision |
| Retrieval | Recall@1, Recall@5, Recall@10, Recall@50 |
| Relative retrieval gain | Lift@K over random retrieval |
| Representation quality | positive-pair cosine similarity |
| Optimization | training loss |

This is important because high retrieval performance without pair-quality analysis can be misleading. A pseudo-pairing method may look useful only because the retrieval pool is easy, or it may have low exact-pair precision but still recover group-level signal.

## 8000-Sample Results

| Condition | Strategy | Pair true precision | Pair group precision | R@1 | R@10 | R@50 | Lift@50 | Pos Sim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | True pair | 1.0000 | 1.0000 | 0.8250 | 0.9975 | 1.0000 | 32.00x | 0.8225 |
| Clean | Random | 0.0004 | 0.0141 | 0.0000 | 0.0063 | 0.0381 | 1.22x | 0.0065 |
| Clean | Metadata similarity | 0.0556 | 0.1830 | 0.0431 | 0.2425 | 0.5438 | 17.40x | 0.4133 |
| Clean | Propensity weighted | 0.0544 | 0.1808 | 0.0388 | 0.2275 | 0.5225 | 16.72x | 0.4123 |
| Moderate noise | True pair | 1.0000 | 1.0000 | 0.3313 | 0.7288 | 0.9106 | 29.14x | 0.6019 |
| Moderate noise | Random | 0.0004 | 0.0141 | 0.0006 | 0.0056 | 0.0375 | 1.20x | 0.0079 |
| Moderate noise | Metadata similarity | 0.0170 | 0.0444 | 0.0031 | 0.0163 | 0.0688 | 2.20x | 0.0671 |
| Moderate noise | Propensity weighted | 0.0178 | 0.0454 | 0.0044 | 0.0200 | 0.0769 | 2.46x | 0.0765 |
| High noise | True pair | 1.0000 | 1.0000 | 0.0288 | 0.1756 | 0.4000 | 12.80x | 0.3468 |
| High noise | Random | 0.0004 | 0.0141 | 0.0006 | 0.0063 | 0.0300 | 0.96x | 0.0022 |
| High noise | Metadata similarity | 0.0121 | 0.0300 | 0.0000 | 0.0088 | 0.0325 | 1.04x | 0.0081 |
| High noise | Propensity weighted | 0.0124 | 0.0320 | 0.0006 | 0.0094 | 0.0375 | 1.20x | 0.0211 |

## 24000-Sample Results

| Condition | Strategy | Pair true precision | Pair group precision | R@1 | R@10 | R@50 | Lift@50 | Pos Sim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | True pair | 1.0000 | 1.0000 | 0.7894 | 0.9923 | 1.0000 | 96.00x | 0.8434 |
| Clean | Random | 0.0000 | 0.0080 | 0.0002 | 0.0017 | 0.0096 | 0.92x | 0.0046 |
| Clean | Metadata similarity | 0.0214 | 0.1181 | 0.0175 | 0.1256 | 0.3348 | 32.14x | 0.4372 |
| Clean | Propensity weighted | 0.0215 | 0.1182 | 0.0177 | 0.1188 | 0.3298 | 31.66x | 0.4416 |
| Moderate noise | True pair | 1.0000 | 1.0000 | 0.2971 | 0.6602 | 0.8715 | 83.66x | 0.6259 |
| Moderate noise | Random | 0.0000 | 0.0080 | 0.0002 | 0.0025 | 0.0133 | 1.28x | 0.0137 |
| Moderate noise | Metadata similarity | 0.0057 | 0.0264 | 0.0004 | 0.0069 | 0.0285 | 2.74x | 0.0613 |
| Moderate noise | Propensity weighted | 0.0065 | 0.0261 | 0.0008 | 0.0073 | 0.0244 | 2.34x | 0.0623 |
| High noise | True pair | 1.0000 | 1.0000 | 0.0252 | 0.1173 | 0.3006 | 28.86x | 0.3728 |
| High noise | Random | 0.0000 | 0.0080 | 0.0004 | 0.0044 | 0.0146 | 1.40x | 0.0092 |
| High noise | Metadata similarity | 0.0044 | 0.0151 | 0.0002 | 0.0029 | 0.0125 | 1.20x | 0.0097 |
| High noise | Propensity weighted | 0.0038 | 0.0153 | 0.0000 | 0.0021 | 0.0125 | 1.20x | 0.0095 |

## Pair Quality Audit

### Oracle supervision is still sensitive to noise

The true-pair strategy acts as an upper bound. It performs strongly in clean and moderate-noise settings, but drops sharply under high noise.

This shows that even correct pair labels are not enough if the modality features become too weak or noisy.

### Random pairing behaves like a lower bound

Random pairing stays close to random retrieval behavior across conditions. This validates the benchmark: arbitrary pairs do not create meaningful cross-modal alignment.

### Metadata matching can recover signal from imperfect pairs

In clean settings, metadata similarity creates useful pseudo-pairs even with low exact true-pair precision.

For example, at 8000 samples, metadata similarity has only 5.56% exact pair precision but still reaches R@50 = 0.5438. This means the method recovers group-level or neighborhood-level signal even when exact pair recovery is imperfect.

### Propensity matching is more useful when the matching task becomes noisier

In the 8000-sample moderate-noise setting, propensity weighting improves over metadata similarity across R@1, R@5, R@10, R@50, lift@50, and positive similarity.

This suggests that learned metadata-based scoring can be more robust than raw metadata similarity when the pair construction problem becomes less clean.

### High noise exposes the breaking point

Under high noise, both metadata similarity and propensity weighting approach the random baseline. This means the metadata signal becomes too weak to support useful pseudo-pair construction.

The important lesson is not that one strategy always wins. The lesson is that pseudo-pairing quality has a failure boundary.

## Practical Interpretation

This benchmark supports a simple but important principle:

> A multimodal retrieval model is only as good as the pairs used to train it.

Better model architecture or longer training cannot fully compensate for weak pair construction. Before improving the encoder, it is important to ask whether the training pairs are meaningful.

In this experiment:

- true pairs show the learning upper bound
- random pairs show the failure baseline
- metadata similarity works when metadata is informative
- propensity weighting helps in some noisier regimes
- high noise breaks both pseudo-pairing methods

## Conclusion

Pseudo-pair construction is a central bottleneck in weakly paired multimodal learning.

The experiments show that metadata-based pseudo-pairing can recover useful retrieval signal when metadata remains informative. Propensity weighting can improve robustness in moderate-noise settings, but it does not solve the problem when metadata becomes too weak.

The final takeaway:

> Pairing strategy should be evaluated as a first-class part of the retrieval pipeline, not treated as a preprocessing detail.

## Boundary of This Benchmark

This benchmark is intentionally built as a controlled pairing laboratory. The synthetic setup allows the experiment to expose how pair quality, metadata reliability, and retrieval learning interact under known conditions.

The results should not be read as a universal ranking of matching algorithms. They show behavior inside a designed stress test where true pairs, noisy metadata, group structure, and retrieval difficulty are observable.

A natural next step would be to replace the simple propensity model with stronger matching estimators, add confidence-thresholded pair filtering, repeat runs across multiple seeds, and test whether the same pair-quality patterns appear in authorized real paired datasets.