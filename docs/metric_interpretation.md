# Metric Interpretation

This benchmark reports both pair-construction metrics and retrieval metrics.

## Pair true precision

Pair true precision measures the fraction of constructed pairs that match the exact ground-truth pair.

```text
pair_true_precision = exact_true_pairs / constructed_pairs
```

This is a strict metric.

A pseudo-pairing method can have low exact-pair precision but still provide useful training signal if it often selects samples from the same latent group.

## Pair group precision

Pair group precision measures the fraction of constructed pairs that come from the same latent group.

```text
pair_group_precision = same_group_pairs / constructed_pairs
```

This is a softer metric. It helps show whether pseudo-pairs are semantically nearby even when they are not exact matches.

## Recall@K

Recall@K measures whether the correct modality-B sample appears in the top K retrieved candidates for a given modality-A sample.

Examples:

- Recall@1: correct match is ranked first
- Recall@5: correct match appears in the top 5
- Recall@10: correct match appears in the top 10
- Recall@50: correct match appears in the top 50

Higher Recall@K means better retrieval.

## Lift@K

Lift@K compares retrieval performance against random chance.

For example, if the held-out retrieval pool contains 1600 candidates, random Recall@50 is:

```text
50 / 1600 = 0.03125
```

If the model gets Recall@50 = 0.50, then:

```text
Lift@50 = 0.50 / 0.03125 = 16.0x
```

This means the model is 16 times better than random retrieval at K=50.

## Positive-pair similarity

Positive-pair similarity is the average cosine similarity between true cross-modal pairs in the learned embedding space.

A higher value means the model is placing true pairs closer together.

However, positive similarity should not be read alone. A model can increase similarity without ranking the true match correctly among many candidates.

## Why pair metrics and retrieval metrics are both needed

Pair quality explains the supervision signal.

Retrieval metrics explain the learned model behavior.

Both are needed because a pseudo-pairing strategy may have low exact-pair precision but still create enough neighborhood-level structure for useful retrieval.