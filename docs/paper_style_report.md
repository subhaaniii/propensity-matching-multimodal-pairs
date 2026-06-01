# Paper-Style Report: Propensity Matching for Multimodal Pairs

## Abstract

This project studies how pseudo-pair construction affects multimodal retrieval when exact cross-modal pairs are unavailable, noisy, or only approximately recoverable. The benchmark compares true-pair supervision, random pairing, metadata-similarity pairing, and propensity-weighted pairing under clean, moderate-noise, and high-noise synthetic settings. The results show that pair quality strongly controls retrieval performance. True pairs act as an oracle upper bound, random pairs behave like a lower bound, and metadata-based strategies can recover useful retrieval signal when metadata remains informative. Under high noise, pseudo-pairing methods approach random behavior, showing a clear failure boundary for weak pair construction.

## 1. Motivation

Multimodal contrastive learning depends on paired examples. In many real-world settings, especially medical and observational datasets, exact pairs may be unavailable or difficult to verify. If the training pairs are weak or noisy, the model may learn poor cross-modal alignment even when the encoder architecture and contrastive loss are reasonable.

This project treats pair construction as a central part of the retrieval pipeline. Instead of asking only whether a model performs well, the project asks whether the training pairs are meaningful enough for the model to learn useful alignment.

## 2. Research Question

The main research question is:

> Can metadata-based pseudo-pair construction recover useful retrieval signal when exact cross-modal pairs are unavailable?

The project studies this by varying:

- pairing strategy
- metadata reliability
- feature noise
- sample size
- exact-pair precision
- same-group pair precision

## 3. Method

The benchmark compares four pairing strategies:

| Strategy | Role |
|---|---|
| True pair | Oracle upper bound using ground-truth pairs |
| Random | Lower bound using arbitrary pairs |
| Metadata similarity | Heuristic matching based on metadata closeness |
| Propensity weighted | Learned metadata-based matching score for pseudo-pair selection |

The true-pair strategy is included only as an upper bound. The random strategy is included as a lower bound. Metadata similarity and propensity weighting are the main pseudo-pair construction methods being tested.

## 4. Dataset and Experimental Setup

The project uses controlled synthetic multimodal data. Each sample has:

- modality A feature vector
- modality B feature vector
- metadata features
- latent group label
- known true pair ID

Three data conditions are tested:

| Condition | Meaning |
|---|---|
| Clean | Metadata and features are relatively aligned |
| Moderate noise | Metadata and features are less reliable |
| High noise | Metadata becomes weak and pair ambiguity becomes severe |

Two sample sizes are used:

| Total samples | Held-out retrieval pool |
|---|---|
| 8000 | 1600 |
| 24000 | 4800 |

All reported runs use 50 training epochs.

## 5. Evaluation Metrics

The project separates pair quality metrics from retrieval quality metrics.

| Metric | Meaning |
|---|---|
| Pair true precision | Fraction of constructed pairs that are exact true pairs |
| Pair group precision | Fraction of constructed pairs from the same latent group |
| Recall@K | Whether the correct match appears in the top K retrieved candidates |
| Lift@K | Retrieval improvement over random chance |
| Positive-pair similarity | Cosine similarity of true pairs in the learned embedding space |
| Training loss | Contrastive optimization loss |

This separation is important because retrieval results can be misleading if pair quality is not measured.

## 6. Experiments

The benchmark runs:

```text
3 data conditions × 2 sample sizes × 4 pairing strategies = 24 runs
