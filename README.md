# Propensity Matching for Multimodal Pairs

A controlled benchmark for studying how pseudo-pair construction affects multimodal retrieval.

This repository focuses on the pairing step: before training a retrieval model, how should samples from two modalities be matched when true pairs are unavailable, noisy, or only approximately recoverable?

The project compares random pairing, metadata-similarity pairing, and propensity-weighted pairing against a true-pair upper bound.

---

## Why Pair Construction Matters

In multimodal contrastive learning, the model learns from paired examples.

If the pair assignments are reliable, the model can learn meaningful cross-modal alignment. If the pair assignments are weak or noisy, the model may learn the wrong structure, even if the encoder and loss function are reasonable.

This repository treats pair construction as a first-class part of the retrieval pipeline.

The central idea is:

> Before asking whether the model is good, ask whether the training pairs are meaningful.

---

## Core Question

Can metadata-based pseudo-pair construction recover useful retrieval signal when exact cross-modal pairs are unavailable or difficult to identify?

The benchmark tests this by changing:

- metadata reliability
- feature noise
- sample size
- pairing strategy
- exact-pair and same-group pair precision

---

## Pairing Strategies

| Strategy | Role |
|---|---|
| True pair | Oracle upper bound using ground-truth pairs |
| Random | Lower bound using arbitrary pairs |
| Metadata similarity | Simple heuristic matching based on metadata closeness |
| Propensity weighted | Learned metadata-based matching score for pseudo-pair selection |

The true-pair strategy is not meant as a deployable method. It is included as an upper bound so the pseudo-pair methods can be interpreted properly.

---

## Related Work

This repository is motivated by propensity-score methods for matching and recent work on unpaired multimodal alignment.

The classical foundation comes from Rosenbaum and Rubin's work on propensity scores, where the propensity score is defined as the conditional probability of treatment assignment given observed covariates. In causal inference, matching on the propensity score is used to make groups more comparable when direct randomized assignment is not available.

This idea is closely related to the problem studied in this repository: when exact cross-modal pairs are unavailable, metadata can be used to estimate whether two samples are likely to belong together.

The most directly related multimodal work is *Propensity Score Alignment of Unpaired Multimodal Data*. That paper uses propensity scores as a shared matching space for unpaired multimodal samples, then applies matching methods such as shared nearest neighbours and optimal transport.

This repository does not reproduce that paper directly. Instead, it builds a smaller controlled benchmark to compare random pairing, metadata-similarity pairing, and propensity-weighted pseudo-pairing under different noise levels.


## Synthetic Benchmark

The benchmark uses controlled synthetic multimodal data with:

| Component | Description |
|---|---|
| Modality A | Synthetic feature vector |
| Modality B | Synthetic feature vector |
| Metadata | Age, severity score, binary conditions, sex |
| Group label | Latent group used to measure approximate semantic matching |
| True pair ID | Known exact cross-modal pair |

Three data conditions are tested:

| Condition | Meaning |
|---|---|
| Clean | Metadata and modality features are relatively aligned |
| Moderate noise | Metadata and features become less reliable |
| High noise | Metadata becomes weak and pair ambiguity becomes severe |

Two sample sizes are used:

| Total samples | Held-out retrieval pool |
|---:|---:|
| 8000 | 1600 |
| 24000 | 4800 |

All reported runs use 50 training epochs.

---

## Experiment Matrix

```text
3 data conditions × 2 sample sizes × 4 pairing strategies = 24 runs
```

| Variable | Values |
|---|---|
| Data condition | clean, moderate noise, high noise |
| Sample size | 8000, 24000 |
| Pairing strategy | true pair, random, metadata similarity, propensity weighted |
| Training duration | 50 epochs |

---

## Metrics

This repository reports both pair quality and retrieval quality.

| Metric | Meaning |
|---|---|
| Pair true precision | Fraction of constructed pairs that are exact true pairs |
| Pair group precision | Fraction of constructed pairs from the same latent group |
| Recall@K | Whether the correct match appears in the top K retrieved candidates |
| Lift@K | Retrieval improvement over random chance |
| Positive-pair similarity | Cosine similarity of true pairs in the learned embedding space |
| Training loss | Contrastive optimization loss |

This separation matters because retrieval results can be misleading if pair quality is not reported.

---

## Key Findings

The benchmark shows that pair construction quality strongly controls retrieval performance.

### True pairs define the upper bound

When the true pair is available, the retrieval model learns strong alignment in clean and moderate-noise settings. However, even true-pair performance drops under high noise, showing that feature quality still matters.

### Random pairs behave like a lower bound

Random pairing stays close to random retrieval behavior. This validates that the model cannot learn useful cross-modal alignment from arbitrary pair assignments.

### Metadata similarity can recover useful signal

In clean settings, metadata similarity recovers meaningful retrieval signal even when exact-pair precision is low.

For example, in the 8000-sample clean setting, metadata similarity reaches strong Recall@50 despite having only a small fraction of exact true pairs. This suggests that approximate pair construction can still provide useful neighborhood-level supervision.

### Propensity weighting helps in noisier regimes

In the 8000-sample moderate-noise setting, propensity-weighted matching improves over metadata similarity across several retrieval metrics, including Recall@1, Recall@10, Recall@50, lift@50, and positive-pair similarity.

This suggests that learned metadata-based scoring can be more robust than raw metadata similarity when pair construction becomes harder.

### High noise exposes the failure boundary

Under high noise, both metadata similarity and propensity weighting approach the random baseline. This means that when metadata becomes too weak, pseudo-pair construction cannot reliably recover useful training pairs.

The important lesson is not that one strategy always wins. The lesson is that pair construction has a measurable breaking point.

---

## Results

The full benchmark results are available here:

- [Full benchmark summary](experiments/results_summary.md)
- [Raw result table](experiments/results_table.csv)

The summary file includes the complete 24-run comparison across clean, moderate-noise, and high-noise conditions.

---

## Repository Structure

```text
propensity-matching-multimodal-pairs/
│
├── src/
│   ├── make_demo_data.py
│   ├── build_pairs.py
│   ├── model.py
│   ├── metrics.py
│   ├── train.py
│   └── collect_results.py
│
├── data_demo/
│   ├── modality_a.csv
│   ├── modality_b.csv
│   ├── true_pairs.csv
│   └── demo_metadata.json
│
├── experiments/
│   ├── results_table.csv
│   └── results_summary.md
│
├── docs/
│
├── requirements.txt
└── README.md
```

---

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Generate synthetic multimodal data:

```powershell
python src/make_demo_data.py --mode moderate_noise --n-samples 8000 --n-groups 80
```

Build pseudo-pairs with metadata similarity:

```powershell
python src/build_pairs.py --strategy metadata_similarity --out-csv data_demo/pairs_metadata.csv
```

Build pseudo-pairs with propensity weighting:

```powershell
python src/build_pairs.py --strategy propensity_weighted --out-csv data_demo/pairs_propensity.csv
```

Train using one pair file:

```powershell
python src/train.py --pairs-csv data_demo/pairs_propensity.csv --epochs 50 --batch-size 512 --output-dir outputs/example_propensity --checkpoint-dir checkpoints/example_propensity
```

Collect result tables:

```powershell
python src/collect_results.py
```

Generated folders such as `outputs/` and `checkpoints/` are intentionally ignored by Git.

---

## Interpretation Guide

Use the strategies as reference points:

| Strategy | How to interpret it |
|---|---|
| True pair | Best-case learning when supervision is correct |
| Random | Failure baseline |
| Metadata similarity | Simple metadata-based pseudo-pairing |
| Propensity weighted | Learned metadata-based pseudo-pairing |

A useful pseudo-pairing method should do better than random and move closer to the true-pair upper bound.

---

## References

- Paul R. Rosenbaum and Donald B. Rubin. *The Central Role of the Propensity Score in Observational Studies for Causal Effects*. Biometrika, 1983.
- Peter C. Austin. *An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies*. Multivariate Behavioral Research, 2011.
- Johnny Xi, Jana Osea, Zuheng Xu, and Jason Hartford. *Propensity Score Alignment of Unpaired Multimodal Data*. NeurIPS, 2024.
- Cédric Villani. *Optimal Transport: Old and New*. Springer, 2009.

## Boundary of This Benchmark

This benchmark is a controlled pairing laboratory. The synthetic setup makes it possible to observe exact pair quality, same-group quality, metadata noise, and retrieval behavior under known conditions.

The results should not be read as a universal ranking of matching algorithms. They show how these specific pair-construction strategies behave inside a designed stress test.

A natural next step would be to test stronger matching estimators, add confidence-thresholded pair filtering, repeat runs across multiple seeds, and evaluate whether the same pair-quality patterns appear in authorized real paired datasets.