# Propensity Matching for Multimodal Pairs

A controlled benchmark for studying how pseudo-pair construction affects multimodal retrieval.

This repository focuses on the pairing step: before training a retrieval model, how should samples from two modalities be matched when true pairs are unavailable, noisy, or only approximately recoverable?

The project compares random pairing, metadata-similarity pairing, and propensity-weighted pairing against a true-pair upper bound.

## Research Report

A paper-style summary of this project is available here:

[Read the paper-style report](docs/paper_style_report.md)

---

## Why Pair Construction Matters

In multimodal contrastive learning, the model learns from paired examples.

If the pair assignments are reliable, the model can learn meaningful cross-modal alignment. If the pair assignments are weak or noisy, the model may learn the wrong structure, even if the encoder and loss function are reasonable.

This repository treats pair construction as a first-class part of the retrieval pipeline.

The central idea is:

> Before asking whether the model is good, ask whether the training pairs are meaningful.

---

## Core Question

> Can metadata-based pseudo-pair construction recover useful retrieval signal when exact cross-modal pairs are unavailable or difficult to identify?

The benchmark tests this by changing:

- metadata reliability
- feature noise
- sample size
- pairing strategy
- exact-pair and same-group pair precision

---

## Pairing Strategies

This benchmark compares true-pair supervision, random pairing, metadata-similarity pairing, and propensity-weighted pseudo-pairing.

The goal is to test whether pseudo-pairing methods can move above random pairing and closer to the true-pair upper bound.

For the detailed method description, see the [paper-style report](docs/paper_style_report.md#3-method).

---

## Benchmark Setup

The benchmark uses controlled synthetic multimodal data with modality-A features, modality-B features, metadata, latent group labels, and known true-pair IDs.

Experiments are run across clean, moderate-noise, and high-noise settings, using two dataset sizes: 8000 and 24000 samples.

The full experiment matrix and metric definitions are included in the [paper-style report](docs/paper_style_report.md#4-dataset-and-experimental-setup).

---

## Related Work

This project is motivated by propensity-score matching and recent work on unpaired multimodal alignment.

For the full related-work discussion and references, see the [paper-style report](docs/paper_style_report.md#12-related-work).

---

## Pair-Construction Visualizations

The plots below summarize pair quality, retrieval performance, and the constructed pseudo-pair space.

<table>
  <tr>
    <th>Pair quality</th>
    <th>Retrieval comparison</th>
    <th>Propensity pairing space</th>
  </tr>
  <tr>
    <td width="33%">
      <a href="figures/pair_quality_comparison.png">
        <img src="figures/pair_quality_comparison.png" alt="Pair quality comparison" width="100%">
      </a>
    </td>
    <td width="33%">
      <a href="figures/retrieval_strategy_comparison.png">
        <img src="figures/retrieval_strategy_comparison.png" alt="Retrieval strategy comparison" width="100%">
      </a>
    </td>
    <td width="33%">
      <a href="figures/propensity_pairing_space.png">
        <img src="figures/propensity_pairing_space.png" alt="Propensity pairing space" width="100%">
      </a>
    </td>
  </tr>
</table>

Each panel links to the full-resolution figure.

| Panel | What to notice |
|---|---|
| **Pair quality** | True-pair supervision is the oracle upper bound, random pairing is the lower bound, and metadata/propensity methods recover partial group-level pairing signal. |
| **Retrieval comparison** | Retrieval performance follows pair quality: true pairs perform best, random pairs perform worst, and pseudo-pairing methods sit between them. |
| **Propensity pairing space** | Circles and crosses represent the two modalities. Lines show selected propensity-weighted constructed pairs. Shorter and more local lines suggest more geometrically plausible pseudo-pairs. |

These figures are qualitative diagnostics. The main conclusions are based on the quantitative results in `experiments/results_table.csv`.

## Results

The full benchmark results are available here:

- [Full benchmark summary](experiments/results_summary.md)
- [Raw result table](experiments/results_table.csv)

The summary file includes the complete 24-run comparison across clean, moderate-noise, and high-noise conditions.

---

## Main Findings

- True-pair supervision gives the strongest retrieval performance and acts as an oracle upper bound.
- Random pairing behaves like a lower bound and does not provide useful alignment signal.
- Metadata similarity can recover useful signal when metadata is informative.
- Propensity-weighted matching can improve over raw metadata similarity in harder settings.
- Under high noise, pseudo-pairing methods approach the random baseline, showing a clear failure boundary.

For the full key findings and interpretation, see the [paper-style report](docs/paper_style_report.md#8-key-findings).

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

A useful pseudo-pairing method should perform better than random pairing and move closer to the true-pair upper bound.

For the full interpretation of each strategy, see the [paper-style report](docs/paper_style_report.md#3-method).

---

## Benchmark Boundary

This is a controlled synthetic benchmark, so the results should be interpreted as method-behavior analysis rather than a universal ranking of matching algorithms.

For limitations and future work, see the [paper-style report](docs/paper_style_report.md#9-limitations).

## Documentation

- [Pair quality audit](docs/pair_quality_audit.md)
- [Matching strategy notes](docs/matching_strategy_notes.md)
- [Metric interpretation](docs/metric_interpretation.md)
- [Reproducibility protocol](docs/reproducibility_protocol.md)

---

## References

References are included in the [paper-style report](docs/paper_style_report.md).

