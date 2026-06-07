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
```

The experiment matrix tests whether pseudo-pairing methods can move closer to the true-pair upper bound and away from the random-pair lower bound.

## 7. Results

The results show that pair construction quality strongly affects retrieval performance.

True-pair supervision gives the strongest retrieval performance in clean and moderate-noise settings, but performance still drops under high noise. Random pairing stays close to random retrieval behavior, confirming that arbitrary pairs do not provide useful alignment signal.

Metadata similarity can recover useful signal when metadata is informative. Propensity-weighted matching can improve over raw metadata similarity in noisier settings, especially when the metadata contains learnable structure. However, under high noise, both metadata similarity and propensity weighting approach the random baseline.

## 8. Key Findings

### 8.1 True pairs define the upper bound

When exact pairs are available, the retrieval model learns stronger alignment. This gives a useful reference point for evaluating pseudo-pairing methods.

### 8.2 Random pairs define the lower bound

Random pair construction does not provide meaningful cross-modal supervision. This validates the benchmark design.

### 8.3 Metadata similarity can recover useful signal

When metadata is informative, approximate pairing can still produce useful neighborhood-level supervision even when exact-pair precision is low.

### 8.4 Propensity weighting helps in harder settings

Propensity-weighted matching can outperform raw metadata similarity when pair construction becomes more difficult but metadata still contains useful signal.

### 8.5 High noise exposes the failure boundary

When metadata becomes too weak, pseudo-pair construction fails to recover useful alignment. This shows that weak supervision has a measurable breaking point.

## 9. Limitations

This benchmark uses synthetic data, so the results should not be interpreted as a universal ranking of matching algorithms. The controlled setup is useful for studying failure modes, but real-world multimodal datasets may contain more complex noise, missingness, bias, and domain-specific structure.

The benchmark also uses a limited set of pairing strategies. Stronger matching approaches, confidence filtering, optimal transport variants, and repeated multi-seed evaluation could provide deeper evidence.

## 10. Future Work

Possible extensions include:

- testing stronger matching estimators
- adding confidence-thresholded pair filtering
- repeating experiments across multiple random seeds
- applying the benchmark to authorized real multimodal datasets
- studying uncertainty in pseudo-pair construction
- comparing additional contrastive loss variants
- analyzing embedding geometry after different pairing strategies

## 11. What I Learned

This project taught me that multimodal retrieval performance is not only a model problem. It is also a data-pairing problem. A strong encoder and a reasonable loss function cannot fully compensate for weak or misleading pair assignments.

The most important lesson is:

> Before asking whether the retrieval model is good, first ask whether the training pairs are meaningful.

This changed how I think about multimodal learning pipelines. Pair construction, supervision quality, retrieval metrics, and embedding behavior must be evaluated together.

## 12. Related Work

This repository is motivated by propensity-score methods for matching and recent work on unpaired multimodal alignment.

The classical foundation comes from Rosenbaum and Rubin's work on propensity scores, where the propensity score is defined as the conditional probability of treatment assignment given observed covariates. In causal inference, matching on the propensity score is used to make groups more comparable when direct randomized assignment is not available.

This idea is closely related to the problem studied in this repository: when exact cross-modal pairs are unavailable, metadata can be used to estimate whether two samples are likely to belong together.

The most directly related multimodal work is *Propensity Score Alignment of Unpaired Multimodal Data*. That paper uses propensity scores as a shared matching space for unpaired multimodal samples, then applies matching methods such as shared nearest neighbours and optimal transport.

This repository does not reproduce that paper directly. Instead, it builds a smaller controlled benchmark to compare random pairing, metadata-similarity pairing, and propensity-weighted pseudo-pairing under different noise levels.

## 13. References

- Paul R. Rosenbaum and Donald B. Rubin. *The Central Role of the Propensity Score in Observational Studies for Causal Effects*. Biometrika, 1983.
- Peter C. Austin. *An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies*. Multivariate Behavioral Research, 2011.
- Johnny Xi, Jana Osea, Zuheng Xu, and Jason Hartford. *Propensity Score Alignment of Unpaired Multimodal Data*. NeurIPS, 2024.
- Cédric Villani. *Optimal Transport: Old and New*. Springer, 2009.

### Why these references?

- **Rosenbaum and Rubin** provide the classical statistical foundation for propensity scores and matching.
- **Austin** gives a practical overview of propensity-score methods and why they are useful for reducing confounding in observational data.
- **Propensity Score Alignment of Unpaired Multimodal Data** is the closest modern reference because it applies propensity-score ideas to unpaired multimodal matching.
- **Villani** is included because optimal transport is a major matching framework used in multimodal alignment literature, including propensity-score-based alignment work.


