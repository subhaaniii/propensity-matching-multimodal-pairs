# Reproducibility Protocol

This document gives the basic steps for reproducing the benchmark.

## Install dependencies

```powershell
pip install -r requirements.txt
```

## Generate synthetic data

Clean setting:

```powershell
python src/make_demo_data.py --mode clean --n-samples 8000 --n-groups 80
```

Moderate-noise setting:

```powershell
python src/make_demo_data.py --mode moderate_noise --n-samples 8000 --n-groups 80
```

High-noise setting:

```powershell
python src/make_demo_data.py --mode high_noise --n-samples 8000 --n-groups 80
```

For the larger benchmark, use:

```powershell
python src/make_demo_data.py --mode clean --n-samples 24000 --n-groups 120
```

## Build pair files

True-pair upper bound:

```powershell
python src/build_pairs.py --strategy true_pair --out-csv data_demo/pairs_true.csv
```

Random lower bound:

```powershell
python src/build_pairs.py --strategy random --out-csv data_demo/pairs_random.csv
```

Metadata-similarity pseudo-pairs:

```powershell
python src/build_pairs.py --strategy metadata_similarity --out-csv data_demo/pairs_metadata.csv
```

Propensity-weighted pseudo-pairs:

```powershell
python src/build_pairs.py --strategy propensity_weighted --out-csv data_demo/pairs_propensity.csv
```

## Train a retrieval model

```powershell
python src/train.py --pairs-csv data_demo/pairs_propensity.csv --epochs 50 --batch-size 512 --output-dir outputs/example_propensity --checkpoint-dir checkpoints/example_propensity
```

## Collect results

```powershell
python src/collect_results.py
```

This writes:

```text
experiments/results_table.csv
```

## Files not tracked by Git

The following generated folders are intentionally ignored:

```text
outputs/
checkpoints/
```

The repository only tracks source code, small demo data, documentation, and aggregate experiment results.