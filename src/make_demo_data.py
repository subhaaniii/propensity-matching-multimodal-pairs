from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled synthetic multimodal data for pair-matching experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["clean", "moderate_noise", "high_noise"],
        default="clean",
        help="Controls metadata noise and pair ambiguity.",
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-groups", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def make_projection(rng: np.random.Generator, in_dim: int, out_dim: int) -> np.ndarray:
    mat = rng.normal(0.0, 1.0, size=(in_dim, out_dim))
    return mat / np.sqrt(in_dim)


def mode_settings(mode: str) -> dict[str, float]:
    if mode == "clean":
        return {
            "latent_noise": 0.12,
            "metadata_noise": 0.05,
            "feature_noise": 0.10,
        }
    if mode == "moderate_noise":
        return {
            "latent_noise": 0.25,
            "metadata_noise": 0.18,
            "feature_noise": 0.18,
        }
    if mode == "high_noise":
        return {
            "latent_noise": 0.40,
            "metadata_noise": 0.35,
            "feature_noise": 0.28,
        }
    raise ValueError(f"Unknown mode: {mode}")


def generate_samples(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if args.n_samples < args.n_groups:
        raise ValueError("--n-samples must be >= --n-groups")

    rng = np.random.default_rng(args.seed)
    settings = mode_settings(args.mode)

    sample_ids = np.arange(args.n_samples)

    group_ids = np.arange(args.n_samples) % args.n_groups
    rng.shuffle(group_ids)

    group_centers = rng.normal(0.0, 1.0, size=(args.n_groups, args.latent_dim))
    group_centers = normalize_rows(group_centers)

    latent = (
        group_centers[group_ids]
        + rng.normal(0.0, settings["latent_noise"], size=(args.n_samples, args.latent_dim))
    )
    latent = normalize_rows(latent)

    proj_a = make_projection(rng, args.latent_dim, args.feature_dim)
    proj_b = make_projection(rng, args.latent_dim, args.feature_dim)

    a_features = latent @ proj_a + rng.normal(0.0, settings["feature_noise"], size=(args.n_samples, args.feature_dim))
    b_features = latent @ proj_b + rng.normal(0.0, settings["feature_noise"], size=(args.n_samples, args.feature_dim))

    a_features = normalize_rows(a_features)
    b_features = normalize_rows(b_features)

    # Metadata: deliberately correlated with latent group, but noisy.
    age_base = 40 + 25 * (group_ids / max(args.n_groups - 1, 1))
    age = age_base + rng.normal(0, 5 + 15 * settings["metadata_noise"], args.n_samples)
    age = np.clip(age, 18, 90)

    severity_score = (
        0.55 * latent[:, 0]
        + 0.30 * latent[:, 1]
        + rng.normal(0, settings["metadata_noise"], args.n_samples)
    )
    severity_score = (severity_score - severity_score.min()) / (
        severity_score.max() - severity_score.min() + 1e-8
    )

    condition_a = (latent[:, 2] + rng.normal(0, settings["metadata_noise"], args.n_samples) > 0).astype(int)
    condition_b = (latent[:, 3] + rng.normal(0, settings["metadata_noise"], args.n_samples) > 0).astype(int)
    condition_c = (latent[:, 4] + rng.normal(0, settings["metadata_noise"], args.n_samples) > 0).astype(int)

    sex = np.where(
        latent[:, 5] + rng.normal(0, settings["metadata_noise"], args.n_samples) > 0,
        "M",
        "F",
    )

    # Make modality-specific metadata versions.
    # This simulates metadata measured from two different sources.
    age_a = np.clip(age + rng.normal(0, 1.5 + 6 * settings["metadata_noise"], args.n_samples), 18, 90)
    age_b = np.clip(age + rng.normal(0, 1.5 + 6 * settings["metadata_noise"], args.n_samples), 18, 90)

    severity_a = np.clip(severity_score + rng.normal(0, settings["metadata_noise"], args.n_samples), 0, 1)
    severity_b = np.clip(severity_score + rng.normal(0, settings["metadata_noise"], args.n_samples), 0, 1)

    def make_modality_df(prefix: str, features: np.ndarray, age_vals: np.ndarray, severity_vals: np.ndarray) -> pd.DataFrame:
        rows = []
        for i in range(args.n_samples):
            row = {
                f"{prefix}_id": int(sample_ids[i]),
                "true_pair_id": int(sample_ids[i]),
                "group_id": int(group_ids[i]),
                "age": float(age_vals[i]),
                "severity_score": float(severity_vals[i]),
                "condition_a": int(condition_a[i]),
                "condition_b": int(condition_b[i]),
                "condition_c": int(condition_c[i]),
                "sex": str(sex[i]),
            }
            for j in range(args.feature_dim):
                row[f"{prefix}_feat_{j:03d}"] = float(features[i, j])
            rows.append(row)
        return pd.DataFrame(rows)

    a_df = make_modality_df("a", a_features, age_a, severity_a)
    b_df = make_modality_df("b", b_features, age_b, severity_b)

    true_pairs = pd.DataFrame(
        {
            "a_id": sample_ids.astype(int),
            "b_id": sample_ids.astype(int),
            "true_pair": 1,
            "group_id": group_ids.astype(int),
        }
    )

    metadata = {
        "mode": args.mode,
        "n_samples": args.n_samples,
        "n_groups": args.n_groups,
        "latent_dim": args.latent_dim,
        "feature_dim": args.feature_dim,
        "seed": args.seed,
        "settings": settings,
        "description": (
            "Controlled synthetic multimodal data for pseudo-pair construction "
            "and propensity-style matching experiments."
        ),
    }

    return a_df, b_df, true_pairs, metadata


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data_demo"
    data_dir.mkdir(parents=True, exist_ok=True)

    a_df, b_df, true_pairs, metadata = generate_samples(args)

    a_path = data_dir / "modality_a.csv"
    b_path = data_dir / "modality_b.csv"
    true_pairs_path = data_dir / "true_pairs.csv"
    metadata_path = data_dir / "demo_metadata.json"

    a_df.to_csv(a_path, index=False)
    b_df.to_csv(b_path, index=False)
    true_pairs.to_csv(true_pairs_path, index=False)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Mode          : {args.mode}")
    print(f"Samples       : {args.n_samples}")
    print(f"Groups        : {args.n_groups}")
    print(f"Modality A    : {a_path}")
    print(f"Modality B    : {b_path}")
    print(f"True pairs    : {true_pairs_path}")
    print(f"Metadata      : {metadata_path}")


if __name__ == "__main__":
    main()