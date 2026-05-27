from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


METADATA_NUMERIC = ["age", "severity_score", "condition_a", "condition_b", "condition_c"]
METADATA_CATEGORICAL = ["sex"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multimodal training pairs using different pairing strategies."
    )
    parser.add_argument("--a-csv", type=Path, default=Path("data_demo/modality_a.csv"))
    parser.add_argument("--b-csv", type=Path, default=Path("data_demo/modality_b.csv"))
    parser.add_argument("--true-pairs-csv", type=Path, default=Path("data_demo/true_pairs.csv"))
    parser.add_argument(
        "--strategy",
        choices=["true_pair", "random", "metadata_similarity", "propensity_weighted"],
        default="metadata_similarity",
    )
    parser.add_argument("--top-k-candidates", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=Path("data_demo/train_pairs.csv"))
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def prepare_metadata(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df[[f"{prefix}_id", "age", "severity_score", "condition_a", "condition_b", "condition_c", "sex"]].copy()
    out = pd.get_dummies(out, columns=METADATA_CATEGORICAL, drop_first=False)
    return out


def metadata_feature_matrix(a_df: pd.DataFrame, b_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    a_meta = prepare_metadata(a_df, "a")
    b_meta = prepare_metadata(b_df, "b")

    a_ids = a_meta["a_id"].astype(int).tolist()
    b_ids = b_meta["b_id"].astype(int).tolist()

    a_meta = a_meta.drop(columns=["a_id"])
    b_meta = b_meta.drop(columns=["b_id"])

    all_cols = sorted(set(a_meta.columns) | set(b_meta.columns))
    a_meta = a_meta.reindex(columns=all_cols, fill_value=0)
    b_meta = b_meta.reindex(columns=all_cols, fill_value=0)

    scaler = StandardScaler()
    combined = pd.concat([a_meta, b_meta], axis=0)
    scaler.fit(combined)

    a_x = scaler.transform(a_meta)
    b_x = scaler.transform(b_meta)

    return a_x, b_x, a_ids, b_ids


def build_true_pairs(a_df: pd.DataFrame, b_df: pd.DataFrame, true_pairs: pd.DataFrame) -> pd.DataFrame:
    pair_df = true_pairs[["a_id", "b_id"]].copy()
    pair_df["pair_score"] = 1.0
    pair_df["pair_strategy"] = "true_pair"
    return attach_pair_quality(pair_df, a_df, b_df)


def build_random_pairs(a_df: pd.DataFrame, b_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    a_ids = a_df["a_id"].astype(int).to_numpy()
    b_ids = b_df["b_id"].astype(int).to_numpy().copy()
    rng.shuffle(b_ids)

    pair_df = pd.DataFrame({"a_id": a_ids, "b_id": b_ids})
    pair_df["pair_score"] = 1.0
    pair_df["pair_strategy"] = "random"

    return attach_pair_quality(pair_df, a_df, b_df)


def build_metadata_similarity_pairs(a_df: pd.DataFrame, b_df: pd.DataFrame) -> pd.DataFrame:
    a_x, b_x, a_ids, b_ids = metadata_feature_matrix(a_df, b_df)

    sim = cosine_similarity(a_x, b_x)
    best_idx = sim.argmax(axis=1)

    pair_df = pd.DataFrame(
        {
            "a_id": a_ids,
            "b_id": [b_ids[j] for j in best_idx],
            "pair_score": sim[np.arange(len(a_ids)), best_idx],
            "pair_strategy": "metadata_similarity",
        }
    )

    return attach_pair_quality(pair_df, a_df, b_df)


def build_propensity_weighted_pairs(
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    true_pairs: pd.DataFrame,
    top_k_candidates: int,
    seed: int,
) -> pd.DataFrame:
    """
    Build propensity-style pair scores.

    The classifier is trained to distinguish true pairs from sampled non-pairs
    using metadata-difference features. It then scores candidate A-B pairs.
    """
    rng = np.random.default_rng(seed)

    true_pairs_small = true_pairs[["a_id", "b_id"]].copy()
    true_pairs_small["label"] = 1

    a_ids = a_df["a_id"].astype(int).to_numpy()
    b_ids = b_df["b_id"].astype(int).to_numpy()

    # Negative pairs: shuffled B ids.
    neg_b = b_ids.copy()
    rng.shuffle(neg_b)

    neg_pairs = pd.DataFrame({"a_id": a_ids, "b_id": neg_b})
    neg_pairs = neg_pairs[neg_pairs["a_id"] != neg_pairs["b_id"]].copy()
    neg_pairs["label"] = 0

    train_pairs = pd.concat([true_pairs_small, neg_pairs], axis=0).reset_index(drop=True)

    x_train = pair_metadata_features(train_pairs, a_df, b_df)
    y_train = train_pairs["label"].astype(int).to_numpy()

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(x_train, y_train)

    # Use metadata similarity to reduce candidate search.
    a_x, b_x, a_id_list, b_id_list = metadata_feature_matrix(a_df, b_df)
    sim = cosine_similarity(a_x, b_x)

    k = min(top_k_candidates, len(b_id_list))
    top_idx = np.argpartition(sim, -k, axis=1)[:, -k:]

    candidate_rows = []
    for i, a_id in enumerate(a_id_list):
        for j in top_idx[i]:
            candidate_rows.append({"a_id": int(a_id), "b_id": int(b_id_list[j])})

    candidate_df = pd.DataFrame(candidate_rows)
    x_candidates = pair_metadata_features(candidate_df, a_df, b_df)
    propensity_scores = clf.predict_proba(x_candidates)[:, 1]

    candidate_df["pair_score"] = propensity_scores
    candidate_df["pair_strategy"] = "propensity_weighted"

    # Keep best B candidate per A.
    candidate_df = candidate_df.sort_values(["a_id", "pair_score"], ascending=[True, False])
    pair_df = candidate_df.groupby("a_id", as_index=False).head(1).reset_index(drop=True)

    return attach_pair_quality(pair_df, a_df, b_df)


def pair_metadata_features(pair_df: pd.DataFrame, a_df: pd.DataFrame, b_df: pd.DataFrame) -> np.ndarray:
    a_meta = a_df[["a_id", "age", "severity_score", "condition_a", "condition_b", "condition_c", "sex"]].copy()
    b_meta = b_df[["b_id", "age", "severity_score", "condition_a", "condition_b", "condition_c", "sex"]].copy()

    merged = pair_df.merge(a_meta, on="a_id", how="left", suffixes=("", "_a"))
    merged = merged.merge(b_meta, on="b_id", how="left", suffixes=("_a", "_b"))

    features = pd.DataFrame()
    features["age_abs_diff"] = (merged["age_a"] - merged["age_b"]).abs()
    features["severity_abs_diff"] = (merged["severity_score_a"] - merged["severity_score_b"]).abs()
    features["condition_a_match"] = (merged["condition_a_a"] == merged["condition_a_b"]).astype(float)
    features["condition_b_match"] = (merged["condition_b_a"] == merged["condition_b_b"]).astype(float)
    features["condition_c_match"] = (merged["condition_c_a"] == merged["condition_c_b"]).astype(float)
    features["sex_match"] = (merged["sex_a"] == merged["sex_b"]).astype(float)

    return features.to_numpy(dtype=np.float32)


def attach_pair_quality(pair_df: pd.DataFrame, a_df: pd.DataFrame, b_df: pd.DataFrame) -> pd.DataFrame:
    a_info = a_df[["a_id", "true_pair_id", "group_id"]].rename(
        columns={"true_pair_id": "a_true_pair_id", "group_id": "a_group_id"}
    )
    b_info = b_df[["b_id", "true_pair_id", "group_id"]].rename(
        columns={"true_pair_id": "b_true_pair_id", "group_id": "b_group_id"}
    )

    out = pair_df.merge(a_info, on="a_id", how="left")
    out = out.merge(b_info, on="b_id", how="left")

    out["is_true_pair"] = (out["a_true_pair_id"] == out["b_true_pair_id"]).astype(int)
    out["is_same_group"] = (out["a_group_id"] == out["b_group_id"]).astype(int)

    return out[
        [
            "a_id",
            "b_id",
            "pair_strategy",
            "pair_score",
            "is_true_pair",
            "is_same_group",
            "a_group_id",
            "b_group_id",
        ]
    ]


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    a_path = resolve_path(args.a_csv, repo_root)
    b_path = resolve_path(args.b_csv, repo_root)
    true_pairs_path = resolve_path(args.true_pairs_csv, repo_root)
    out_path = resolve_path(args.out_csv, repo_root)

    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)
    true_pairs = pd.read_csv(true_pairs_path)

    if args.strategy == "true_pair":
        pair_df = build_true_pairs(a_df, b_df, true_pairs)
    elif args.strategy == "random":
        pair_df = build_random_pairs(a_df, b_df, seed=args.seed)
    elif args.strategy == "metadata_similarity":
        pair_df = build_metadata_similarity_pairs(a_df, b_df)
    elif args.strategy == "propensity_weighted":
        pair_df = build_propensity_weighted_pairs(
            a_df=a_df,
            b_df=b_df,
            true_pairs=true_pairs,
            top_k_candidates=args.top_k_candidates,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(out_path, index=False)

    summary = {
        "strategy": args.strategy,
        "n_pairs": int(len(pair_df)),
        "pair_precision_true": float(pair_df["is_true_pair"].mean()),
        "pair_precision_same_group": float(pair_df["is_same_group"].mean()),
        "pair_score_mean": float(pair_df["pair_score"].mean()),
        "pair_score_std": float(pair_df["pair_score"].std()),
    }

    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Strategy              : {args.strategy}")
    print(f"Pairs written          : {out_path}")
    print(f"Summary written        : {summary_path}")
    print(f"Pair precision true    : {summary['pair_precision_true']:.4f}")
    print(f"Pair precision group   : {summary['pair_precision_same_group']:.4f}")
    print(f"Mean pair score        : {summary['pair_score_mean']:.4f}")


if __name__ == "__main__":
    main()