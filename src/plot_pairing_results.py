from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data_demo"
RESULTS_PATH = REPO_ROOT / "experiments" / "results_table.csv"
FIGURES_DIR = REPO_ROOT / "figures"


STRATEGY_ORDER = [
    "true_pair",
    "random",
    "metadata_similarity",
    "propensity_weighted",
]

STRATEGY_LABELS = {
    "true_pair": "True pair",
    "random": "Random",
    "metadata_similarity": "Metadata similarity",
    "propensity_weighted": "Propensity weighted",
}

MODE_ORDER = ["clean", "moderate_noise", "high_noise"]

MODE_LABELS = {
    "clean": "Clean",
    "moderate_noise": "Moderate noise",
    "high_noise": "High noise",
}


def feature_columns(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    for prefix in prefixes:
        cols = sorted([c for c in df.columns if c.startswith(prefix)])
        if cols:
            return cols

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in {"sample_id", "group_id", "pair_id", "true_pair_id"}
    ]

    if numeric_cols:
        return numeric_cols

    raise RuntimeError(f"No feature columns found for prefixes: {prefixes}")

def get_id_series(df: pd.DataFrame, one_based: bool = True) -> pd.Series:
    for col in ["sample_id", "id", "idx", "index"]:
        if col in df.columns:
            return df[col].astype(int)

    # Fallback: use original row number as sample id.
    # Most generated pair files use 1-based IDs, so default is 1-based.
    offset = 1 if one_based else 0
    return pd.Series(np.arange(len(df)) + offset, index=df.index, dtype=int)

def load_pair_file(strategy: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"pairs_{strategy}.csv")


def plot_pair_quality(n_samples: int = 24000) -> None:
    df = pd.read_csv(RESULTS_PATH)
    df = df[df["n_samples"] == n_samples].copy()

    if df.empty:
        raise RuntimeError(f"No rows found for n_samples={n_samples}")

    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    df["strategy"] = pd.Categorical(df["strategy"], categories=STRATEGY_ORDER, ordered=True)
    df = df.sort_values(["mode", "strategy"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)

    metrics = [
        ("pair_precision_true", "Exact true-pair precision"),
        ("pair_precision_group", "Same-group pair precision"),
    ]

    x = np.arange(len(MODE_ORDER))
    width = 0.20

    for ax, (metric_col, title) in zip(axes, metrics):
        for i, strategy in enumerate(STRATEGY_ORDER):
            values = []
            for mode in MODE_ORDER:
                row = df[(df["mode"] == mode) & (df["strategy"] == strategy)]
                values.append(float(row[metric_col].iloc[0]) if len(row) else np.nan)

            ax.bar(
                x + (i - 1.5) * width,
                values,
                width=width,
                label=STRATEGY_LABELS[strategy],
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
        ax.set_ylabel("Precision")
        ax.grid(axis="y", alpha=0.25)

    axes[1].legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
    )

    fig.suptitle(f"Pair-construction quality ({n_samples} samples)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pair_quality_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_comparison(n_samples: int = 24000) -> None:
    df = pd.read_csv(RESULTS_PATH)
    df = df[df["n_samples"] == n_samples].copy()

    if df.empty:
        raise RuntimeError(f"No rows found for n_samples={n_samples}")

    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    df["strategy"] = pd.Categorical(df["strategy"], categories=STRATEGY_ORDER, ordered=True)
    df = df.sort_values(["mode", "strategy"])

    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    x = np.arange(len(MODE_ORDER))
    width = 0.20

    for i, strategy in enumerate(STRATEGY_ORDER):
        values = []
        for mode in MODE_ORDER:
            row = df[(df["mode"] == mode) & (df["strategy"] == strategy)]
            values.append(float(row["val_recall@50"].iloc[0]) if len(row) else np.nan)

        ax.bar(
            x + (i - 1.5) * width,
            values,
            width=width,
            label=STRATEGY_LABELS[strategy],
        )

    ax.set_title(f"Retrieval comparison by pair-construction strategy ({n_samples} samples)")
    ax.set_ylabel("Validation Recall@50")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "retrieval_strategy_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def infer_pair_columns(pair_df: pd.DataFrame) -> tuple[str, str]:
    possible_a = ["a_sample_id", "sample_id_a", "modality_a_id", "a_id", "source_id"]
    possible_b = ["b_sample_id", "sample_id_b", "modality_b_id", "b_id", "target_id"]

    a_col = next((c for c in possible_a if c in pair_df.columns), None)
    b_col = next((c for c in possible_b if c in pair_df.columns), None)

    if a_col is not None and b_col is not None:
        return a_col, b_col

    # Fallback for simple pair files with first two integer columns.
    numeric_cols = [
        c for c in pair_df.columns
        if pd.api.types.is_numeric_dtype(pair_df[c])
    ]

    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]

    raise RuntimeError(
        f"Could not infer pair id columns from: {list(pair_df.columns)}"
    )


def plot_pairing_space(
    strategy: str = "propensity_weighted",
    max_points: int = 2000,
    max_lines: int = 120,
    seed: int = 42,
) -> None:
    a_df = pd.read_csv(DATA_DIR / "modality_a.csv")
    b_df = pd.read_csv(DATA_DIR / "modality_b.csv")
    pair_df = load_pair_file(strategy)

    a_cols = feature_columns(a_df, ["a_feat_", "feat_", "modality_a_", "a_"])
    b_cols = feature_columns(b_df, ["b_feat_", "feat_", "modality_b_", "b_"])

    a_x = a_df[a_cols].astype(np.float32).to_numpy()
    b_x = b_df[b_cols].astype(np.float32).to_numpy()

    n = min(len(a_df), len(b_df), max_points)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(min(len(a_df), len(b_df)), size=n, replace=False))

    a_sample = a_df.iloc[idx].reset_index(drop=True)
    b_sample = b_df.iloc[idx].reset_index(drop=True)

    a_x_sample = a_x[idx]
    b_x_sample = b_x[idx]

    combined = np.vstack([a_x_sample, b_x_sample])
    coords = PCA(n_components=2, random_state=42).fit_transform(combined)

    a_z = coords[: len(a_x_sample)]
    b_z = coords[len(a_x_sample) :]

    a_pair_col, b_pair_col = infer_pair_columns(pair_df)

    # Detect whether the pair files use 0-based or 1-based IDs.
    pair_min_id = min(
        int(pair_df[a_pair_col].min()),
        int(pair_df[b_pair_col].min()),
    )
    one_based_ids = pair_min_id >= 1

    # Build IDs before resetting/sampling changes the visible row positions.
    a_full_ids = get_id_series(a_df, one_based=one_based_ids)
    b_full_ids = get_id_series(b_df, one_based=one_based_ids)

    a_ids = a_full_ids.iloc[idx].astype(int).tolist()
    b_ids = b_full_ids.iloc[idx].astype(int).tolist()

    a_id_to_pos = {int(sid): i for i, sid in enumerate(a_ids)}
    b_id_to_pos = {int(sid): i for i, sid in enumerate(b_ids)}


    eligible_pairs = pair_df[
        pair_df[a_pair_col].astype(int).isin(a_id_to_pos)
        & pair_df[b_pair_col].astype(int).isin(b_id_to_pos)
    ].copy()

    if len(eligible_pairs) > max_lines:
        eligible_pairs = eligible_pairs.sample(n=max_lines, random_state=seed)

    fig, ax = plt.subplots(figsize=(8.0, 6.2))

    if "group_id" in a_sample.columns:
        colors = a_sample["group_id"].astype(int).to_numpy()
    elif "group" in a_sample.columns:
        colors = a_sample["group"].astype(int).to_numpy()
    else:
        colors = np.arange(len(a_sample)) % 20

    ax.scatter(
        a_z[:, 0],
        a_z[:, 1],
        c=colors,
        cmap="tab20",
        s=8,
        alpha=0.70,
        marker="o",
        linewidths=0,
        label="Modality A",
    )

    ax.scatter(
        b_z[:, 0],
        b_z[:, 1],
        c=colors,
        cmap="tab20",
        s=9,
        alpha=0.70,
        marker="x",
        linewidths=0.45,
        label="Modality B",
    )

    for _, row in eligible_pairs.iterrows():
        a_id = int(row[a_pair_col])
        b_id = int(row[b_pair_col])

        ai = a_id_to_pos.get(a_id)
        bi = b_id_to_pos.get(b_id)

        if ai is None or bi is None:
            continue

        ax.plot(
            [a_z[ai, 0], b_z[bi, 0]],
            [a_z[ai, 1], b_z[bi, 1]],
            alpha=0.12,
            linewidth=0.6,
        )

    ax.set_title(f"Constructed pairs: {STRATEGY_LABELS.get(strategy, strategy)}")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "propensity_pairing_space.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_pair_quality(n_samples=24000)
    plot_retrieval_comparison(n_samples=24000)
    plot_pairing_space(strategy="propensity_weighted")

    print("Saved figures:")
    print(FIGURES_DIR / "pair_quality_comparison.png")
    print(FIGURES_DIR / "retrieval_strategy_comparison.png")
    print(FIGURES_DIR / "propensity_pairing_space.png")


if __name__ == "__main__":
    main()