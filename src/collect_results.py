from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final metrics from pairing-strategy experiments.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out-csv", type=Path, default=Path("experiments/results_table.csv"))
    return parser.parse_args()


def parse_run_name(run_name: str) -> dict[str, str]:
    parts = run_name.split("_")

    info = {
        "run_name": run_name,
        "mode": "unknown",
        "n_samples": "unknown",
        "strategy": "unknown",
    }

    if len(parts) >= 3:
        info["mode"] = parts[0]

        for part in parts:
            if part.isdigit():
                info["n_samples"] = part

        if "true" in parts:
            info["strategy"] = "true_pair"
        elif "random" in parts:
            info["strategy"] = "random"
        elif "metadata" in parts:
            info["strategy"] = "metadata_similarity"
        elif "propensity" in parts:
            info["strategy"] = "propensity_weighted"

    return info


def main() -> None:
    args = parse_args()

    rows = []

    for metrics_path in sorted(args.outputs_dir.glob("*/metrics.csv")):
        run_dir = metrics_path.parent
        metrics_df = pd.read_csv(metrics_path)

        if metrics_df.empty:
            continue

        final = metrics_df.iloc[-1].to_dict()
        parsed = parse_run_name(run_dir.name)

        config_path = run_dir / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        row = {
            **parsed,
            "epoch": final.get("epoch"),
            "train_loss": final.get("train_loss"),
            "train_pos_sim": final.get("train_pos_sim"),
            "train_neg_sim": final.get("train_neg_sim"),
            "val_recall@1": final.get("val_recall@1"),
            "val_recall@5": final.get("val_recall@5"),
            "val_recall@10": final.get("val_recall@10"),
            "val_recall@50": final.get("val_recall@50"),
            "val_lift@1": final.get("val_lift@1"),
            "val_lift@5": final.get("val_lift@5"),
            "val_lift@10": final.get("val_lift@10"),
            "val_lift@50": final.get("val_lift@50"),
            "val_pos_sim_mean": final.get("val_pos_sim_mean"),
            "val_n_pool": final.get("val_n_pool"),
            "pair_precision_true": final.get("pair_precision_true", config.get("pair_precision_true")),
            "pair_precision_group": final.get("pair_precision_group", config.get("pair_precision_group")),
        }

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No metrics.csv files found under {args.outputs_dir}")

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()