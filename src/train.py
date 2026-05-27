from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metrics import retrieval_metrics
from model import DualEncoder


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.10
    hidden_dim: int = 256
    embed_dim: int = 128
    dropout: float = 0.10
    val_fraction: float = 0.20
    k_values: tuple[int, ...] = (1, 5, 10, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dual encoder using pairs produced by a selected matching strategy."
    )

    parser.add_argument("--a-csv", type=Path, default=Path("data_demo/modality_a.csv"))
    parser.add_argument("--b-csv", type=Path, default=Path("data_demo/modality_b.csv"))
    parser.add_argument("--pairs-csv", type=Path, default=Path("data_demo/train_pairs.csv"))
    parser.add_argument("--true-pairs-csv", type=Path, default=Path("data_demo/true_pairs.csv"))

    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)

    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairTrainingDataset(Dataset):
    def __init__(
        self,
        pair_df: pd.DataFrame,
        a_df: pd.DataFrame,
        b_df: pd.DataFrame,
        a_cols: list[str],
        b_cols: list[str],
    ):
        self.pair_df = pair_df.reset_index(drop=True)

        self.a_features = {
            int(row["a_id"]): row[a_cols].to_numpy(dtype=np.float32)
            for _, row in a_df.iterrows()
        }
        self.b_features = {
            int(row["b_id"]): row[b_cols].to_numpy(dtype=np.float32)
            for _, row in b_df.iterrows()
        }

    def __len__(self) -> int:
        return len(self.pair_df)

    def __getitem__(self, idx: int):
        row = self.pair_df.iloc[idx]
        a_id = int(row["a_id"])
        b_id = int(row["b_id"])
        pair_score = float(row.get("pair_score", 1.0))

        return (
            torch.from_numpy(self.a_features[a_id].copy()),
            torch.from_numpy(self.b_features[b_id].copy()),
            torch.tensor(pair_score, dtype=torch.float32),
            a_id,
            b_id,
        )


class EvalFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id_col: str, feature_cols: list[str]):
        self.ids = df[id_col].astype(int).to_numpy()
        self.features = df[feature_cols].astype(np.float32).to_numpy()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.features[idx].copy()), int(self.ids[idx])


def collate_train(batch):
    x_a = torch.stack([item[0] for item in batch])
    x_b = torch.stack([item[1] for item in batch])
    pair_scores = torch.stack([item[2] for item in batch])
    a_ids = torch.tensor([item[3] for item in batch], dtype=torch.long)
    b_ids = torch.tensor([item[4] for item in batch], dtype=torch.long)
    return x_a, x_b, pair_scores, a_ids, b_ids


def collate_eval(batch):
    x = torch.stack([item[0] for item in batch])
    ids = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return x, ids


def split_a_ids(a_ids: np.ndarray, seed: int, val_fraction: float) -> tuple[set[int], set[int]]:
    rng = np.random.default_rng(seed)
    ids = a_ids.copy()
    rng.shuffle(ids)

    n_val = max(1, int(round(len(ids) * val_fraction)))
    val_ids = set(int(x) for x in ids[:n_val])
    train_ids = set(int(x) for x in ids[n_val:])

    return train_ids, val_ids


def weighted_symmetric_infonce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    pair_scores: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Symmetric InfoNCE where each positive pair can be weighted by pair confidence.

    This allows propensity-weighted pairs to contribute more when confidence is high.
    """
    batch_size = z_a.size(0)
    labels = torch.arange(batch_size, device=z_a.device)

    logits = z_a @ z_b.T / temperature

    loss_a = F.cross_entropy(logits, labels, reduction="none")
    loss_b = F.cross_entropy(logits.T, labels, reduction="none")
    loss_per_sample = 0.5 * (loss_a + loss_b)

    weights = pair_scores.to(z_a.device).clamp(min=0.05, max=1.0)
    loss = (loss_per_sample * weights).sum() / (weights.sum() + 1e-8)

    with torch.no_grad():
        sim = z_a @ z_b.T
        eye = torch.eye(batch_size, dtype=torch.bool, device=z_a.device)
        pos_sim = sim.diagonal().mean().item()
        neg_sim = sim[~eye].mean().item()

    return loss, {"pos_sim": pos_sim, "neg_sim": neg_sim}


@torch.no_grad()
def embed_a(model: DualEncoder, loader: DataLoader, device: torch.device):
    model.eval()
    embeds, ids = [], []

    for x, batch_ids in loader:
        x = x.to(device)
        z = model.encode_a(x)
        embeds.append(z.cpu())
        ids.extend(batch_ids.numpy().tolist())

    return torch.cat(embeds), np.asarray(ids, dtype=np.int64)


@torch.no_grad()
def embed_b(model: DualEncoder, loader: DataLoader, device: torch.device):
    model.eval()
    embeds, ids = [], []

    for x, batch_ids in loader:
        x = x.to(device)
        z = model.encode_b(x)
        embeds.append(z.cpu())
        ids.extend(batch_ids.numpy().tolist())

    return torch.cat(embeds), np.asarray(ids, dtype=np.int64)


def evaluate(
    model: DualEncoder,
    a_loader: DataLoader,
    b_loader: DataLoader,
    true_b_for_a: dict[int, int],
    device: torch.device,
    k_values: tuple[int, ...],
) -> dict[str, float]:
    z_a, a_ids = embed_a(model, a_loader, device)
    z_b, b_ids = embed_b(model, b_loader, device)

    return retrieval_metrics(
        z_a=z_a,
        z_b=z_b,
        a_ids=a_ids,
        b_ids=b_ids,
        true_b_for_a=true_b_for_a,
        k_values=k_values,
    )


def train_one_epoch(
    model: DualEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    steps = 0

    for x_a, x_b, pair_scores, _a_ids, _b_ids in tqdm(loader, desc="train", leave=False):
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        pair_scores = pair_scores.to(device)

        optimizer.zero_grad(set_to_none=True)

        z_a, z_b = model(x_a, x_b)
        loss, metrics = weighted_symmetric_infonce(
            z_a=z_a,
            z_b=z_b,
            pair_scores=pair_scores,
            temperature=temperature,
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_pos_sim += metrics["pos_sim"]
        total_neg_sim += metrics["neg_sim"]
        steps += 1

    return {
        "train_loss": total_loss / max(steps, 1),
        "train_pos_sim": total_pos_sim / max(steps, 1),
        "train_neg_sim": total_neg_sim / max(steps, 1),
    }


def save_checkpoint(path: Path, model: DualEncoder, optimizer, epoch: int, row: dict, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": row,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    a_path = resolve_path(args.a_csv, repo_root)
    b_path = resolve_path(args.b_csv, repo_root)
    pairs_path = resolve_path(args.pairs_csv, repo_root)
    true_pairs_path = resolve_path(args.true_pairs_csv, repo_root)

    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)
    pair_df = pd.read_csv(pairs_path)
    true_pairs = pd.read_csv(true_pairs_path)

    strategy = pair_df["pair_strategy"].iloc[0] if "pair_strategy" in pair_df.columns else "unknown"

    cfg = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
    )

    set_seed(cfg.seed)

    a_cols = sorted([c for c in a_df.columns if c.startswith("a_feat_")])
    b_cols = sorted([c for c in b_df.columns if c.startswith("b_feat_")])

    if len(a_cols) == 0 or len(a_cols) != len(b_cols):
        raise RuntimeError("Feature column mismatch between modality A and B.")

    all_a_ids = a_df["a_id"].astype(int).to_numpy()
    train_ids, val_ids = split_a_ids(all_a_ids, seed=cfg.seed, val_fraction=cfg.val_fraction)

    train_pairs = pair_df[pair_df["a_id"].astype(int).isin(train_ids)].reset_index(drop=True)

    val_a_df = a_df[a_df["a_id"].astype(int).isin(val_ids)].reset_index(drop=True)
    val_b_df = b_df[b_df["b_id"].astype(int).isin(val_ids)].reset_index(drop=True)

    true_b_for_a = {
        int(row["a_id"]): int(row["b_id"])
        for _, row in true_pairs[true_pairs["a_id"].astype(int).isin(val_ids)].iterrows()
    }

    run_name = f"{strategy}_{len(a_df)}"
    output_dir = args.output_dir or (repo_root / "outputs" / run_name)
    checkpoint_dir = args.checkpoint_dir or (repo_root / "checkpoints" / run_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairTrainingDataset(train_pairs, a_df, b_df, a_cols, b_cols)
    val_a_ds = EvalFeatureDataset(val_a_df, "a_id", a_cols)
    val_b_ds = EvalFeatureDataset(val_b_df, "b_id", b_cols)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_train)
    val_a_loader = DataLoader(val_a_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_eval)
    val_b_loader = DataLoader(val_b_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_eval)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualEncoder(
        input_dim=len(a_cols),
        hidden_dim=cfg.hidden_dim,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    pair_precision_true = float(pair_df["is_true_pair"].mean()) if "is_true_pair" in pair_df.columns else float("nan")
    pair_precision_group = float(pair_df["is_same_group"].mean()) if "is_same_group" in pair_df.columns else float("nan")

    config = {
        **asdict(cfg),
        "strategy": strategy,
        "n_samples": int(len(a_df)),
        "train_pairs": int(len(train_pairs)),
        "val_size": int(len(val_a_df)),
        "pair_precision_true": pair_precision_true,
        "pair_precision_group": pair_precision_group,
        "device": str(device),
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Strategy             : {strategy}")
    print(f"Samples              : {len(a_df)}")
    print(f"Train pairs          : {len(train_pairs)}")
    print(f"Val size             : {len(val_a_df)}")
    print(f"Pair precision true  : {pair_precision_true:.4f}")
    print(f"Pair precision group : {pair_precision_group:.4f}")
    print(f"Device               : {device}")
    print(f"Output dir           : {output_dir}")

    history = []
    metrics_path = output_dir / "metrics.csv"
    best_lift50 = -math.inf

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=cfg.temperature,
        )

        val_metrics = evaluate(
            model=model,
            a_loader=val_a_loader,
            b_loader=val_b_loader,
            true_b_for_a=true_b_for_a,
            device=device,
            k_values=cfg.k_values,
        )

        row = {
            "epoch": float(epoch),
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "pair_precision_true": pair_precision_true,
            "pair_precision_group": pair_precision_group,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False)

        print(
            f"loss={row['train_loss']:.4f} "
            f"R@10={row['val_recall@10']:.4f} "
            f"R@50={row['val_recall@50']:.4f} "
            f"Lift@50={row['val_lift@50']:.2f}x "
            f"pos_sim={row['val_pos_sim_mean']:.4f}"
        )

        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, row, config)

        if row["val_lift@50"] > best_lift50:
            best_lift50 = row["val_lift@50"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row, config)

    print("\nDone.")
    print(f"Best Lift@50 : {best_lift50:.2f}x")
    print(f"Metrics      : {metrics_path}")


if __name__ == "__main__":
    main()