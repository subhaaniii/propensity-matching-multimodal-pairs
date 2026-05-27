from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def retrieval_metrics(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    a_ids: np.ndarray,
    b_ids: np.ndarray,
    true_b_for_a: dict[int, int],
    k_values: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, float]:
    """
    Evaluate A-to-B retrieval.

    A retrieval is correct if the true B id for an A id appears in the top K.
    """
    z_a = z_a.float().cpu()
    z_b = z_b.float().cpu()

    sim = (z_a @ z_b.T).numpy()
    n_a, n_b = sim.shape

    max_k = min(max(k_values), n_b)
    top_part = np.argpartition(sim, -max_k, axis=1)[:, -max_k:]
    top_scores = sim[np.arange(n_a)[:, None], top_part]
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    top_sorted = top_part[np.arange(n_a)[:, None], order]

    metrics: dict[str, float] = {}

    true_b = np.asarray([true_b_for_a[int(a_id)] for a_id in a_ids], dtype=np.int64)

    for k_req in k_values:
        k = min(k_req, n_b)
        retrieved_b_ids = b_ids[top_sorted[:, :k]]
        hit = (retrieved_b_ids == true_b[:, None]).any(axis=1)

        recall = float(hit.mean())
        random_recall = float(min(k / n_b, 1.0))

        metrics[f"recall@{k_req}"] = recall
        metrics[f"lift@{k_req}"] = (
            float(recall / random_recall) if random_recall > 0 else math.inf
        )

    b_id_to_idx = {int(b_id): idx for idx, b_id in enumerate(b_ids)}
    true_b_idx = np.asarray([b_id_to_idx[int(b_id)] for b_id in true_b], dtype=np.int64)

    pos_sim = sim[np.arange(n_a), true_b_idx]
    metrics["pos_sim_mean"] = float(pos_sim.mean())
    metrics["n_pool"] = float(n_b)

    return metrics