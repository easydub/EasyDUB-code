from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


def binned_kl_divergence(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bin_count: int = 20,
    eps: float = 1e-5,
    min_val: float = -100.0,
    max_val: float = 100.0,
) -> float:
    """KL(p || q) between two 1D sample sets using shared-bin histograms.

    Bin edges are computed from the combined range of both p and q, matching
    the reference implementation in data-unlearning-bench.
    """
    p_arr = np.clip(p_samples, min_val, max_val)
    q_arr = np.clip(q_samples, min_val, max_val)

    bins_start = min(p_arr.min(), q_arr.min())
    bins_end = max(p_arr.max(), q_arr.max())
    if bins_start >= bins_end:
        bins_end = bins_start + 1.0
    edges = np.linspace(bins_start, bins_end, bin_count + 1)

    p_indices = np.digitize(p_arr, edges)
    q_indices = np.digitize(q_arr, edges)
    p_counts = np.array([np.sum(p_indices == i) for i in range(1, bin_count + 1)])
    q_counts = np.array([np.sum(q_indices == i) for i in range(1, bin_count + 1)])

    p_total = p_counts.sum()
    q_total = q_counts.sum()
    p_probs = p_counts / p_total if p_total > 0 else np.zeros_like(p_counts, dtype=float)
    q_probs = q_counts / q_total if q_total > 0 else np.zeros_like(q_counts, dtype=float)

    q_safe = np.where(p_probs > 0.0, np.maximum(q_probs, eps), q_probs)

    mask = p_probs > 0.0
    kl = np.sum(p_probs[mask] * np.log(p_probs[mask] / q_safe[mask]))
    return float(kl)


def klom_from_margins(
    oracle_margins: torch.Tensor,
    unlearned_margins: torch.Tensor,
    clip_min: float = -100.0,
    clip_max: float = 100.0,
) -> np.ndarray:
    """Compute KLOM scores per sample from oracle and unlearned margins.

    Both inputs should have shape (n_models, n_samples).
    Returns an array of shape (n_samples,) with per-sample KL divergences.
    """
    assert oracle_margins.shape == unlearned_margins.shape

    n_samples = oracle_margins.shape[1]
    results = []
    for idx in tqdm(range(n_samples), desc="KLOM", leave=False):
        oracle_arr = oracle_margins[:, idx].cpu().numpy()
        unlearned_arr = unlearned_margins[:, idx].cpu().numpy()
        kl = binned_kl_divergence(
            p_samples=unlearned_arr,
            q_samples=oracle_arr,
            min_val=clip_min,
            max_val=clip_max,
        )
        results.append(kl)
    return np.asarray(results)


def load_margins_matrix(
    data_root: Path,
    kind: str,
    split: str,
    forget_id: Optional[int],
    n_models: int,
) -> torch.Tensor:
    """Load (n_models, n_samples) margins matrix.

    Calls load_margins_array for each model ID and stacks into a single tensor.

    Args:
        data_root: Path to EasyDUB dataset root.
        kind: "pretrain" or "oracle".
        split: "retain", "forget", or "val".
        forget_id: Forget set ID (required for oracle and pretrain/forget).
        n_models: Number of models to load.
    """
    from .data import load_margins_array

    arrays = [
        load_margins_array(data_root, kind, split, forget_id, mid)
        for mid in range(n_models)
    ]
    return torch.from_numpy(np.stack(arrays))

