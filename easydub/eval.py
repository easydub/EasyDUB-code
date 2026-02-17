from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_margins(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute per-sample margins for all examples in `loader`.

    Margin is defined as:
        margin = logit_correct - log_sum_exp(logit_other)
    """
    model = model.to(device).eval()
    all_margins = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            batch_indices = torch.arange(logits.size(0), device=device)
            logits_correct = logits[batch_indices, y]
            masked = logits.clone()
            masked[batch_indices, y] = -torch.inf
            lse_other = masked.logsumexp(dim=-1)
            margins = logits_correct - lse_other
            all_margins.append(margins.cpu())
    return torch.cat(all_margins, dim=0)


def _histogram_probs(
    values: np.ndarray,
    bin_count: int = 20,
    min_val: float = -100.0,
    max_val: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(values, min_val, max_val)
    bins = np.linspace(clipped.min(), clipped.max(), bin_count + 1)
    counts, edges = np.histogram(clipped, bins=bins)
    total = counts.sum()
    if total == 0:
        probs = np.zeros_like(counts, dtype=float)
    else:
        probs = counts.astype(float) / float(total)
    return probs, edges


def binned_kl_divergence(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bin_count: int = 20,
    eps: float = 1e-5,
    min_val: float = -100.0,
    max_val: float = 100.0,
) -> float:
    """KL(p || q) between two 1D sample sets using binning."""
    p_probs, edges = _histogram_probs(p_samples, bin_count, min_val, max_val)
    q_probs, _ = _histogram_probs(q_samples, bin_count, min_val, max_val)

    # Ensure q has support wherever p has support
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

