from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


UnlearningResult = Dict[int, torch.nn.Module]


def do_nothing(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: Dict,
    epochs: List[int],
    **_: object,
) -> UnlearningResult:
    """Baseline: return the original model unchanged."""
    return {max(epochs): deepcopy(model)}


def ascent_forget(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: Dict,
    epochs: List[int],
    loss_fn=F.cross_entropy,
    device: str = "cuda",
    **_: object,
) -> UnlearningResult:
    """Gradient ascent on the forget set."""
    model = model.to(device).train()
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    epoch_models: UnlearningResult = {}

    for it in range(1, max(epochs) + 1):
        for x, y in forget_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()
        if it in epochs:
            epoch_models[it] = deepcopy(model)
    return epoch_models


def ascent_descent(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: Dict,
    epochs: List[int],
    loss_fn=F.cross_entropy,
    device: str = "cuda",
    ascent_epochs: int = 1,
    **_: object,
) -> UnlearningResult:
    """Ascent on forget set, then descent on retain set each epoch."""
    model = model.to(device).train()
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    epoch_models: UnlearningResult = {}

    for it in range(1, max(epochs) + 1):
        if it <= ascent_epochs:
            for x, y in forget_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                optimizer.zero_grad()
                (-loss).backward()
                optimizer.step()
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if it in epochs:
            epoch_models[it] = deepcopy(model)
    return epoch_models


def noisy_descent(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: Dict,
    epochs: List[int],
    device: str = "cuda",
    noise_std: float = 0.01,
    max_grad_norm: float = 1.0,
    **_: object,
) -> UnlearningResult:
    """Noisy SGD on the retain set only (DP-SGD style).

    - Uses cross-entropy loss on retain batches.
    - Clips gradients to `max_grad_norm`.
    - Adds Gaussian noise N(0, noise_std^2) to each gradient tensor.
    """
    model = model.to(device).train()
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    epoch_models: UnlearningResult = {}

    for it in range(1, max(epochs) + 1):
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            optimizer.zero_grad()
            loss.backward()

            # Global gradient clipping.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Add Gaussian noise to gradients.
            for p in model.parameters():
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.grad) * noise_std
                p.grad.add_(noise)

            optimizer.step()

        if it in epochs:
            epoch_models[it] = deepcopy(model)

    return epoch_models


UNLEARNING_METHODS = {
    "do_nothing": do_nothing,
    "ascent_forget": ascent_forget,
    "ascent_descent": ascent_descent,
    "noisy_descent": noisy_descent,
}

OPTIMIZERS = {"sgd": torch.optim.SGD}

