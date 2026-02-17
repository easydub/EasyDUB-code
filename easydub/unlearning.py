from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List

import numpy as np
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


def _adjust_learning_rate(
    epoch: int,
    lr_decay_epochs: Iterable[int],
    lr_decay_rate: float,
    base_lr: float,
    optimizer: torch.optim.Optimizer,
) -> float:
    steps = np.sum(epoch > np.asarray(list(lr_decay_epochs)))
    new_lr = base_lr * (lr_decay_rate ** steps) if steps > 0 else base_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def _distill_kl_loss(y_s: torch.Tensor, y_t: torch.Tensor, temperature: float) -> torch.Tensor:
    p_s = F.log_softmax(y_s / temperature, dim=1)
    p_t = F.softmax(y_t / temperature, dim=1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (temperature ** 2)


def scrub(
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
    """SCRUB-style unlearning: KL ascent on forget set, then KL + CE descent on retain."""
    gamma = 0.99
    alpha = 0.1
    lr_decay_epochs = [3, 5, 9]
    lr_decay_rate = 0.1
    kd_T = 4.0

    teacher = model.to(device).eval()
    student = deepcopy(model).to(device).train()
    optimizer = optimizer_cls(student.parameters(), **optimizer_kwargs)
    epoch_models: UnlearningResult = {}

    for epoch in range(1, max(epochs) + 1):
        _adjust_learning_rate(
            epoch,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_rate=lr_decay_rate,
            base_lr=optimizer_kwargs["lr"],
            optimizer=optimizer,
        )

        if epoch <= ascent_epochs:
            for x, y in forget_loader:
                x, y = x.to(device), y.to(device)
                logit_s = student(x)
                with torch.no_grad():
                    logit_t = teacher(x)
                loss = -_distill_kl_loss(logit_s, logit_t, kd_T)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            logit_s = student(x)
            with torch.no_grad():
                logit_t = teacher(x)
            loss_cls = loss_fn(logit_s, y)
            loss_div = _distill_kl_loss(logit_s, logit_t, kd_T)
            loss = gamma * loss_cls + alpha * loss_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch in epochs:
            epoch_models[epoch] = deepcopy(student)

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
    "scrub": scrub,
    "noisy_descent": noisy_descent,
}

OPTIMIZERS = {"sgd": torch.optim.SGD}

