from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar_dataloader(
    data_root: Path,
    split: str = "train",
    indices: Optional[Sequence[int]] = None,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Return a CIFAR-10 dataloader with the standard normalization.

    `split` is one of {"train", "val", "all"}.
    If `indices` is provided, it is applied to the train split.
    """
    assert split in {"train", "val", "all"}
    if indices is not None:
        assert split == "train", "indices are only supported for the train split"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    if split == "all":
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        val_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        dataset = torch.utils.data.ConcatDataset([train_ds, val_ds])
    else:
        dataset = datasets.CIFAR10(
            root=data_root,
            train=(split == "train"),
            download=True,
            transform=transform,
        )

    if indices is not None:
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def list_pretrain_model_paths(dataset_root: Path) -> List[Path]:
    """List all pretrain ResNet9 checkpoints in EasyDUB-dataset."""
    model_dir = dataset_root / "models" / "cifar10" / "pretrain" / "resnet9"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Pretrain model directory not found: {model_dir}")
    return sorted(model_dir.glob("id_*_epoch_23.pt"))


def load_pretrain_checkpoints(
    dataset_root: Path,
    n_models: Optional[int] = None,
    device: str = "cpu",
) -> List[torch.nn.Module]:
    """Load up to `n_models` pretrain checkpoints as ready-to-use models.

    Models are loaded on `device`. This function only wraps the state dict;
    construction of the architecture is left to the caller.
    """
    from .models import resnet9

    paths = list_pretrain_model_paths(dataset_root)
    if n_models is not None:
        paths = paths[:n_models]

    models: List[torch.nn.Module] = []
    for p in paths:
        state = torch.load(p, map_location=device, weights_only=True)
        # Older checkpoints may have a "model." prefix; strip it if present.
        if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            cleaned = {
                k.removeprefix("model.").removeprefix("module."): v for k, v in state.items()
            }
        else:
            cleaned = state
        m = resnet9(num_classes=10)
        m.load_state_dict(cleaned, strict=True)
        m.to(device)
        models.append(m)
    return models


def load_forget_indices(dataset_root: Path, forget_id: int) -> np.ndarray:
    path = dataset_root / "forget_sets" / "cifar10" / f"forget_set_{forget_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    indices = np.load(path)
    return indices.astype(np.int64)


def load_margins_array(
    dataset_root: Path,
    kind: str,
    phase: str,
    forget_id: Optional[int],
    model_id: int,
) -> np.ndarray:
    """Load a single margins array from disk.

    kind  in {"pretrain", "oracle"}
    phase depends on kind:
      - pretrain: {"retain", "val", "forget"}
      - oracle:   {"retain", "val", "forget"}
    forget_id is required when kind == "oracle" or phase == "forget".
    """
    assert kind in {"pretrain", "oracle"}
    assert phase in {"retain", "val", "forget"}

    base = dataset_root / "margins" / "cifar10"

    if kind == "pretrain":
        if phase == "forget":
            if forget_id is None:
                raise ValueError("forget_id is required for pretrain/forget margins")
            path = (
                base
                / "pretrain"
                / f"forget_{forget_id}"
                / "resnet9"
                / f"id_{model_id}_epoch_23.npy"
            )
        else:
            path = base / "pretrain" / phase / "resnet9" / f"id_{model_id}_epoch_23.npy"
    else:
        if forget_id is None:
            raise ValueError("forget_id is required for oracle margins")
        path = (
            base
            / "oracle"
            / f"forget_{forget_id}"
            / phase
            / "resnet9"
            / f"id_{model_id}_epoch_23.npy"
        )

    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)

