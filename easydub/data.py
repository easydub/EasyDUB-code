from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

DATASET_REPO = "easydub/EasyDUB-dataset"
DEFAULT_CACHE = Path.home() / ".cache" / "easydub" / "EasyDUB-dataset"


def ensure_dataset(data_dir: Optional[Path] = None) -> Path:
    """Return a local path to the EasyDUB dataset, downloading from HuggingFace if needed."""
    path = Path(data_dir) if data_dir is not None else DEFAULT_CACHE
    if (path / "margins").is_dir():
        return path
    print(f"Downloading EasyDUB dataset to {path} ...")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=str(path))
    if not (path / "margins").is_dir():
        raise RuntimeError(
            f"Dataset download incomplete: {path / 'margins'} not found. "
            f"Please download manually from "
            f"https://huggingface.co/datasets/{DATASET_REPO}"
        )
    return path


def retain_mask(n_total: int, forget_indices: np.ndarray) -> np.ndarray:
    """Return a boolean mask that is True for retain (non-forget) indices."""
    mask = np.ones(n_total, dtype=bool)
    mask[forget_indices] = False
    return mask


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

