"""Evaluate pretrain models against oracle baselines using KLOM.

Prints a P95 KLOM table for retain/forget/val splits, measuring how far
pretrain model margins are from oracle (retrained-from-scratch) margins.

Usage:
    python eval_pretrain.py --data-dir /path/to/EasyDUB-dataset
    python eval_pretrain.py --data-dir /path/to/EasyDUB-dataset --forget-set 2 --n-models 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from easydub.data import ensure_dataset, load_forget_indices, retain_mask
from easydub.eval import klom_from_margins, load_margins_matrix

SPLITS = ["retain", "forget", "val"]


def main():
    p = argparse.ArgumentParser(description="Pretrain vs oracle KLOM evaluation")
    p.add_argument("--data-dir", type=str, required=True, help="Path to EasyDUB-dataset")
    p.add_argument("--forget-set", type=int, default=2, help="Forget set ID")
    p.add_argument("--n-models", type=int, default=100, help="Number of models")
    args = p.parse_args()

    data_root = ensure_dataset(Path(args.data_dir))
    forget_idx = load_forget_indices(data_root, args.forget_set)

    print(f"Loading margins for {args.n_models} models, forget set {args.forget_set}...")
    print()

    print(f"{'Split':<12} {'P95 KLOM':>10}")
    print("-" * 24)
    for split in SPLITS:
        oracle = load_margins_matrix(data_root, "oracle", split, args.forget_set, args.n_models)
        fid = args.forget_set if split == "forget" else None
        pretrain = load_margins_matrix(data_root, "pretrain", split, fid, args.n_models)

        # Pretrain retain has 50k samples; oracle retain has 49.9k (forget removed).
        # Mask out forget indices to align shapes.
        if split == "retain" and pretrain.shape[1] > oracle.shape[1]:
            pretrain = pretrain[:, retain_mask(pretrain.shape[1], forget_idx)]

        klom = klom_from_margins(oracle, pretrain)
        print(f"{split:<12} {np.percentile(klom, 95):>10.4f}")

    print()


if __name__ == "__main__":
    main()
