from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from easydub.data import (
    ensure_dataset,
    get_cifar_dataloader,
    load_forget_indices,
    load_margins_array,
    load_pretrain_checkpoints,
)
from easydub.eval import compute_margins, klom_from_margins
from easydub.unlearning import OPTIMIZERS, UNLEARNING_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EasyDUB unlearning + KLOM demo.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to EasyDUB-dataset root. If omitted, downloads from HuggingFace automatically.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="noisy_descent",
        choices=sorted(UNLEARNING_METHODS.keys()),
        help="Unlearning method to run.",
    )
    parser.add_argument(
        "--forget-set",
        type=int,
        default=1,
        help="Forget set ID (1–10).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the unlearning optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of unlearning epochs per model.",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=10,
        help="Number of pretrain models to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g. cuda or cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = ensure_dataset(args.data_dir)
    device = args.device

    print(f"Using data root: {data_root}")
    print(f"Using device:    {device}")
    print(f"Method:          {args.method}")
    print(f"Forget set:      {args.forget_set}")
    print(f"Models:          {args.n_models}")
    print(f"Epochs:          {args.epochs}")

    # Load CIFAR-10 loaders for train/val
    cifar_root = data_root / "data"
    train_loader = get_cifar_dataloader(cifar_root, split="train", shuffle=False, batch_size=256)
    val_loader = get_cifar_dataloader(cifar_root, split="val", shuffle=False, batch_size=256)

    # Forget / retain indices for this forget set
    forget_indices = load_forget_indices(data_root, args.forget_set)
    retain_indices = np.setdiff1d(np.arange(50_000, dtype=np.int64), forget_indices)

    forget_loader = get_cifar_dataloader(
        cifar_root,
        split="train",
        indices=forget_indices.tolist(),
        shuffle=True,
        batch_size=128,
    )
    retain_loader = get_cifar_dataloader(
        cifar_root,
        split="train",
        indices=retain_indices.tolist(),
        shuffle=True,
        batch_size=256,
    )

    # Load pretrain models
    pretrain_models = load_pretrain_checkpoints(data_root, n_models=args.n_models, device=device)

    # Load oracle margins for this forget set (train + val concatenated)
    oracle_margins_list = []
    for model_id in range(len(pretrain_models)):
        m_retain = load_margins_array(
            data_root,
            kind="oracle",
            phase="retain",
            forget_id=args.forget_set,
            model_id=model_id,
        )
        m_forget = load_margins_array(
            data_root,
            kind="oracle",
            phase="forget",
            forget_id=args.forget_set,
            model_id=model_id,
        )
        m_val = load_margins_array(
            data_root,
            kind="oracle",
            phase="val",
            forget_id=args.forget_set,
            model_id=model_id,
        )
        oracle_margins_list.append(np.concatenate([m_retain, m_forget, m_val], axis=0))
    oracle_margins = torch.from_numpy(np.stack(oracle_margins_list, axis=0))

    # Run unlearning and compute margins
    method_fn = UNLEARNING_METHODS[args.method]
    optimizer_cls = OPTIMIZERS["sgd"]

    unlearned_margins_list = []
    for model_idx, base_model in enumerate(pretrain_models):
        epoch_models = method_fn(
            base_model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs={"lr": args.lr},
            epochs=[args.epochs],
            device=device,
            ascent_epochs=max(1, args.epochs // 2),
        )
        unlearned_model = epoch_models[args.epochs]

        # Compute margins on retain / forget / val splits
        retain_loader_eval = get_cifar_dataloader(
            cifar_root,
            split="train",
            indices=retain_indices.tolist(),
            shuffle=False,
            batch_size=256,
        )
        forget_loader_eval = get_cifar_dataloader(
            cifar_root,
            split="train",
            indices=forget_indices.tolist(),
            shuffle=False,
            batch_size=256,
        )

        margins_retain = compute_margins(unlearned_model, retain_loader_eval, device=device)
        margins_forget = compute_margins(unlearned_model, forget_loader_eval, device=device)
        margins_val = compute_margins(unlearned_model, val_loader, device=device)

        margins_all = torch.cat([margins_retain, margins_forget, margins_val], dim=0)
        unlearned_margins_list.append(margins_all)

    unlearned_margins = torch.stack(unlearned_margins_list, dim=0)

    # Compute per-sample KLOM scores
    klom_scores = klom_from_margins(oracle_margins, unlearned_margins)

    print()
    print("KLOM summary (lower is better, closer to oracle):")
    print(f"  mean:  {klom_scores.mean():.4f}")
    print(f"  std:   {klom_scores.std():.4f}")
    print(f"  min:   {klom_scores.min():.4f}")
    print(f"  max:   {klom_scores.max():.4f}")


if __name__ == "__main__":
    main()

