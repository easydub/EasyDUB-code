from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from easydub.data import (
    get_cifar_dataloader,
    load_forget_indices,
    load_margins_array,
    load_pretrain_checkpoints,
)
from easydub.eval import compute_margins, klom_from_margins
from easydub.unlearning import OPTIMIZERS, UNLEARNING_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strong noisy-SGD unlearning test: compare KLOM(pretrain, oracle) vs KLOM(noisy-descent, oracle)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to EasyDUB-dataset root (with models/, margins/, logits/, forget_sets/).",
    )
    parser.add_argument(
        "--forget-set",
        type=int,
        default=1,
        help="Forget set ID in [1, 10].",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=100,
        help="Number of pretrain models to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of noisy-descent epochs per model.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for noisy SGD.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Standard deviation of Gaussian noise added to gradients.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Global gradient clipping norm.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g. cuda or cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--recompute-pretrain-margins",
        action="store_true",
        help="If set, recompute pretrain margins from checkpoints instead of loading from disk.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = args.data_dir
    device = args.device

    print("Strong noisy-SGD unlearning test")
    print(f"  data_dir        = {data_root}")
    print(f"  device          = {device}")
    print(f"  forget_set      = {args.forget_set}")
    print(f"  n_models        = {args.n_models}")
    print(f"  epochs          = {args.epochs}")
    print(f"  lr              = {args.lr}")
    print(f"  noise_std       = {args.noise_std}")
    print(f"  max_grad_norm   = {args.max_grad_norm}")
    print(f"  seed            = {args.seed}")
    print(f"  recompute_pretrain_margins = {args.recompute_pretrain_margins}")

    # CIFAR-10 loaders for evaluation (validation only for KLOM).
    cifar_root = data_root / "data"
    val_loader = get_cifar_dataloader(cifar_root, split="val", shuffle=False, batch_size=256)

    # Forget / retain indices (used only for training unlearning).
    forget_indices = load_forget_indices(data_root, args.forget_set)
    retain_indices = np.setdiff1d(np.arange(50_000, dtype=np.int64), forget_indices)

    # Oracle margins (validation only) for each model.
    oracle_margins_list = []
    for model_id in range(args.n_models):
        m_val = load_margins_array(
            data_root,
            kind="oracle",
            phase="val",
            forget_id=args.forget_set,
            model_id=model_id,
        )
        oracle_margins_list.append(m_val)
    oracle_margins = torch.from_numpy(np.stack(oracle_margins_list, axis=0))

    # Pretrain margins baseline.
    if args.recompute_pretrain_margins:
        print("Recomputing pretrain margins from checkpoints...")
        pretrain_models = load_pretrain_checkpoints(data_root, n_models=args.n_models, device=device)

        pretrain_margins_list = []
        for model in pretrain_models:
            margins_val = compute_margins(model, val_loader, device=device)
            pretrain_margins_list.append(margins_val.cpu().numpy())
        pretrain_margins = torch.from_numpy(np.stack(pretrain_margins_list, axis=0))
    else:
        print("Loading pretrain margins from disk...")
        pretrain_margins_list = []
        for model_id in range(args.n_models):
            m_val = load_margins_array(
                data_root,
                kind="pretrain",
                phase="val",
                forget_id=None,
                model_id=model_id,
            )
            pretrain_margins_list.append(m_val)
        pretrain_margins = torch.from_numpy(np.stack(pretrain_margins_list, axis=0))

    # Noisy-descent unlearning.
    print("Running noisy_descent unlearning...")
    pretrain_models = load_pretrain_checkpoints(data_root, n_models=args.n_models, device=device)
    optimizer_cls = OPTIMIZERS["sgd"]
    method_fn = UNLEARNING_METHODS["noisy_descent"]

    unlearned_margins_list = []

    for model_idx, base_model in enumerate(pretrain_models):
        # Fresh RNG seed per model for noise, if desired.
        set_seed(args.seed + model_idx)

        retain_loader_train = get_cifar_dataloader(
            cifar_root,
            split="train",
            indices=retain_indices.tolist(),
            shuffle=True,
            batch_size=256,
        )

        # forget_loader is unused inside noisy_descent, so we can pass retain_loader_train again.
        epoch_models = method_fn(
            base_model,
            forget_loader=retain_loader_train,
            retain_loader=retain_loader_train,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs={"lr": args.lr},
            epochs=[args.epochs],
            device=device,
            noise_std=args.noise_std,
            max_grad_norm=args.max_grad_norm,
        )
        unlearned_model = epoch_models[args.epochs]

        margins_val = compute_margins(unlearned_model, val_loader, device=device)

        unlearned_margins_list.append(margins_val.cpu().numpy())

    unlearned_margins = torch.from_numpy(np.stack(unlearned_margins_list, axis=0))

    # KLOM comparisons.
    print("Computing KLOM scores...")
    klom_pretrain = klom_from_margins(oracle_margins, pretrain_margins)
    klom_unlearned = klom_from_margins(oracle_margins, unlearned_margins)

    def summarize(name: str, arr: np.ndarray) -> None:
        print(f"{name}:")
        print(f"  mean: {arr.mean():.6f}")
        print(f"  std:  {arr.std():.6f}")
        print(f"  min:  {arr.min():.6f}")
        print(f"  max:  {arr.max():.6f}")

    print()
    summarize("KLOM(pretrain, oracle)", klom_pretrain)
    print()
    summarize("KLOM(noisy_descent, oracle)", klom_unlearned)


if __name__ == "__main__":
    main()

