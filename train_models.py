"""Fast multi-model ResNet9 training via grouped convolutions.

Trains N ResNet9 models simultaneously by stacking them into a single network
using grouped convolutions (conv layers) and einsum (linear layer). Each model
gets independent weights, BN stats, and per-model data shuffling.

Usage:
    python train_models.py --data-dir /path/to/EasyDUB-dataset --noise-std 0.0 --seed 0
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydub.data import CIFAR10_MEAN, CIFAR10_STD, ensure_dataset, load_forget_indices, retain_mask
from easydub.eval import klom_from_margins, load_margins_matrix
from easydub.models import Mul, Residual


# ---------------------------------------------------------------------------
# Multi-model modules
# ---------------------------------------------------------------------------

class MultiConvBN(nn.Module):
    """Conv2d + BatchNorm2d + ReLU for M models stacked via grouped conv."""

    def __init__(self, M, c_in, c_out, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.M = M
        self.c_in = c_in
        self.c_out = c_out
        self.conv = nn.Conv2d(
            M * c_in, M * c_out,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=M, bias=False,
        )
        self.bn = nn.BatchNorm2d(M * c_out)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class MultiLinear(nn.Module):
    """Independent linear layers for M models via einsum."""

    def __init__(self, M, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(M, d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):  # x: (M, B, d_in)
        return torch.einsum("moi,mbi->mbo", self.weight, x)


class ReshapeForLinear(nn.Module):
    """(B, M*C, 1, 1) -> (M, B, C) at the conv->linear boundary."""

    def __init__(self, M):
        super().__init__()
        self.M = M

    def forward(self, x):  # x: (B, M*C, 1, 1)
        B = x.size(0)
        x = x.flatten(2).squeeze(-1)  # (B, M*C)
        return x.view(B, self.M, -1).permute(1, 0, 2)  # (M, B, C)


def multi_resnet9(M, num_classes=10):
    """Build M stacked ResNet9 models as a single nn.Sequential."""
    def mcbn(ci, co, k=3, s=1, p=1):
        return MultiConvBN(M, ci, co, k, s, p)

    return nn.Sequential(
        mcbn(3,   64,  k=3, s=1, p=1),
        mcbn(64,  128, k=5, s=2, p=2),
        Residual(nn.Sequential(mcbn(128, 128), mcbn(128, 128))),
        mcbn(128, 256, k=3, s=1, p=1),
        nn.MaxPool2d(2),
        Residual(nn.Sequential(mcbn(256, 256), mcbn(256, 256))),
        mcbn(256, 128, k=3, s=1, p=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        ReshapeForLinear(M),
        MultiLinear(M, 128, num_classes),
        Mul(0.2),
    )


# ---------------------------------------------------------------------------
# GPU-resident data loader with per-model shuffling
# ---------------------------------------------------------------------------

class PreloadedDataLoader:
    """GPU-resident data with independent per-model shuffling."""

    def __init__(self, x, y, n_models, batch_size, shuffle=True):
        self.x = x              # (N, C, H, W) on GPU
        self.y = y              # (N,) on GPU
        self.M = n_models
        self.N = len(x)
        self.bs = batch_size
        self.shuffle = shuffle

    def _mkperm(self):
        if self.shuffle:
            self.perm = torch.stack([
                torch.randperm(self.N, device=self.x.device)
                for _ in range(self.M)
            ])  # (M, N)
        else:
            self.perm = torch.arange(self.N, device=self.x.device).unsqueeze(0).expand(self.M, -1)

    def __iter__(self):
        self._mkperm()
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr:self.ptr + self.bs]  # (M, bs_actual)
        self.ptr += self.bs
        bs = idx.shape[1]

        flat = idx.reshape(-1)  # (M*bs,)
        gx = self.x[flat].view(self.M, bs, *self.x.shape[1:])  # (M, bs, C, H, W)
        batch_x = gx.permute(1, 0, 2, 3, 4).reshape(bs, self.M * self.x.shape[1], *self.x.shape[2:])
        # (bs, M*C, H, W)
        gy = self.y[flat].view(self.M, bs)  # (M, bs)
        return batch_x, gy

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cifar_to_gpu(data_root, forget_id, device):
    """Load CIFAR-10 retain/forget/val as GPU tensors with standard normalization."""
    from torchvision import datasets

    cifar = datasets.CIFAR10(root=data_root / "data", train=True, download=True)
    x_all = torch.from_numpy(cifar.data).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    x_all = (x_all - mean) / std
    y_all = torch.tensor(cifar.targets, dtype=torch.long)

    forget_idx = load_forget_indices(data_root, forget_id)
    mask = retain_mask(len(x_all), forget_idx)
    retain_idx = np.where(mask)[0]

    val_ds = datasets.CIFAR10(root=data_root / "data", train=False, download=True)
    x_val = torch.from_numpy(val_ds.data).permute(0, 3, 1, 2).float() / 255.0
    x_val = (x_val - mean) / std
    y_val = torch.tensor(val_ds.targets, dtype=torch.long)

    return {
        "retain": (x_all[retain_idx].to(device), y_all[retain_idx].to(device)),
        "forget": (x_all[forget_idx].to(device), y_all[forget_idx].to(device)),
        "val":    (x_val.to(device),              y_val.to(device)),
    }


# ---------------------------------------------------------------------------
# Margin computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_margins_multi(model, x, y, M, batch_size=512, use_amp=False):
    """Compute margins for all M models on shared eval data.

    Returns: (M, N) tensor on CPU.
    """
    model.eval()
    chunks = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        xb_m = xb.repeat(1, M, 1, 1)  # (bs, M*3, H, W)
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(xb_m)   # (M, bs, 10)
            logits = logits.float()    # margin math in fp32
        else:
            logits = model(xb_m)       # (M, bs, 10)
        bs = logits.shape[1]
        bidx = torch.arange(bs, device=logits.device)
        correct = logits[:, bidx, yb]  # (M, bs)
        masked = logits.clone()
        masked[:, bidx, yb] = -torch.inf
        lse = masked.logsumexp(dim=-1)  # (M, bs)
        chunks.append((correct - lse).cpu())
    model.train()
    return torch.cat(chunks, dim=1)  # (M, N)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_eval(model, data, M, args, device):
    """Train M models and compute margins at every epoch."""
    loader = PreloadedDataLoader(*data["retain"], M, args.batch_size, shuffle=True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
    )
    pct_start = min(args.peak_epoch / args.epochs, 0.5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.epochs * len(loader),
        pct_start=pct_start,
        anneal_strategy="linear",
        div_factor=1e6, final_div_factor=1e6,
    )

    split_sizes = {s: len(data[s][0]) for s in ["retain", "forget", "val"]}
    margins = {
        s: np.zeros((args.epochs, M, n), dtype=np.float32)
        for s, n in split_sizes.items()
    }

    use_amp = torch.cuda.is_bf16_supported()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loader:
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(batch_x)  # (M, B, 10)
                    loss = F.cross_entropy(
                        logits.reshape(-1, 10), batch_y.reshape(-1)
                    ) * M
            else:
                logits = model(batch_x)  # (M, B, 10)
                loss = F.cross_entropy(
                    logits.reshape(-1, 10), batch_y.reshape(-1)
                ) * M
            optimizer.zero_grad()
            loss.backward()
            if args.noise_std > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * args.noise_std)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  [GPU {device}] Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}")

        for split in ["retain", "forget", "val"]:
            margins[split][epoch] = compute_margins_multi(
                model, *data[split], M, use_amp=use_amp
            ).numpy()

    return margins


# ---------------------------------------------------------------------------
# Multi-GPU orchestration
# ---------------------------------------------------------------------------

def gpu_worker(rank, world_size, args):
    device = f"cuda:{rank}"
    per_gpu = math.ceil(args.n_models / world_size)
    start = rank * per_gpu
    M = min(per_gpu, args.n_models - start)
    if M <= 0:
        return

    print(f"[GPU {rank}] Training models {start}..{start + M - 1} (M={M})")
    torch.manual_seed(args.seed * 100000 + start)
    data = load_cifar_to_gpu(Path(args.data_dir), args.forget_set, device)
    model = multi_resnet9(M, num_classes=10).to(device)

    margins = train_and_eval(model, data, M, args, device)

    results_dir = Path(args.results_dir)
    np.savez(
        results_dir / f"shard_{rank}.npz",
        model_start=start, model_end=start + M,
        **{f"margins_{s}": margins[s] for s in ["retain", "forget", "val"]},
    )
    print(f"[GPU {rank}] Done. Saved shard_{rank}.npz")


# ---------------------------------------------------------------------------
# Merge shards + compute KLOM
# ---------------------------------------------------------------------------

def merge_and_compute_klom(args):
    """Load shard NPZs, concatenate margins, compute KLOM, save result."""
    results_dir = Path(args.results_dir)
    shards = sorted(
        results_dir.glob("shard_*.npz"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not shards:
        print("No shards found!")
        return

    epoch_margins = {}
    for split in ["retain", "forget", "val"]:
        epoch_margins[split] = np.concatenate(
            [np.load(str(f))[f"margins_{split}"] for f in shards], axis=1
        )  # (epochs, n_models, n_samples)

    print(f"Merged margins: {epoch_margins['retain'].shape}")

    data_root = ensure_dataset(Path(args.data_dir))
    n_models = epoch_margins["retain"].shape[1]

    oracle = {
        split: load_margins_matrix(data_root, "oracle", split, args.forget_set, n_models)
        for split in ["retain", "forget", "val"]
    }

    n_epochs = epoch_margins["retain"].shape[0]
    save_data = {}
    for split in ["retain", "forget", "val"]:
        klom_means, klom_p95s, klom_p05s = [], [], []
        for epoch in range(n_epochs):
            unlearned = torch.from_numpy(epoch_margins[split][epoch])
            klom = klom_from_margins(oracle[split], unlearned)
            klom_means.append(float(np.mean(klom)))
            klom_p95s.append(float(np.percentile(klom, 95)))
            klom_p05s.append(float(np.percentile(klom, 5)))
        save_data[f"klom_mean_{split}"] = np.array(klom_means)
        save_data[f"klom_p95_{split}"] = np.array(klom_p95s)
        save_data[f"klom_p05_{split}"] = np.array(klom_p05s)
        save_data[f"margins_{split}"] = epoch_margins[split]

        print(f"  {split}: epoch {n_epochs}: "
              f"mean={klom_means[-1]:.4f} P95={klom_p95s[-1]:.4f}")

    npz_path = results_dir / f"ns{args.noise_std}_seed{args.seed}.npz"
    np.savez(npz_path, **save_data)
    print(f"Saved: {npz_path}")

    for f in shards:
        f.unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fast multi-model ResNet9 training")
    p.add_argument("--data-dir", type=str, required=True,
                    help="Path to EasyDUB-dataset")
    p.add_argument("--noise-std", type=float, default=0.0,
                    help="Gradient noise std")
    p.add_argument("--seed", type=int, default=0,
                    help="RNG seed")
    p.add_argument("--n-models", type=int, default=100,
                    help="Total number of models to train")
    p.add_argument("--epochs", type=int, default=24,
                    help="Training epochs")
    p.add_argument("--batch-size", type=int, default=256,
                    help="Batch size")
    p.add_argument("--lr", type=float, default=0.4,
                    help="Peak learning rate")
    p.add_argument("--peak-epoch", type=int, default=5,
                    help="OneCycleLR peak epoch")
    p.add_argument("--forget-set", type=int, default=2,
                    help="Forget set ID")
    p.add_argument("--results-dir", type=str, default="./results",
                    help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    world_size = min(torch.cuda.device_count(), args.n_models)
    if world_size == 0:
        raise RuntimeError("No CUDA GPUs available")
    print(f"Training {args.n_models} models across {world_size} GPUs")
    print(f"  noise_std={args.noise_std}, seed={args.seed}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    if world_size == 1:
        gpu_worker(0, 1, args)
    else:
        torch.multiprocessing.spawn(
            gpu_worker, nprocs=world_size, args=(world_size, args),
        )
    merge_and_compute_klom(args)


if __name__ == "__main__":
    main()
