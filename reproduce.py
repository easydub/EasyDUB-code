"""Reproduce the Compute vs KLOM figure end-to-end.

Trains retrained models (3 seeds by default) via train_models.py,
then generates the 1x3 P95 figure to assets/compute_vs_klom.png.

Usage:
    python reproduce.py --data-dir /path/to/EasyDUB-dataset
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from easydub.data import ensure_dataset, load_forget_indices, retain_mask
from easydub.eval import klom_from_margins, load_margins_matrix

TRAIN_SCRIPT = str(Path(__file__).resolve().parent / "train_models.py")

DEFAULT_NOISE_STDS = [0.0]
DEFAULT_SEEDS = [0, 1, 2]
SPLITS = ["retain", "forget", "val"]
SPLIT_LABELS = {"retain": "Retain", "forget": "Forget", "val": "Validation"}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
})


# ---------------------------------------------------------------------------
# Grid training
# ---------------------------------------------------------------------------

def run_grid(args):
    """Train models across the noise_stds x seeds grid."""
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (ns, seed)
        for ns in args.noise_stds
        for seed in args.seeds
    ]

    print(f"Grid: {len(args.noise_stds)} noise levels x {len(args.seeds)} seeds "
          f"= {len(jobs)} runs")

    done, skipped, failed = 0, 0, 0
    for i, (ns, seed) in enumerate(jobs, 1):
        npz = results_dir / f"ns{ns}_seed{seed}.npz"
        if npz.exists():
            print(f"[{i}/{len(jobs)}] SKIP ns={ns} seed={seed} -- {npz} exists")
            skipped += 1
            continue

        cmd = [
            sys.executable, TRAIN_SCRIPT,
            "--data-dir", args.data_dir,
            "--noise-std", str(ns),
            "--seed", str(seed),
            "--n-models", str(args.n_models),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--peak-epoch", str(args.peak_epoch),
            "--forget-set", str(args.forget_set),
            "--results-dir", args.results_dir,
        ]

        print(f"\n{'='*60}")
        print(f"[{i}/{len(jobs)}] ns={ns} seed={seed}")
        print(f"{'='*60}")
        try:
            subprocess.run(cmd, check=True)
            done += 1
        except subprocess.CalledProcessError as e:
            print(f"FAILED (exit {e.returncode}): ns={ns} seed={seed}")
            failed += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            sys.exit(1)

    print(f"\nGrid done: {done}  Skipped: {skipped}  Failed: {failed}  Total: {len(jobs)}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def load_baselines(data_root, forget_id, n_models):
    """Compute pretrain and oracle KLOM baselines for each split."""
    forget_idx = load_forget_indices(data_root, forget_id)

    baselines = {}
    for split in SPLITS:
        oracle = load_margins_matrix(data_root, "oracle", split, forget_id, n_models)
        fid = forget_id if split == "forget" else None
        pretrain = load_margins_matrix(data_root, "pretrain", split, fid, n_models)

        # Pretrain retain has 50k; oracle retain has 49.9k. Mask forget indices.
        if split == "retain" and pretrain.shape[1] > oracle.shape[1]:
            pretrain = pretrain[:, retain_mask(pretrain.shape[1], forget_idx)]

        pretrain_klom = klom_from_margins(oracle, pretrain)
        oracle_klom = klom_from_margins(oracle, oracle)
        baselines[split] = {
            "pretrain_p95": float(np.percentile(pretrain_klom, 95)),
            "oracle_p95": float(np.percentile(oracle_klom, 95)),
        }
    return baselines


def plot_klom(args):
    """Load NPZ curves and generate the Compute vs KLOM figure."""
    results_dir = Path(args.results_dir)
    data_root = ensure_dataset(Path(args.data_dir))

    # --- Load training curves (noise=0.0 only) ---
    curves = {s: [] for s in SPLITS}
    for seed in args.seeds:
        npz_path = results_dir / f"ns0.0_seed{seed}.npz"
        if not npz_path.exists():
            print(f"WARNING: missing {npz_path}")
            continue
        d = np.load(str(npz_path))
        for split in SPLITS:
            curves[split].append(d[f"klom_p95_{split}"])

    # x-axis: compute as % of one full epoch on the retain set
    x = np.arange(1, args.epochs + 1) * (
        args.retain_size / args.total_train_size
    ) * (100.0 / args.epochs)

    # --- Baselines ---
    print("Computing baselines...")
    baselines = load_baselines(data_root, args.forget_set, args.n_models)

    # --- Plot: 1 row x 3 cols (P95 for retain, forget, val) ---
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 3.2), sharex=True,
        gridspec_kw={"wspace": 0.30},
    )

    curve_color = "#2563EB"     # blue-600
    pretrain_color = "#6B7280"  # gray-500
    oracle_color = "#DC2626"    # red-600

    for col, split in enumerate(SPLITS):
        ax = axes[col]
        arrs = curves[split]

        # Training curve: mean + seed range
        if arrs:
            stacked = np.stack(arrs)
            mean_curve = stacked.mean(axis=0)
            lo, hi = stacked.min(axis=0), stacked.max(axis=0)
            ax.fill_between(x, lo, hi, color=curve_color, alpha=0.12)
            ax.plot(x, mean_curve, color=curve_color, lw=2,
                    label="Retrained", zorder=3)

        # Pretrain baseline (horizontal)
        pt_val = baselines[split]["pretrain_p95"]
        ax.axhline(pt_val, color=pretrain_color, ls="--", lw=1.2,
                    label="Pretrain", zorder=2)

        # Red star for oracle (100% compute, 0 KLOM by definition)
        ax.plot(100, 0, marker="*", color=oracle_color, markersize=14,
                zorder=10, label="Oracle", linestyle="None",
                clip_on=False)

        # Cosmetics
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.set_title(SPLIT_LABELS[split], fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("P95 KLOM", fontsize=11)
        ax.set_xlabel("Compute (%)", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single legend at top — collect from all axes to include the star
    all_handles, all_labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                all_handles.append(h)
                all_labels.append(l)
                seen.add(l)
    fig.legend(all_handles, all_labels, loc="upper center", ncol=4,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.05))

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce Compute vs KLOM figure (train grid + plot)")
    p.add_argument("--data-dir", type=str, required=True,
                    help="Path to EasyDUB-dataset")

    # Grid parameters
    p.add_argument("--noise-stds", type=float, nargs="+", default=DEFAULT_NOISE_STDS)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--n-models", type=int, default=100)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.4)
    p.add_argument("--peak-epoch", type=int, default=5)
    p.add_argument("--forget-set", type=int, default=2)
    p.add_argument("--results-dir", type=str, default="./results")

    # Plot parameters
    p.add_argument("--output", type=str, default="assets/compute_vs_klom.png")
    p.add_argument("--retain-size", type=int, default=49900)
    p.add_argument("--total-train-size", type=int, default=50000)

    return p.parse_args()


def main():
    args = parse_args()
    ensure_dataset(Path(args.data_dir))

    run_grid(args)
    plot_klom(args)


if __name__ == "__main__":
    main()
