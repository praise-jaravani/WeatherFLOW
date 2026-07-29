"""
Warm the climode_faithful velocity cache for a fixed window set.

The Eq. 10 fit is data-only (V1) → cache once, use across all seeds.
This script enumerates the window sets (train / val-subsample / full-val
/ test), reports counts + projected wall time at the smoke-measured
per-window fit rate, and computes+stores the fits.

Idempotent: existing cache entries are skipped (`use_cache=True`).

Usage (HPC):
    sbatch scripts/jobs/warm_velocity_cache.sbatch --split train --chunk 0 --n-chunks 2
    sbatch scripts/jobs/warm_velocity_cache.sbatch --split val   --chunk 0 --n-chunks 1
    ... etc

Usage (local, small):
    python scripts/warm_velocity_cache.py --split smoke --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.pipeline import run_pipeline
from models.continuous_time.climode_faithful.initial_velocity import (
    gaussian_rbf_kernel,
)
from models.continuous_time.climode_faithful.velocity_cache import (
    load_or_fit_velocity, cache_stats,
)


def _get_split_loader(split, config):
    """Return the DataLoader for a named split."""
    train_loader, val_loader, test_loader, adj, meta = run_pipeline(config=config, force_rerun=False)
    if split in ("train",):
        return train_loader, meta
    if split in ("val", "val_full"):
        return val_loader, meta
    if split in ("test",):
        return test_loader, meta
    raise ValueError(f"unknown split {split!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--n-chunks", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="stop after N windows (for testing)")
    p.add_argument("--vel-fit-steps", type=int, default=200)
    p.add_argument("--vel-fit-lr", type=float, default=2.0)
    p.add_argument("--vel-fit-prior", type=float, default=1e-7)
    args = p.parse_args()

    torch.manual_seed(0)   # velocity_cache uses per-window deterministic seeds; this is just a global hedge
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[warm] split={args.split}  chunk={args.chunk}/{args.n_chunks}  device={device}")

    config = {
        "dataset": "era5",
        "data": {"input_len": 12, "horizon": 12, "label_mode": "per_variable"},
        "preprocessing": {},
        "training": {"batch_size": 1, "epochs": 1, "early_stopping_patience": 1, "learning_rate": 5e-4},
        "task": "regression",
        "num_workers": 0,
    }
    loader, meta = _get_split_loader(args.split, config)
    n_total = len(loader.dataset)

    # Chunk = a subset of windows for this job (index i modulo n_chunks == args.chunk)
    my_indices = [i for i in range(n_total) if (i % args.n_chunks) == args.chunk]
    if args.limit:
        my_indices = my_indices[: args.limit]
    print(f"[warm] {len(my_indices)} of {n_total} windows for this chunk")

    # Grid + RBF kernel (H×W taken from adj/meta)
    coords = np.asarray(meta["coords"], dtype=np.float32)
    _, first_idx_lat = np.unique(coords[:, 0], return_index=True)
    row_order_lats = coords[np.sort(first_idx_lat), 0]
    _, first_idx_lon = np.unique(coords[:, 1], return_index=True)
    row_order_lons = coords[np.sort(first_idx_lon), 1]
    H, W = len(row_order_lats), len(row_order_lons)
    kernel_inv = gaussian_rbf_kernel(H, W, row_order_lats, row_order_lons).to(device)
    print(f"[warm] grid H×W = {H}×{W}  |  kernel_inv on {device}")

    T_in = 12
    started = time.time()
    n_done = n_hit = n_fit = 0
    for i, idx in enumerate(my_indices):
        # Get window `idx` — the DataLoader stores samples in order; we
        # index into the underlying dataset.
        x, _ = loader.dataset[idx]
        x = torch.from_numpy(x.numpy()) if hasattr(x, "numpy") and callable(x.numpy) else x
        # Reshape (T, N, d) → (1, T, d, H, W) for the fit
        x_grid = x.unsqueeze(0).reshape(1, T_in, H, W, -1).permute(0, 1, 4, 2, 3).contiguous().to(device)

        before = time.time()
        v = load_or_fit_velocity(
            split_tag=args.split,
            window_index=int(idx),
            past_frames=x_grid,
            past_time_hours=torch.arange(T_in, dtype=torch.float32, device=device),
            kernel_inv=kernel_inv,
            steps=args.vel_fit_steps,
            lr=args.vel_fit_lr,
            prior_weight=args.vel_fit_prior,
            use_cache=True,
        )
        elapsed = time.time() - before
        # Rough heuristic: <100ms → cache hit
        if elapsed < 0.1:
            n_hit += 1
        else:
            n_fit += 1
        n_done += 1
        if n_done <= 3 or n_done % 100 == 0:
            print(f"[warm] {n_done}/{len(my_indices)}  win_idx={idx}  {elapsed:.2f}s  ({'HIT' if elapsed < 0.1 else 'FIT'})")

    total_elapsed = time.time() - started
    stats = cache_stats(args.split)
    print(f"\n[warm] DONE  split={args.split}  chunk={args.chunk}")
    print(f"       {n_done} windows in {total_elapsed:.0f}s  ({n_fit} fit, {n_hit} hit)")
    print(f"       mean per-window: {total_elapsed / max(n_done, 1):.2f}s")
    print(f"       cache stats: {stats}")


if __name__ == "__main__":
    main()
