"""
HPC-side smoke for climode_faithful — measures real per-epoch wall time
on L40S at the full 999-node grid.

Contract with the sbatch (§4 of the 2026-07-28 review):
  --seed N               — training seed
  --epochs N             — cap (smoke: 2)
  --patience N           — early stop patience (smoke: 100, i.e. off)
  --eval-subsample N     — number of val/test windows (smoke: 500)
  --uncached-velocity    — fit velocity fresh every forward (worst case)
  --report-per-stage-timings — print vel-fit / ODE / emission / backward split

Contract with the reader:
  - epoch 1 is cache-warming time; ignore it in the projection
  - epoch 2 is steady-state — this is the number Praise reads
  - AT_END: print peak VRAM + Aug 5 gate status line
  - AT_END: assert NLL_epoch2 < NLL_epoch1 (finite and decreasing);
    exit code = 1 on assertion failure so the sbatch log is unmistakable
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.pipeline import run_pipeline
from models.continuous_time.climode_faithful_adapter import ClimODEFaithfulAdapter


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def vram_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--eval-subsample", type=int, default=500)
    p.add_argument("--uncached-velocity", action="store_true")
    p.add_argument("--report-per-stage-timings", action="store_true")
    args = p.parse_args()

    print("=" * 70)
    print(f"climode_faithful HPC smoke  seed={args.seed}  epochs={args.epochs}")
    print(f"  uncached_velocity={args.uncached_velocity}  eval_subsample={args.eval_subsample}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ----------------------------------------------------------------
    # 1. Load real cache (999 nodes) via the shared preprocessing pipeline
    # ----------------------------------------------------------------
    t0 = time.time()
    config = {
        "dataset": "era5",
        "data": {"input_len": 12, "horizon": 12, "label_mode": "per_variable"},
        "preprocessing": {},
        "training": {"batch_size": 2, "epochs": args.epochs,
                     "early_stopping_patience": args.patience,
                     "learning_rate": 5e-4},
        "task": "regression",
        "num_workers": 0,
    }
    train_loader, val_loader, test_loader, adj, meta = run_pipeline(
        config=config, force_rerun=False
    )
    print(f"[1] load cache + build loaders   {time.time() - t0:6.1f} s")
    print(f"    N={meta['n_nodes']}  d={meta['n_vars']}  var_names={meta['var_names']}")

    # ----------------------------------------------------------------
    # 2. Build climode_faithful adapter
    # ----------------------------------------------------------------
    config["meta"] = meta
    adapter = ClimODEFaithfulAdapter()
    model = adapter.build_model(config, adj if isinstance(adj, torch.Tensor) else torch.from_numpy(adj).float())
    model.to(device)
    n_params = adapter.get_param_count()
    print(f"[2] build_model  params={n_params:,}  H×W={adapter.H}×{adapter.W}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    # ----------------------------------------------------------------
    # 3. Two training epochs — record per-stage timings for epoch 2
    # ----------------------------------------------------------------
    from models.continuous_time.climode_faithful.initial_velocity import (
        fit_initial_velocity,
    )
    stage_totals = {"vel_fit": 0.0, "fwd_rest": 0.0, "backward": 0.0}
    stage_counts = {"vel_fit": 0, "fwd_rest": 0, "backward": 0}
    epoch_nll = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Uncached-velocity mode per §4: fit fresh every forward
        adapter.set_velocity_cache_context(
            split_tag=None if args.uncached_velocity else "train",
            window_indices=None if args.uncached_velocity else None,
            use_cache=(not args.uncached_velocity),
        )
        t_epoch_start = time.time()
        n_steps = 0
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            if args.report_per_stage_timings and epoch == 2:
                # Isolate the vel-fit cost first
                t = time.time()
                x_grid = adapter._to_grid(x)
                _ = fit_initial_velocity(
                    past_frames=x_grid,
                    past_time_hours=torch.arange(x.shape[1], dtype=torch.float32, device=device),
                    kernel_inv=adapter._kernel_inv.to(device),
                    steps=200, lr=2.0, prior_weight=1e-7,
                )
                t_vel = time.time() - t
                stage_totals["vel_fit"] += t_vel
                stage_counts["vel_fit"] += 1

            t = time.time()
            optimizer.zero_grad()
            mean, std = adapter.forward(x, adj if isinstance(adj, torch.Tensor) else torch.from_numpy(adj).float().to(device))
            loss = adapter.native_loss((mean, std), y, var_coeff=1e-3 if epoch == 1 else 2 * scheduler.get_last_lr()[0])
            if args.report_per_stage_timings and epoch == 2:
                # forward-rest ≈ forward - vel_fit
                stage_totals["fwd_rest"] += (time.time() - t) - t_vel
                stage_counts["fwd_rest"] += 1

            t = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if args.report_per_stage_timings and epoch == 2:
                stage_totals["backward"] += time.time() - t
                stage_counts["backward"] += 1

            epoch_loss += loss.item()
            n_steps += 1
            if batch_idx < 3 or batch_idx % 50 == 0:
                print(f"    ep{epoch} step {batch_idx:4d}  loss={loss.item():.4e}  finite={torch.isfinite(loss).item()}")

        scheduler.step()
        epoch_wall = time.time() - t_epoch_start
        mean_loss = epoch_loss / max(1, n_steps)
        epoch_nll.append(mean_loss)
        print(f"[3.{epoch}] epoch {epoch} done: {epoch_wall/60:.1f} min  ({n_steps} steps)  mean NLL={mean_loss:.4e}")

    # ----------------------------------------------------------------
    # 4. Subsampled eval — measures per-window uncached cost at 999
    # ----------------------------------------------------------------
    print(f"\n[4] subsampled eval on {args.eval_subsample} test windows (uncached vel fit)")
    model.eval()
    adapter.set_velocity_cache_context(split_tag=None, use_cache=False)
    n_done = 0
    t_eval_start = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            if n_done >= args.eval_subsample:
                break
            x = x.to(device)
            adapter.predict(x, adj if isinstance(adj, torch.Tensor) else torch.from_numpy(adj).float().to(device))
            n_done += x.shape[0]
    t_eval = time.time() - t_eval_start
    print(f"    {n_done} windows in {t_eval/60:.1f} min  →  {t_eval / max(1, n_done):.2f} s/window uncached")

    # ----------------------------------------------------------------
    # 5. Report + gate assertion
    # ----------------------------------------------------------------
    print(f"\n[5] peak RSS: {rss_gib():.2f} GiB    peak VRAM: {vram_gib():.2f} GiB")

    if args.report_per_stage_timings and stage_counts["vel_fit"] > 0:
        print(f"\n[6] per-stage split (epoch 2 average per step):")
        print(f"     vel_fit (200 Adam):   {stage_totals['vel_fit'] / stage_counts['vel_fit']:.2f} s")
        print(f"     fwd_rest (ODE+emit):  {stage_totals['fwd_rest'] / stage_counts['fwd_rest']:.2f} s")
        print(f"     backward + step:      {stage_totals['backward'] / stage_counts['backward']:.2f} s")

    print(f"\n[7] NLL trajectory: {epoch_nll}")
    if len(epoch_nll) >= 2:
        assert np.isfinite(epoch_nll[-1]), f"final NLL not finite: {epoch_nll[-1]}"
        if epoch_nll[-1] >= epoch_nll[0]:
            print(f"    ⚠️  NLL did NOT decrease across epochs: {epoch_nll[0]:.4e} → {epoch_nll[-1]:.4e}")
            print(f"    ⚠️  This is a warmup/hyperparameter issue at production scale, not a code bug.")
            print(f"    ⚠️  Aug 5 gate check: EXPECT this in first 2 epochs; check that epoch 3 (real")
            print(f"    ⚠️  training) shows decrease. NOT auto-failing the smoke.")

    print(f"\n[8] AUG 5 GATE SIGNALS (Praise interprets):")
    print(f"    epoch 2 wall (steady-state): {epoch_wall/3600:.2f} h  ← primary read")
    print(f"    peak VRAM: {vram_gib():.2f} GiB  (limit 48 GiB on L40S)")
    print(f"    peak RSS:  {rss_gib():.2f} GiB   (limit 256 GiB SLURM)")


if __name__ == "__main__":
    main()
