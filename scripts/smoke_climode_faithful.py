"""
B.5 local smoke — end-to-end climode_faithful pipeline on a cropped
subgrid at production-like scale.

Runs on a 10×12 subgrid sliced from the (27, 37) SA box (preserves grid
semantics; does NOT random-sample nodes), K=8 variables, T=12, H=12:

  1. build adapter + config
  2. run forward → returns (μ, σ) tuple of expected shape
  3. compute native NLL loss
  4. one training step (backward + optimizer.step)
  5. eval-mode predict → returns (B, H, N, d) tensor of finite floats
  6. run through evaluation.regression.compute_all end-to-end

Records wall-clock per stage + peak RSS. Output: prints a table with
timings and the projection formula for the 999-node epoch time.

CPU is acceptable. Not a pytest test — this is the operator smoke that
gates the sbatch preparation.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation import regression as reg_eval
from models.continuous_time.climode_faithful_adapter import ClimODEFaithfulAdapter


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def build_subgrid_config(H: int, W: int, K: int, T: int, horizon: int) -> dict:
    """Realistic-looking coords for a cropped subgrid within the SA box."""
    lats = np.linspace(-30.0, -22.0, H, dtype=np.float32)[::-1].copy()   # north-first
    lons = np.linspace(15.0, 25.0, W, dtype=np.float32)
    coords = np.stack(np.meshgrid(lats, lons, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)
    return {
        "meta": {"coords": coords, "n_vars": K},
        "data": {"input_len": T, "horizon": horizon},
        "model": {
            "solver": "euler",
            "atol": 0.1, "rtol": 0.1,
            "use_att": True, "use_err": True, "use_pos": False,
            "vel_fit_steps": 200,        # full upstream value at smoke scale — this is where wall time will be
            "vel_fit_lr": 2.0,
            "vel_fit_prior": 1e-7,
        },
    }


def main():
    H_SUB, W_SUB = 10, 12
    N_SUB = H_SUB * W_SUB
    K = 8
    T = 12
    HORIZON = 12
    B = 2

    print("=" * 70)
    print(f"B.5 SMOKE — climode_faithful on cropped subgrid")
    print(f"  H×W = {H_SUB}×{W_SUB} = {N_SUB} nodes  |  K={K}  |  T={T}  |  H={HORIZON}  |  B={B}")
    print("=" * 70)

    torch.manual_seed(42)
    timings = {}

    # ------------------------------------------------------------------
    # Stage 1: build model
    # ------------------------------------------------------------------
    t0 = time.time()
    adapter = ClimODEFaithfulAdapter()
    cfg = build_subgrid_config(H_SUB, W_SUB, K, T, HORIZON)
    adj = torch.zeros((N_SUB, N_SUB))
    model = adapter.build_model(cfg, adj)
    timings["build_model"] = time.time() - t0
    print(f"\n[stage 1] build_model                {timings['build_model']:6.2f} s")
    print(f"          params:              {adapter.get_param_count():,}")
    print(f"          H×W captured:        {adapter.H}×{adapter.W}")

    # ------------------------------------------------------------------
    # Stage 2: one training forward + backward
    # ------------------------------------------------------------------
    x = torch.randn(B, T, N_SUB, K)
    y = torch.randn(B, HORIZON, N_SUB, K)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    t0 = time.time()
    optimizer.zero_grad()
    mean, std = adapter.forward(x, adj)
    t_fwd_train = time.time() - t0

    loss = adapter.native_loss((mean, std), y, var_coeff=1e-3)
    print(f"\n[stage 2] training forward           {t_fwd_train:6.2f} s")
    print(f"          mean shape:          {tuple(mean.shape)}")
    print(f"          std shape:           {tuple(std.shape)}")
    print(f"          initial NLL:         {loss.item():.4e}  (finite: {torch.isfinite(loss).item()})")

    t0 = time.time()
    loss.backward()
    optimizer.step()
    t_backward = time.time() - t0
    timings["train_forward"] = t_fwd_train
    timings["train_backward"] = t_backward
    print(f"          backward + step:     {t_backward:6.2f} s")

    # ------------------------------------------------------------------
    # Stage 3: eval-mode predict
    # ------------------------------------------------------------------
    t0 = time.time()
    pred = adapter.predict(x, adj)
    timings["predict"] = time.time() - t0
    print(f"\n[stage 3] eval predict               {timings['predict']:6.2f} s")
    print(f"          pred shape:          {tuple(pred.shape)}  (expected ({B}, {HORIZON}, {N_SUB}, {K}))")
    print(f"          finite:              {torch.isfinite(pred).all().item()}")
    assert pred.shape == (B, HORIZON, N_SUB, K)

    # ------------------------------------------------------------------
    # Stage 4: metrics integration — push predictions through compute_all
    # ------------------------------------------------------------------
    # Simulate a batch of test samples: use (samples, H, N, d) format
    n_test_samples = 4
    preds_np = torch.randn(n_test_samples, HORIZON, N_SUB, K).float().numpy()   # stand-in for real preds
    targets_np = torch.randn(n_test_samples, HORIZON, N_SUB, K).float().numpy()
    # Fake mean/std matching our K=8 var order
    mean = np.zeros((1, N_SUB, K), dtype=np.float32)
    std = np.ones((1, N_SUB, K), dtype=np.float32)

    t0 = time.time()
    var_names = ["t2m", "d2m", "tp", "u10", "v10", "sp", "ws10", "rh"]
    metrics = reg_eval.compute_all(
        preds_np, targets_np, extreme_mask=None,
        vpt_epsilon=0.3, mean=mean, std=std, var_names=var_names,
    )
    timings["compute_all"] = time.time() - t0
    print(f"\n[stage 4] evaluation.compute_all     {timings['compute_all']:6.2f} s")
    print(f"          returned keys:       {list(metrics.keys())[:6]}")
    assert "mae" in metrics and "vpt" in metrics

    # ------------------------------------------------------------------
    # Stage 5: peak RSS
    # ------------------------------------------------------------------
    peak_gib = rss_gib()
    print(f"\n[stage 5] peak RSS:                  {peak_gib:6.2f} GiB")

    # ------------------------------------------------------------------
    # Stage 6: 999-node projection for Aug 5 gate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("999-NODE PROJECTION FOR AUG 5 GATE")
    print("=" * 70)
    scale_factor = 999 / N_SUB
    # Forward cost scales roughly O(N) for attention (KQV bmm over grid),
    # O(N) for convs (H×W). Backward scales the same. Velocity fit
    # scales similarly (all ops are on the grid).
    projected_fwd_train_sec = t_fwd_train * scale_factor
    projected_backward_sec = t_backward * scale_factor
    print(f"  subgrid N = {N_SUB},  target N = 999,  scale = {scale_factor:.1f}×")
    print(f"  per-sample forward (train mode):   {t_fwd_train:5.2f} s  →  ~{projected_fwd_train_sec:5.1f} s at 999 nodes")
    print(f"  per-sample backward + optim step:  {t_backward:5.2f} s  →  ~{projected_backward_sec:5.1f} s at 999 nodes")
    per_step_sec = projected_fwd_train_sec + projected_backward_sec
    print(f"  per-training-step (999 nodes, B={B}):  ~{per_step_sec:5.1f} s")
    # Assume ~200 steps per epoch (upstream default niters=300 with batch_size=8;
    # ours will be similar order — refine after real cache size is known).
    steps_per_epoch_est = 200
    per_epoch_sec = per_step_sec * steps_per_epoch_est
    per_epoch_hr = per_epoch_sec / 3600
    print(f"  per-epoch (~{steps_per_epoch_est} steps/epoch, EST):     {per_epoch_sec:6.0f} s  ≈  {per_epoch_hr:.2f} h")
    print(f"  24h SLURM wall / ~{per_epoch_hr:.2f}h per epoch  =  ~{24/per_epoch_hr:.0f} epochs feasible per job")

    # Save a machine-readable summary
    out = {
        "subgrid": {"H": H_SUB, "W": W_SUB, "N": N_SUB, "K": K, "T": T, "horizon": HORIZON, "B": B},
        "params": adapter.get_param_count(),
        "timings_sec": timings,
        "peak_rss_gib": peak_gib,
        "projection_999_nodes": {
            "scale_factor": scale_factor,
            "per_step_sec_est": per_step_sec,
            "per_epoch_hr_est": per_epoch_hr,
            "epochs_per_24h_est": 24 / per_epoch_hr,
        },
    }
    (REPO_ROOT / "docs").mkdir(exist_ok=True)
    with open(REPO_ROOT / "docs" / "smoke_climode_faithful_output.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote docs/smoke_climode_faithful_output.json")
    print()


if __name__ == "__main__":
    main()
