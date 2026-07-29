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
    # Stage 2: per-stage timing decomposition (§3 of review directive).
    # Split the forward into (velocity-fit / ODE integrate + emission)
    # so the 999-node projection can size the cache win correctly.
    # ------------------------------------------------------------------
    x = torch.randn(B, T, N_SUB, K)
    y = torch.randn(B, HORIZON, N_SUB, K)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    # Time the velocity-fit in isolation (biggest single cost)
    from models.continuous_time.climode_faithful.initial_velocity import (
        fit_initial_velocity,
    )
    t0 = time.time()
    x_grid_manual = x.reshape(B, T, H_SUB, W_SUB, K).permute(0, 1, 4, 2, 3).contiguous()
    _ = fit_initial_velocity(
        past_frames=x_grid_manual,
        past_time_hours=torch.arange(T, dtype=torch.float32),
        kernel_inv=adapter._kernel_inv,
        steps=200, lr=2.0, prior_weight=1e-7,
    )
    t_vel_fit_only = time.time() - t0
    timings["vel_fit_only"] = t_vel_fit_only
    print(f"\n[stage 2a] velocity fit (200-step Adam alone)   {t_vel_fit_only:6.2f} s   ← biggest single cost")

    # Now the full forward (which INCLUDES another velocity fit inside).
    # The delta is roughly the ODE + emission cost.
    t0 = time.time()
    optimizer.zero_grad()
    mean, std = adapter.forward(x, adj)
    t_fwd_train = time.time() - t0
    t_ode_emission_approx = max(0.0, t_fwd_train - t_vel_fit_only)
    timings["train_forward_full"] = t_fwd_train
    timings["ode_and_emission_approx"] = t_ode_emission_approx

    loss = adapter.native_loss((mean, std), y, var_coeff=1e-3)
    print(f"[stage 2b] full train forward (fit + ODE + emission)  {t_fwd_train:6.2f} s")
    print(f"           of which ODE + emission (approx):           {t_ode_emission_approx:6.2f} s")
    print(f"           mean shape:                         {tuple(mean.shape)}")
    print(f"           std shape:                          {tuple(std.shape)}")
    print(f"           initial NLL:                        {loss.item():.4e}  (finite: {torch.isfinite(loss).item()})")

    # Second forward to check NLL trajectory sign (should decrease)
    t0 = time.time()
    loss.backward()
    optimizer.step()
    t_backward = time.time() - t0
    timings["train_backward"] = t_backward
    print(f"[stage 2c] backward + optim step              {t_backward:6.2f} s")

    # Sanity: second forward step, verify NLL moved (may be up or down at
    # random init before warmup — we assert only that it's finite)
    optimizer.zero_grad()
    mean2, std2 = adapter.forward(x, adj)
    loss2 = adapter.native_loss((mean2, std2), y, var_coeff=1e-3)
    print(f"           NLL after 1 optim step:            {loss2.item():.4e}  (finite: {torch.isfinite(loss2).item()})")
    assert torch.isfinite(loss2), "NLL went non-finite after one training step — hyperparameter blow-up"

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
    # Stage 6: 999-node projection for Aug 5 gate — CORRECTED per §3
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("999-NODE PROJECTION FOR AUG 5 GATE  (corrected math)")
    print("=" * 70)
    N_target = 999
    # Convs + ODE state cost is O(N) (H*W). The KQV attention has a
    # quadratic component O(N^2) — beta = softmax(q^T k) is N×N. At
    # our grid sizes the linear term dominates, but for the projection
    # we quote BOTH so we can bracket the true scaling:
    scale_linear = N_target / N_SUB                          # 999 / 120 ≈ 8.3
    scale_quadratic = (N_target / N_SUB) ** 2                # ≈ 69
    # Attention itself has stride-2 downsampling internally (Self_attn_conv_reg
    # applies stride=2 in key/value convs), so N_tokens_attn ≈ N/4.
    # But even that grows: (N/4)^2 / (N_sub/4)^2 = (N/N_sub)^2 = 69×.
    print(f"  Subgrid N = {N_SUB},  target N = {N_target}")
    print(f"    linear-in-N (convs, ODE state):     scale = {scale_linear:.1f}×")
    print(f"    quadratic (KQV attention softmax):  scale = {scale_quadratic:.1f}×")
    print(f"  Real scaling is between these two — attention softmax is quadratic in tokens,")
    print(f"  but grid-conv work dominates at our grid sizes.")
    print()
    print(f"  === CPU per-step at 999 nodes (linear scaling floor) ===")
    print(f"     vel_fit (200 steps):              {t_vel_fit_only:5.2f}s × {scale_linear:.1f} ≈ {t_vel_fit_only * scale_linear:5.1f}s")
    print(f"     ODE + emission (forward):         {t_ode_emission_approx:5.2f}s × {scale_linear:.1f} ≈ {t_ode_emission_approx * scale_linear:5.1f}s")
    print(f"     backward + optim step:            {t_backward:5.2f}s × {scale_linear:.1f} ≈ {t_backward * scale_linear:5.1f}s")
    per_step_linear = (t_vel_fit_only + t_ode_emission_approx + t_backward) * scale_linear
    per_step_quadratic = per_step_linear * (scale_quadratic / scale_linear)  # attention share only, upper bound
    print(f"     TOTAL per-training-step (linear floor):     ~{per_step_linear:5.1f} s  (OPTIMISTIC)")
    print(f"     TOTAL per-training-step (attn-quad ceiling): ~{per_step_quadratic:5.1f} s  (PESSIMISTIC)")
    print()
    # The velocity cache changes the game — once warmed, vel_fit is a
    # file read (~10ms), so the *cached* per-step time drops massively.
    per_step_cached_linear = (t_ode_emission_approx + t_backward) * scale_linear
    print(f"  === Same, but with warmed velocity cache (vel_fit cost eliminated) ===")
    print(f"     per-training-step (cached, linear):  ~{per_step_cached_linear:5.1f} s")
    print(f"     ⇒ cache win = {per_step_linear / max(per_step_cached_linear, 0.01):.1f}× per step at 999 nodes")

    steps_per_epoch_est = 200
    per_epoch_sec_uncached = per_step_linear * steps_per_epoch_est
    per_epoch_sec_cached   = per_step_cached_linear * steps_per_epoch_est
    per_epoch_hr_uncached = per_epoch_sec_uncached / 3600
    per_epoch_hr_cached   = per_epoch_sec_cached / 3600
    print()
    print(f"  === Per-epoch projections (CPU; GPU L40S is 5–20× faster) ===")
    print(f"     uncached (first epoch, cache-warming): ~{per_epoch_hr_uncached:.2f} h CPU  →  ~{per_epoch_hr_uncached/10:.2f} h on L40S at 10× speedup")
    print(f"     cached (steady state):                 ~{per_epoch_hr_cached:.2f} h CPU  →  ~{per_epoch_hr_cached/10:.2f} h on L40S at 10× speedup")
    print()
    print(f"  === Validation cost (the ADDITIONAL threat §3 flagged) ===")
    # Full val: 46,311 windows in our test split (per §3 review — dwarfs training)
    val_windows_full = 46311
    val_windows_sub  = 2000    # subsampled — from config
    print(f"     full test/val pass (46K windows):  ~{val_windows_full * (t_ode_emission_approx * scale_linear) / 3600:.1f} h CPU cached")
    print(f"     subsampled val (2K windows):       ~{val_windows_sub * (t_ode_emission_approx * scale_linear) / 3600:.1f} h CPU cached")
    print(f"     ⇒ subsampled val for checkpoint selection is REQUIRED at 999 nodes;")
    print(f"       full val runs once at end. Configured in climode_faithful.yaml.")
    per_step_sec = per_step_linear      # for JSON output

    # Save a machine-readable summary
    out = {
        "subgrid": {"H": H_SUB, "W": W_SUB, "N": N_SUB, "K": K, "T": T, "horizon": HORIZON, "B": B},
        "params": adapter.get_param_count(),
        "timings_sec": timings,
        "peak_rss_gib": peak_gib,
        "nll_at_init": float(loss.item()),
        "nll_after_1_step": float(loss2.item()),
        "projection_999_nodes": {
            "scale_linear": scale_linear,
            "scale_quadratic_attention": scale_quadratic,
            "cpu_per_step_sec_uncached_linear": per_step_linear,
            "cpu_per_step_sec_uncached_attn_quadratic": per_step_quadratic,
            "cpu_per_step_sec_cached_linear": per_step_cached_linear,
            "cache_win_multiplier": per_step_linear / max(per_step_cached_linear, 0.01),
            "cpu_per_epoch_hr_uncached": per_epoch_hr_uncached,
            "cpu_per_epoch_hr_cached": per_epoch_hr_cached,
            "gpu_per_epoch_hr_cached_at_10x": per_epoch_hr_cached / 10,
            "steps_per_epoch_est": steps_per_epoch_est,
            "full_val_hr_cpu_cached": val_windows_full * (t_ode_emission_approx * scale_linear) / 3600,
            "subsampled_val_hr_cpu_cached": val_windows_sub * (t_ode_emission_approx * scale_linear) / 3600,
        },
    }
    (REPO_ROOT / "docs").mkdir(exist_ok=True)
    with open(REPO_ROOT / "docs" / "smoke_climode_faithful_output.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote docs/smoke_climode_faithful_output.json")
    print()


if __name__ == "__main__":
    main()
