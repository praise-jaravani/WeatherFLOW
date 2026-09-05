"""
Deliverable C of specs/VPT_Extended_Spec.md — block-autoregressive Rollout VPT.

Runs K rollout blocks per start point on a trained checkpoint, applies
per-block hygiene (physical-bounds clamps on tp / rh), records per-step
NRMSE, computes first-crossing RVPT at each ε.

For SAWS (input_len=12, horizon=12), K=4 gives a 48-hour rollout — the
zero-retraining partial answer to Moodley's 24h+ horizon question.

Guards implemented:
  - NaN/Inf terminates that start (spec §5.4); records crossing at current step
  - Early stop when max variable NRMSE > 5.0
  - Mean-collapse screen (NRMSE hovering in [0.9, 1.1] for ≥12 steps)
  - Regression guard: block k=0 == direct forecast (tested via
    tests/test_vpt_extended.py::test_rollout_first_block_equals_direct)

Usage:
    python scripts/run_rollout_vpt.py \
        --model mtgode --dataset saws --seed 42 \
        --checkpoint outputs/mtgode_saws_regression_XXXX/seed_42/best.pt \
        --K 4 --stride 24
"""

from __future__ import annotations

import os
import sys
_ROOT = "/home/praise/Computer_Science_Resources/Weatherflow"
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from evaluation.vpt_extended import (
    DEFAULT_EPS_GRID,
    summarise_vfh,
    vfh_grid,
)
from evaluation.rollout import (
    apply_hygiene,
    detect_collapse,
    detect_numerical_failure,
    early_stop_start,
    persistence_rollout,
    rollout_nrmse_curve,
    rollout_predict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_all_for_rollout(model_name: str, dataset: str, checkpoint_path: Path):
    from cli import load_adapter, load_config
    from preprocessing.pipeline import load_cache, run_pipeline

    config = load_config(model_name, {})
    config["dataset"] = dataset
    config["task"] = "regression"

    _, _, _, adj, meta = run_pipeline(config=config)
    var_names = meta["var_names"]
    mean, std = meta["mean"], meta["std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj_t = torch.from_numpy(adj).float().to(device)

    adapter_cls = load_adapter(model_name)
    adapter = adapter_cls()
    config["meta"] = meta
    adapter.build_model(config, adj_t)
    adapter.model.to(device)

    if checkpoint_path.exists():
        adapter.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        log.info(f"Loaded checkpoint {checkpoint_path}")
    else:
        log.warning(f"No checkpoint at {checkpoint_path} — untrained model")

    # Load raw test data for rollout starts (bypass the sliding-window dataloader)
    cached = load_cache(dataset)
    X_test = cached["X_test"].astype(np.float32)   # (T, N, d)

    return adapter, adj_t, var_names, mean, std, X_test


def _run_one_start(adapter, adj_t, X_test, t0, input_len, K, horizon,
                   mean, std, var_names, device):
    """Run K-block rollout for one start point at t0.

    Returns: (curve (K·H, d) float32, flags dict {nan, early_stop, collapse})
    """
    seed_window = X_test[t0 - input_len : t0]     # (input_len, N, d)
    truth = X_test[t0 : t0 + K * horizon]         # (K·H, N, d)
    if truth.shape[0] < K * horizon:
        return None, {"insufficient_truth": True}

    flags = {"nan": False, "early_stop": False, "collapse": False}

    def _predict(w_np):
        # Reseed torch's global RNG before every forward — see
        # specs/ADJACENCY_AUDIT.md. MTGODE's adaptive graph builder calls
        # torch.rand during forward even in eval + no_grad, giving ~5e-3
        # drift per call. Rollout compounds this across K blocks, so
        # unseeded rollouts have non-reproducible per-start RVPT numbers.
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        with torch.no_grad():
            adapter.model.eval()
            w = torch.from_numpy(w_np[None]).to(device)   # (1, T, N, d)
            p = adapter.predict(w, adj_t)                  # (1, H, N, d)
            return p[0].cpu().numpy()

    curve_blocks = []
    window = seed_window.copy()
    for k in range(K):
        pred = _predict(window)
        if detect_numerical_failure(pred):
            flags["nan"] = True
            log.warning(f"  t0={t0}: NaN/Inf at block {k}; terminating start")
            # Fill remaining steps with large NRMSE to force crossing.
            remaining = K * horizon - k * horizon
            curve_blocks.append(np.full((remaining, pred.shape[-1]), 1e6, dtype=np.float32))
            break
        block_curve = rollout_nrmse_curve(pred, truth[k * horizon : (k + 1) * horizon])
        curve_blocks.append(block_curve)
        if early_stop_start(block_curve):
            flags["early_stop"] = True
            remaining = K * horizon - (k + 1) * horizon
            if remaining > 0:
                curve_blocks.append(np.full((remaining, pred.shape[-1]), 1e6, dtype=np.float32))
            break
        window = apply_hygiene(pred, mean, std, var_names)

    curve = np.concatenate(curve_blocks, axis=0)
    if detect_collapse(curve).any():
        flags["collapse"] = True
    return curve, flags


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="saws")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--K", type=int, default=4, help="rollout blocks (spec default 4 → 48h)")
    p.add_argument("--stride", type=int, default=24, help="stride between starts")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/vpt_extended/rollout"))
    p.add_argument("--eps", nargs="+", type=float, default=list(DEFAULT_EPS_GRID))
    p.add_argument("--input-len", type=int, default=12)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    seed_dir = args.out_dir / args.model / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    values_path = seed_dir / "rvpt_values.npz"
    curves_path = seed_dir / "rollout_curves.npz"
    if values_path.exists() and curves_path.exists() and not args.force:
        log.info(f"Skipping seed {args.seed} — cached at {values_path}")
        return

    adapter, adj_t, var_names, mean, std, X_test = _load_all_for_rollout(
        args.model, args.dataset, args.checkpoint
    )
    device = adj_t.device
    T = X_test.shape[0]

    # Start points: every stride hours where we have a full seed + full K*H truth
    starts = list(range(args.input_len, T - args.K * args.horizon + 1, args.stride))
    log.info(f"Running {len(starts)} rollout starts (K={args.K}, stride={args.stride})")

    all_curves = []
    all_flags = []
    all_t0 = []
    for i, t0 in enumerate(starts):
        curve, flags = _run_one_start(
            adapter, adj_t, X_test, t0, args.input_len, args.K, args.horizon,
            mean, std, var_names, device,
        )
        if curve is None:
            continue
        all_curves.append(curve)
        all_t0.append(t0)
        all_flags.append(flags)
        if (i + 1) % 100 == 0:
            log.info(f"  {i+1}/{len(starts)} done")

    curves_arr = np.stack(all_curves).astype(np.float32)     # (S0, K·H, d)
    t0_arr = np.array(all_t0, dtype=np.int64)
    flag_bitmask = np.array([
        (1 if f["nan"] else 0) | (2 if f["early_stop"] else 0) | (4 if f["collapse"] else 0)
        for f in all_flags
    ], dtype=np.uint8)

    n_num_fail = int(sum(f["nan"] for f in all_flags))
    n_collapse = int(sum(f["collapse"] for f in all_flags))
    log.info(f"Numerical failures: {n_num_fail}/{len(all_flags)}  "
             f"Collapsed: {n_collapse}/{len(all_flags)}")

    # RVPT: first-crossing per start per variable per ε
    rvpt = vfh_grid(curves_arr, args.eps).astype(np.int16)   # (S0, d, n_eps)

    np.savez_compressed(curves_path, nrmse=curves_arr,
                        start_t0=t0_arr, flags=flag_bitmask,
                        eps_grid=np.array(args.eps))
    np.savez_compressed(values_path, rvpt=rvpt, eps_grid=np.array(args.eps))
    log.info(f"Wrote {curves_path} + {values_path}")

    # Summary
    H_cap = args.K * args.horizon
    summary = summarise_vfh(rvpt.astype(np.int32), H_cap=H_cap,
                            var_names=var_names, eps_grid=args.eps,
                            curves_for_survival=curves_arr)
    summary["_meta"] = {
        "n_starts": len(all_flags),
        "n_numerical_failures": n_num_fail,
        "n_collapsed": n_collapse,
        "K": args.K, "stride": args.stride,
        "H_cap": H_cap,
    }
    (seed_dir / "rvpt_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote {seed_dir / 'rvpt_summary.json'}")


if __name__ == "__main__":
    main()
