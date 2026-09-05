"""
Deliverable B of specs/VPT_Extended_Spec.md — per-sample Valid Forecast
Horizon distributions.

Streams inference on the test set, per batch reducing to per-sample NRMSE
curves (S, H, d) — never materialises the full (S, H, N, d) preds+targets.
Saves per-seed npz artifacts + per-model summary.

Under SAWS (N=25, ~14k test hours) this runs in seconds on CPU. Under ERA5
(N=999, ~46k samples) memory rails matter — but this script's streaming
reducer never exceeds one batch of (B, H, N, d) at a time.

Usage:
    # Run on the current SAWS cache using seed 42 (needs a trained checkpoint)
    python scripts/run_per_sample_vfh.py \
        --model mtgode --dataset saws --seed 42 \
        --checkpoint outputs/mtgode_saws_regression_XXXX/seed_42/best.pt

Resumability: skips a seed if its vfh_values.npz already exists.
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
from typing import Any, Dict

import numpy as np
import torch

from evaluation.vpt_extended import (
    DEFAULT_EPS_GRID,
    aggregate_vfh_across_seeds,
    per_sample_nrmse,
    pooled_curve_matches,
    summarise_vfh,
    vfh_grid,
)
from evaluation.regression import nrmse_per_horizon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_model_and_data(model_name: str, dataset: str, checkpoint_path: Path,
                         config_overrides: Dict[str, Any]):
    """
    Rebuild the pipeline: dataloaders + adapter + adj + var_names + norm stats.
    Loads the checkpoint into the adapter.

    Returns:
      (test_loader, adapter, adj_tensor, var_names, mean, std)
    """
    from cli import ADAPTER_REGISTRY, load_adapter, load_config
    from preprocessing.pipeline import run_pipeline

    config = load_config(model_name, config_overrides)
    config["dataset"] = dataset
    config["task"] = "regression"

    train_loader, val_loader, test_loader, adj, meta = run_pipeline(config=config)
    var_names = meta["var_names"]
    mean = meta["mean"]
    std = meta["std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj_t = torch.from_numpy(adj).float().to(device)

    adapter_cls = load_adapter(model_name)
    adapter = adapter_cls()
    config["meta"] = meta
    adapter.build_model(config, adj_t)
    adapter.model.to(device)

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        adapter.model.load_state_dict(state)
        log.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        log.warning(f"No checkpoint at {checkpoint_path} — using untrained model "
                    f"(useful for pipeline smoke, not for real numbers)")

    return test_loader, adapter, adj_t, var_names, mean, std


def _streaming_per_sample_curves(
    adapter, test_loader, adj_t, rng_seed: int = 42,
) -> np.ndarray:
    """Iterate test batches; reduce each to (B, H, d) per-sample NRMSE;
    concatenate to (S, H, d).

    RNG reseeding: the adjacency audit (specs/ADJACENCY_AUDIT.md) established
    that MTGODE's adaptive-graph builder calls torch.rand during forward
    even in eval + no_grad mode, giving ~5e-3 drift across consecutive calls.
    Unseeded, per-sample VFH near an ε boundary can flip by a step between
    reruns and rollout compounds the noise across blocks. Reseed torch's
    global RNG before every batch to guarantee bitwise-identical outputs
    on rerun.
    """
    device = adj_t.device
    all_curves = []
    for x, y in test_loader:
        # Reseed BEFORE the forward pass — see docstring
        torch.manual_seed(rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng_seed)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            adapter.model.eval()
            pred = adapter.predict(x, adj_t)   # (B, H, N, d)
        # z-space per-sample NRMSE — pool over N only
        batch_curves = per_sample_nrmse(pred.cpu().numpy(), y.cpu().numpy())
        all_curves.append(batch_curves)
    return np.concatenate(all_curves, axis=0)


def _do_cross_seed_aggregation(args) -> None:
    """
    Deliverable B cross-seed aggregation (spec §4.3). Reads existing
    per-seed vfh_summary.json files and produces the model-level summary.
    """
    seed_summaries = []
    var_names = None
    H_cap = None
    for seed in args.aggregate:
        seed_path = args.out_dir / args.model / f"seed_{seed}" / "vfh_summary.json"
        if not seed_path.exists():
            raise FileNotFoundError(
                f"per-seed summary missing for seed {seed}: {seed_path}\n"
                f"Run --seed {seed} first to produce it."
            )
        summary = json.loads(seed_path.read_text())
        seed_summaries.append(summary)
        if var_names is None:
            var_names = list(summary.keys())
            H_cap = int(np.load(
                args.out_dir / args.model / f"seed_{seed}" / "vfh_curves.npz"
            )["nrmse"].shape[1])
    log.info(f"Aggregating across {len(seed_summaries)} seed(s): {args.aggregate}")
    log.info(f"  var_names = {var_names}, H_cap = {H_cap}")

    # Fix the per-seed summary key type: JSON deserialises float keys as strings.
    for s in seed_summaries:
        for v in s:
            s[v] = {float(k): v_ for k, v_ in s[v].items()}

    cross = aggregate_vfh_across_seeds(
        seed_summaries, H_cap=H_cap, var_names=var_names, eps_grid=args.eps
    )
    out_path = args.out_dir / f"{args.model}_vfh_summary.json"
    out_path.write_text(json.dumps(cross, indent=2))
    log.info(f"Wrote {out_path}")


def _pooled_curve_from_batches(adapter, test_loader, adj_t, rng_seed: int = 42) -> np.ndarray:
    """
    Compute the pooled (H, d) NRMSE curve the same way evaluation.regression
    does — sqrt(mean_(s,N)(sq_error)) — for the §4.5 consistency check.
    Streaming: accumulates sum of squared error, not the full arrays.
    """
    device = adj_t.device
    sq_sum = None   # will hold cumulative sum of squared error, shape (H, N, d)
    n_samples = 0
    for x, y in test_loader:
        torch.manual_seed(rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng_seed)
        x = x.to(device); y = y.to(device)
        with torch.no_grad():
            adapter.model.eval()
            pred = adapter.predict(x, adj_t)
        sq = ((pred - y) ** 2).cpu().numpy()   # (B, H, N, d)
        if sq_sum is None:
            sq_sum = sq.sum(axis=0)
        else:
            sq_sum += sq.sum(axis=0)
        n_samples += sq.shape[0]
    mse_per_hd = sq_sum / n_samples   # mean over samples → (H, N, d)
    mse_pooled = mse_per_hd.mean(axis=1)   # mean over N → (H, d)
    return np.sqrt(mse_pooled).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="saws")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/vpt_extended/per_sample"))
    p.add_argument("--eps", nargs="+", type=float, default=list(DEFAULT_EPS_GRID))
    p.add_argument("--force", action="store_true", help="Reprocess even if cached npz exists")
    p.add_argument("--rng-seed", type=int, default=42,
                   help="Seed torch's global RNG before every forward pass. "
                        "Required for MTGODE's non-deterministic adaptive-graph builder "
                        "(see specs/ADJACENCY_AUDIT.md).")
    p.add_argument("--consistency-rtol", type=float, default=1e-3,
                   help="Relative tolerance for the §4.5 pooled-consistency stop-the-line guard.")
    p.add_argument("--aggregate", nargs="+", type=int, default=None, metavar="SEED",
                   help="Instead of running one seed, aggregate existing per-seed "
                        "summaries into a cross-seed summary (spec §4.3). "
                        "Requires all listed seeds' vfh_summary.json to exist.")
    args = p.parse_args()

    if args.aggregate is not None:
        _do_cross_seed_aggregation(args)
        return

    seed_dir = args.out_dir / args.model / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    values_path = seed_dir / "vfh_values.npz"
    curves_path = seed_dir / "vfh_curves.npz"

    if values_path.exists() and curves_path.exists() and not args.force:
        log.info(f"Skipping seed {args.seed} — cached at {values_path}")
        return

    test_loader, adapter, adj_t, var_names, mean, std = _load_model_and_data(
        args.model, args.dataset, args.checkpoint, {}
    )
    curves = _streaming_per_sample_curves(adapter, test_loader, adj_t,
                                          rng_seed=args.rng_seed)   # (S, H, d)
    log.info(f"Per-sample curves shape: {curves.shape}, dtype {curves.dtype}, "
             f"peak abs {np.abs(curves).max():.4f}")

    # STOP-THE-LINE guard (spec §4.5): the S-axis RMS of per-sample curves must
    # reproduce the pooled per-horizon NRMSE the Phase-1 pipeline reports.
    # If they diverge, we are computing a DIFFERENT quantity than the rest of
    # the paper — abort before writing outputs.
    pooled = _pooled_curve_from_batches(adapter, test_loader, adj_t,
                                        rng_seed=args.rng_seed)
    ok, max_rel = pooled_curve_matches(curves, pooled, rtol=args.consistency_rtol)
    if not ok:
        raise RuntimeError(
            f"Pooled-consistency check FAILED (spec §4.5): max relative error "
            f"{max_rel:.2e} exceeds rtol={args.consistency_rtol:.2e}. "
            f"Per-sample curves are computing a different quantity than the "
            f"stored pooled NRMSE. STOP — investigate before proceeding."
        )
    log.info(f"Pooled-consistency check: max rel err = {max_rel:.2e} (rtol={args.consistency_rtol:.2e})  OK")

    # Compute VFH grid
    vfh = vfh_grid(curves, args.eps)
    log.info(f"VFH grid shape: {vfh.shape}")

    # Save
    np.savez_compressed(curves_path, nrmse=curves, eps_grid=np.array(args.eps))
    np.savez_compressed(values_path, vfh=vfh, eps_grid=np.array(args.eps))
    log.info(f"Wrote {curves_path} + {values_path}")

    # Also write summary
    H_cap = curves.shape[1]
    summary = summarise_vfh(vfh, H_cap=H_cap, var_names=var_names,
                            eps_grid=args.eps, curves_for_survival=curves)
    (seed_dir / "vfh_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote {seed_dir / 'vfh_summary.json'}")


if __name__ == "__main__":
    main()
