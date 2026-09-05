#!/usr/bin/env python3
"""HPO v2 confirmation stage — top-K candidates × N fresh seeds × full production
protocol. Pre-registered best-mean-val rule.

Consumes the `{study_name}_top3.json` written by `scripts/hpo_run_optuna.py`.
Runs each of the top-K configurations under a fresh set of seeds (default {42, 43, 44})
with the FULL production protocol (epoch cap 200, patience 15, full train split —
NOT the truncated HPO protocol). Selects the winner by best mean val MAE across
the fresh seeds; that winner alone is eligible to update `experiments/configs/mtgode.yaml`.

Why the confirmation stage exists (from the reviewer's spec):
  "TPE + fixed seed still leaves a winner conditioned on one seed and a truncated
  protocol. Top-3 × 3 fresh seeds × full production, pre-registered rule (best mean
  val MAE), then and only then the yaml updates."

Usage (after HPO v2 completes):
    python scripts/hpo_confirm.py \
        --top3 runs/hpo_optuna/hpo_mtgode_v2_top3.json \
        --seeds 42 43 44 \
        --out runs/hpo_optuna/confirmation

Task IDs: `hpo_confirm_mtgode_t2m_rank{R}_s{seed}`  (R = 1..3, seed ∈ user-provided).
Outputs per task: `runs/hpo_optuna/confirmation/{task_id}/` — standard trainer artefacts
(summary.json, test_metrics.json, per_station.npz, state.pt, checkpoint.pt, train.log).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def build_confirmation_config(sampled_hp: Dict[str, Any]) -> Dict[str, Any]:
    """Build a config for a confirmation-stage run.

    Loads base mtgode.yaml (paper-provenance) and applies the sampled top-K
    hyperparameters on top. Uses the FULL production protocol — NOT the truncated
    HPO protocol (no train_hours_cap, epoch cap from yaml=200, patience from
    yaml=15). Target = t2m.
    """
    from cli import load_config  # noqa: PLC0415

    config = load_config("mtgode", {})
    config["task"] = "regression"
    config["dataset"] = "saws"
    config.setdefault("data", {})
    config["data"]["target_variable"] = "t2m"
    # Explicitly ensure no train_hours_cap leaks in from anywhere
    config["data"].pop("train_hours_cap", None)

    for key, value in sampled_hp.items():
        _set_nested(config, key, value)

    return config


def run_one_confirmation(sampled_hp: Dict[str, Any], seed: int, task_id: str, out_root: Path) -> Dict[str, Any]:
    """Run one (config, seed) pair to completion. Returns the trainer's history."""
    from cli import load_adapter, _build_loss                 # noqa: PLC0415
    from preprocessing.pipeline import run_pipeline           # noqa: PLC0415
    from training.seed import seed_everything                 # noqa: PLC0415
    from training.trainer import Trainer, TrainingDiverged   # noqa: PLC0415
    from evaluation.regression import ERA5_VAR_NAMES         # noqa: PLC0415

    log = logging.getLogger(task_id)
    log.info("=" * 70)
    log.info("Confirmation stage: %s", task_id)
    log.info("sampled hp: %s", sampled_hp)
    log.info("seed: %d  (full production protocol)", seed)

    config = build_confirmation_config(sampled_hp)
    log.info("resolved config: epochs=%d patience=%d target=%s",
             config["training"]["epochs"],
             config["training"]["early_stopping_patience"],
             config["data"]["target_variable"])

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, adj, meta = run_pipeline(
        config=config, force_rerun=False,
    )
    config["meta"] = meta
    adj_device = adj.to(device) if isinstance(adj, torch.Tensor) else torch.from_numpy(adj).float().to(device)

    loss_fn = _build_loss("regression", train_loader, device)
    for key in list(sys.modules.keys()):
        if key == "model" or key.startswith("model."):
            del sys.modules[key]

    adapter_cls = load_adapter("mtgode")
    adapter = adapter_cls()
    adapter.build_model(config, adj_device.cpu())
    adapter.model.to(device)

    run_dir = out_root / task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        adapter=adapter,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        loss_fn=loss_fn, adj=adj_device, config=config,
        output_dir=run_dir, seed=seed, mlflow_enabled=False,
        norm_mean=meta["mean"], norm_std=meta["std"],
        thresholds=meta["thresholds"],
        var_names=meta.get("var_names", ERA5_VAR_NAMES),
        vpt_epsilon=float(config.get("evaluation", {}).get("vpt_epsilon", 0.3)),
    )

    t0 = time.time()
    try:
        result = trainer.train(stop_check=None, epoch_callback=None)
    except TrainingDiverged as e:
        log.error("DIVERGED at epoch=%d streak=%d", e.epoch, e.streak)
        return {"task_id": task_id, "seed": seed, "diverged": True,
                "epoch": e.epoch, "elapsed_seconds": time.time() - t0}
    elapsed = time.time() - t0

    val_hist = result.get("val_loss", [])
    best_val = float(min(val_hist)) if val_hist else float("inf")
    test_metrics = result.get("test_metrics", {})

    log.info("done  best_val=%.4f  stopped_epoch=%d  elapsed=%.0fs",
             best_val, result.get("stopped_epoch", -1), elapsed)

    del adapter, trainer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

    return {
        "task_id": task_id, "seed": seed,
        "diverged": False,
        "best_val_loss": best_val,
        "stopped_epoch": result.get("stopped_epoch"),
        "elapsed_seconds": elapsed,
        "test_metrics": test_metrics,
        "sampled_hp": sampled_hp,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top3", required=True, help="Path to {study}_top3.json from hpo_run_optuna.py")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="Fresh seeds to run each top-K config under (default 42 43 44)")
    p.add_argument("--out", default="runs/hpo_optuna/confirmation",
                   help="Output root for per-(rank, seed) run directories")
    p.add_argument("--top-k", type=int, default=3, help="How many top configs to confirm (default 3)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    log = logging.getLogger("hpo_confirm")

    with open(args.top3) as f:
        top3_data = json.load(f)
    top_configs = top3_data["top3"][:args.top_k]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("HPO v2 confirmation stage")
    log.info("top-%d configs from %s", len(top_configs), args.top3)
    log.info("seeds: %s", args.seeds)
    log.info("out: %s", out_root)
    log.info("=" * 70)

    all_results: List[Dict[str, Any]] = []
    for rank, cfg in enumerate(top_configs, start=1):
        for seed in args.seeds:
            task_id = f"hpo_confirm_mtgode_t2m_rank{rank}_s{seed}"
            summary_path = out_root / task_id / "summary.json"
            if summary_path.exists():
                log.info("SKIP %s (already complete)", task_id)
                with open(summary_path) as f:
                    all_results.append(json.load(f))
                continue

            r = run_one_confirmation(cfg["params"], seed, task_id, out_root)
            r["rank"] = rank
            r["hpo_val_mae"] = cfg["val_mae"]
            r["hpo_trial_number"] = cfg["trial_number"]
            all_results.append(r)
            # persist per-run summary
            with open(out_root / task_id / "confirmation_summary.json", "w") as f:
                json.dump(r, f, indent=2, default=str)

    # Aggregate: for each rank, best mean val MAE across seeds
    log.info("=" * 70)
    log.info("Confirmation results")
    log.info("=" * 70)
    by_rank: Dict[int, List[Dict[str, Any]]] = {}
    for r in all_results:
        by_rank.setdefault(r["rank"], []).append(r)

    ranked: List[Dict[str, Any]] = []
    for rank in sorted(by_rank):
        seed_runs = by_rank[rank]
        # Exclude diverged
        clean = [r for r in seed_runs if not r.get("diverged")]
        if not clean:
            log.warning("rank %d: ALL SEEDS DIVERGED", rank)
            continue
        vals = [r["best_val_loss"] for r in clean]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        test_mae_t2m = [r["test_metrics"].get("mae", {}).get("t2m") for r in clean
                        if r.get("test_metrics")]
        test_mae_mean = float(np.mean([v for v in test_mae_t2m if v is not None])) if test_mae_t2m else None

        hp = seed_runs[0]["sampled_hp"]
        entry = {
            "rank": rank, "hp": hp,
            "n_seeds": len(clean), "seeds": [r["seed"] for r in clean],
            "mean_val_mae": mean_val, "std_val_mae": std_val,
            "individual_val_mae": vals,
            "mean_test_mae_t2m": test_mae_mean,
        }
        ranked.append(entry)
        log.info("rank=%d  mean_val=%.4f (std=%.4f)  test_mae_t2m=%s  hp=%s",
                 rank, mean_val, std_val,
                 f"{test_mae_mean:.3f}" if test_mae_mean is not None else "n/a",
                 hp)

    # Winner: min mean_val_mae
    if ranked:
        winner = min(ranked, key=lambda e: e["mean_val_mae"])
        log.info("=" * 70)
        log.info("WINNER: rank %d (mean_val_MAE=%.4f, n=%d seeds)",
                 winner["rank"], winner["mean_val_mae"], winner["n_seeds"])
        log.info("hp: %s", winner["hp"])
        log.info("=" * 70)
        log.info("This is the config to fold into experiments/configs/mtgode.yaml")

    final_path = out_root / "confirmation_report.json"
    with open(final_path, "w") as f:
        json.dump({
            "seeds": args.seeds,
            "top_k": args.top_k,
            "top3_source": args.top3,
            "ranked": ranked,
            "winner": winner if ranked else None,
        }, f, indent=2, default=str)
    log.info("Confirmation report → %s", final_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
