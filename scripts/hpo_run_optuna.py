#!/usr/bin/env python3
"""HPO v2 — sequential Optuna TPE + MedianPruner for MTGODE on N=32 SAWS.

Supersedes HPO v1 (pre-sampled random search, parked at
`runs/hpo_v1_random_search_parked/`). Design rationale is documented in
`specs/HYPERPARAMETERS.md §HPO`; the summary is:

  - Sequential execution (single L40S) eliminates the NFS-concurrency failure
    class and the constant_liar/stale-trial interop the parallel Optuna variant
    would have needed. One overnight of wall-clock is the whole price.
  - MedianPruner requires live coupling between the training loop and the
    Optuna study (`trial.report()` mid-training + `trial.should_prune()`), so
    HPO runs in-process — the trainer is imported directly, not spawned via
    subprocess. See `training/trainer.py::train`'s `epoch_callback` hook.
  - Fixed trial seed (42 for every trial) removes the config-vs-seed confound
    that biased v1's marginals. Matches Chikwanha's design choice deliberately.
  - Reports best-so-far val (monotone) not this-epoch's val, so that the
    scheduler's cosine-annealing restart bumps at epochs 10, 30, ... don't
    cause a recovering trial to get median-pruned.
  - `n_warmup_steps=10` clears the first cosine-restart cycle before any
    pruning judgment; combined with best-so-far reporting, pruning fires only
    when the trial is genuinely worse-than-median in *cumulative* progress.
  - Startup guard marks any RUNNING trial FAILED before optimizing — under
    sequential execution any RUNNING trial at process start is necessarily
    dead from a killed prior job; ~10 lines that close the original Optuna
    rejection reason.

The winner from this script is NOT the winner in `experiments/configs/mtgode.yaml` —
that comes from the *confirmation stage* (`scripts/hpo_confirm.py`, top-K × 3
fresh seeds × full production protocol). This script's output is a ranked
short-list of candidates.

Usage — local smoke:
    python scripts/hpo_run_optuna.py --n-trials 2 --smoke \
        --storage sqlite:///runs/hpo_optuna_smoke/study.db

Usage — HPC production (submitted via sbatch):
    python scripts/hpo_run_optuna.py --n-trials 30 \
        --storage sqlite:///~/hpo/hpo_mtgode_v2.db
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import torch
import yaml

# repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Study configuration — all pins live here so the paper's methods section can
# be regenerated from this file alone.
# ---------------------------------------------------------------------------

# Trial-invariant seed. Every trial re-seeds torch/numpy/python to 42 at start,
# so trial comparisons attribute differences to hyperparameters not seed luck.
TRIAL_SEED = 42

# Sampler seed for TPE's own randomness. Independent of TRIAL_SEED so that
# study reproducibility (given identical trials-database) is decoupled from
# per-trial reproducibility.
SAMPLER_SEED = 42

# Per-model search spaces. Add a new entry to HPO another model.
# All knobs are categorical → readable marginals for the paper.
SEARCH_SPACES: Dict[str, Dict[str, list]] = {
    # MTGODE v2 (reviewer-approved: 5 knobs, boundary-extended lr)
    "mtgode": {
        "training.learning_rate":      [1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2],
        "training.batch_size":         [32, 64, 128],
        "model.dropout":               [0.1, 0.2, 0.3, 0.4],
        "model.hidden_dim":            [32, 64, 128],
        "model.graph_constructor.k":   [10, 13, 20],
    },
    # STGODE (Fang et al. KDD 2021). Post-2026-08-11 peer-review rewrite
    # (specs/STGODE_INTEGRATION_AUDIT.md):
    #   - Search space now targets paper-Eq-4 knobs (sigma_km, cutoff_km)
    #     and paper-Eq-6 (epsilon_dtw). Upstream z-scored knobs are ablation-
    #     only and NOT searched — see audit §C1.
    #   - `ode.time` is now a LIVE knob (H3 fix) — wrapper overrides upstream's
    #     hardcoded 6 via post-construction module walk. Kept in the space
    #     because integration horizon genuinely affects the ODE solution.
    #   - `model.dropout` NOT included: upstream TCN dropout hardcoded 0.2.
    #   - `data.input_len` NOT included: upstream STGCNBlock temporal_dim=12
    #     hardcoded — wrapper asserts input_len == 12.
    "stgode": {
        "training.learning_rate":      [1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 6e-3],
        "training.batch_size":         [8, 16, 32],
        "ode.time":                    [3.0, 6.0, 12.0],
        "model.sigma_km":              [100.0, 200.0, 500.0],   # paper Eq. 4 spatial Gaussian bandwidth
        "model.epsilon_dtw":           [0.3, 0.5, 0.8],         # paper Eq. 6 semantic threshold
    },
    # CLCRN (Lin et al. 2022 / Gaibie 2024). Mirrors Gen's AGCRN HPO structure
    # (batch_size, lr, input_len, num_layers, hidden_dim/rnn_units, embed_dim)
    # plus one CLCRN-specific knob (k_neighbors — graph connectivity). Grid
    # size = 6×3×2×3×3×3×3 = 2916; 30 trials (~1% sample) via 3-worker parallel.
    # lck_structure and max_view excluded: paper defaults are well-motivated
    # and Gaibie's HPO did not search them.
    # NOTE: HPO v2 ran under CosineAnnealingWarmRestarts. Winner (lr=6e-3,
    # input_len=24) applies to the new MultiStepLR campaign without a separate
    # sweep — first 10 epochs are identical under both schedules, and the
    # MedianPruner fires mainly in that window. Monitor seed-42 health gate.
    "clcrn": {
        "training.learning_rate": [1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2],
        "training.batch_size":    [32, 64, 128],
        "data.input_len":         [12, 24],           # Gaibie full search {12,24,48,72,168}; optimum=12
        "model.num_layers":       [1, 2, 3],
        "model.hidden_dim":       [32, 64, 128],      # rnn_units in upstream
        "model.embed_dim":        [8, 16, 32],
        "model.k_neighbors":      [10, 15, 25],       # CLCRN-specific: graph neighbourhood size
    },
    # AGCRN (Bai et al. 2020 / Gen's HPO). 6 knobs matching Gen's search dimensions
    # (lr, batch_size, input_len, num_layers, hidden_dim/rnn_units, embed_dim).
    # Gen searched lr (1e-4, 5e-2) log-continuous and embed_dim {1..30}; we
    # use discrete grids that cover her winner (lr=2e-4, embed_dim=30).
    # Grid size = 6×3×2×3×3×5 = 1620; 30 TPE trials via 3-worker parallel.
    # cheb_k excluded: Bai paper pins it at 2; not searched by Gen.
    "agcrn": {
        "training.learning_rate": [1e-4, 2e-4, 5e-4, 1e-3, 3e-3, 6e-3],
        "training.batch_size":    [32, 64, 128],
        "data.input_len":         [12, 24],
        "model.num_layers":       [1, 2, 3],
        "model.hidden_dim":       [16, 32, 64],       # rnn_units
        "model.embed_dim":        [10, 20, 30, 40, 50],  # Gen winner=30; extend upper boundary
    },
}

# Trial protocol — every trial trains under these settings (regardless of model)
TRIAL_PROTOCOL: Dict[str, Any] = {
    "data.target_variable":              "t2m",
    "training.epochs":                   60,     # raised from v1's 30; pruner absorbs the cost
    "training.early_stopping_patience":  5,      # Gaibie protocol
    "data.train_hours_cap":              17520,  # first 2 train years (Gaibie protocol)
}

# Pruner: MedianPruner with n_warmup_steps=10 (clears first cosine-restart cycle
# before pruning; with best-so-far reporting, only cumulative losers get killed)
PRUNER_N_WARMUP_STEPS = 10


# ---------------------------------------------------------------------------
# Startup guard — safe under sequential execution
# ---------------------------------------------------------------------------

def mark_stale_running_trials_failed(study: optuna.Study) -> int:
    """Any trial in RUNNING state at study-load time is necessarily dead.

    Under sequential execution (this script's only supported mode), only one
    trial can be RUNNING at a time. If we're just loading the study, no trial
    should currently be running — any RUNNING trial is a zombie from a killed
    prior process (walltime, OOM, SIGKILL). Mark them FAILED so TPE's
    posterior isn't polluted by results it will never receive.

    Returns count of trials marked FAILED.
    """
    stale = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    for t in stale:
        study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
    return len(stale)


# ---------------------------------------------------------------------------
# Per-trial training driver — in-process (imports our stack)
# ---------------------------------------------------------------------------

def build_trial_config(trial: optuna.trial.Trial, model: str) -> Dict[str, Any]:
    """Sample per-model knobs + return a full config dict for the trial."""
    from cli import load_config  # noqa: PLC0415

    if model not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for model {model!r}. "
                         f"Available: {list(SEARCH_SPACES)}")

    # Load base yaml (paper-provenance config for this model)
    config = load_config(model, {})
    config["task"] = "regression"
    config["dataset"] = "saws"

    # 1. Sampled hyperparameters (categorical for readable marginals)
    sampled: Dict[str, Any] = {}
    for key, choices in SEARCH_SPACES[model].items():
        v = trial.suggest_categorical(key, choices)
        # numpy → python native (optuna serialisation)
        if hasattr(v, "item"):
            v = v.item()
        sampled[key] = v
        _set_nested(config, key, v)

    # 2. Trial protocol (pinned; every trial gets these)
    for key, v in TRIAL_PROTOCOL.items():
        _set_nested(config, key, v)

    trial.set_user_attr("sampled_hyperparams", sampled)
    trial.set_user_attr("model", model)
    return config


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """`_set_nested(d, 'model.graph_constructor.k', 20)` sets d['model']['graph_constructor']['k']."""
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def run_one_trial(trial: optuna.trial.Trial, args) -> float:
    """Run one Optuna trial in-process. Returns best-so-far val MAE."""
    from cli import load_adapter, _build_loss                 # noqa: PLC0415
    from preprocessing.pipeline import run_pipeline           # noqa: PLC0415
    from training.seed import seed_everything                 # noqa: PLC0415
    from training.trainer import Trainer, TrainingDiverged   # noqa: PLC0415
    from evaluation.regression import ERA5_VAR_NAMES         # noqa: PLC0415

    log = logging.getLogger(f"trial_{trial.number:02d}")
    task_id = f"hpo_{args.model}_t2m_trial_{trial.number:02d}"
    log.info("=" * 70)
    log.info("HPO v2 trial %02d starting (model=%s task_id=%s)",
             trial.number, args.model, task_id)

    # Build config for the model this study is HPO'ing
    config = build_trial_config(trial, args.model)
    sampled = trial.user_attrs["sampled_hyperparams"]
    log.info("sampled: %s", sampled)
    log.info("protocol: %s", TRIAL_PROTOCOL)

    # Fixed trial seed (search conditioned on seed=42 for every trial)
    seed_everything(TRIAL_SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing — cache-hit path (hash defaults dataset-aware)
    train_loader, val_loader, test_loader, adj, meta = run_pipeline(
        config=config, force_rerun=False,
    )
    log.info("pipeline: train samples=%d, val samples=%d, test samples=%d",
             len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    # Apply train_hours_cap — sub-sample train_loader.dataset (Gaibie protocol)
    train_hours_cap = config.get("data", {}).get("train_hours_cap")
    if train_hours_cap is not None:
        from torch.utils.data import Subset, DataLoader  # noqa: PLC0415
        ds = train_loader.dataset
        max_start = int(train_hours_cap) - (ds.input_len + ds.horizon)
        n_kept = min(max_start, len(ds))
        subset = Subset(ds, list(range(n_kept)))
        # Mirror univariate-slice + label attrs onto Subset (Subset doesn't proxy)
        for attr in ("target_variable_index", "input_len", "horizon",
                     "task", "label_mode", "label_indices",
                     "features", "labels"):
            if hasattr(ds, attr):
                setattr(subset, attr, getattr(ds, attr))
        train_loader = DataLoader(
            subset, batch_size=train_loader.batch_size, shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=getattr(train_loader, "pin_memory", False),
            drop_last=getattr(train_loader, "drop_last", False),
        )
        log.info("train_hours_cap=%d: train samples %d → %d",
                 train_hours_cap, len(ds), n_kept)

    config["meta"] = meta
    adj_device = adj.to(device) if isinstance(adj, torch.Tensor) else torch.from_numpy(adj).float().to(device)

    # Loss + adapter (with sys.modules purge for adapter cross-loading safety)
    loss_fn = _build_loss("regression", train_loader, device)
    # Same purge pattern as run_worker.py — safe here because we're in-process
    # across many trials of the same model
    for key in list(sys.modules.keys()):
        if key == "model" or key.startswith("model."):
            del sys.modules[key]

    adapter_cls = load_adapter(args.model)
    adapter = adapter_cls()
    adapter.build_model(config, adj_device.cpu())
    adapter.model.to(device)

    # Per-trial output dir under runs/hpo_optuna/ (invisible to production)
    run_dir = Path(args.hpo_root) / "runs" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    vpt_epsilon = float(config.get("evaluation", {}).get("vpt_epsilon", 0.3))
    trainer = Trainer(
        adapter=adapter,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        loss_fn=loss_fn, adj=adj_device, config=config,
        output_dir=run_dir, seed=TRIAL_SEED, mlflow_enabled=False,
        norm_mean=meta["mean"], norm_std=meta["std"],
        thresholds=meta["thresholds"],
        var_names=meta.get("var_names", ERA5_VAR_NAMES),
        vpt_epsilon=vpt_epsilon,
    )

    # HPO hook: report best-so-far val + check should_prune each epoch
    best_val_so_far = float("inf")

    def epoch_callback(epoch: int, val_loss: float, best_val_loss: float) -> bool:
        nonlocal best_val_so_far
        # `best_val_loss` from trainer is already the best-so-far by construction
        best_val_so_far = min(best_val_so_far, best_val_loss)
        # Report best-so-far (monotone) so cosine restarts don't spuriously prune
        trial.report(best_val_so_far, step=epoch)
        # MedianPruner decides
        if trial.should_prune():
            log.info("epoch %d: pruned by MedianPruner (best_val=%.4f)",
                     epoch, best_val_so_far)
            return True
        return False

    t0 = time.time()
    try:
        result = trainer.train(stop_check=None, epoch_callback=epoch_callback)
    except TrainingDiverged as e:
        elapsed = time.time() - t0
        log.error("DIVERGED at epoch=%d streak=%d (elapsed=%.0fs)", e.epoch, e.streak, elapsed)
        # Optuna: treat as failed trial with worst-case objective
        raise optuna.TrialPruned(f"diverged at epoch {e.epoch}")

    elapsed = time.time() - t0
    stopped_reason = result.get("stopped_reason", "completed")
    stopped_epoch = result.get("stopped_epoch", -1)

    # If we pruned mid-training, propagate to Optuna
    if stopped_reason == "pruned":
        log.info("trial %d PRUNED at epoch %d in %.0fs (best_val=%.4f)",
                 trial.number, stopped_epoch, elapsed, best_val_so_far)
        raise optuna.TrialPruned(f"pruned at epoch {stopped_epoch}")

    val_history = result.get("val_loss", [])
    best_val = float(min(val_history)) if val_history else float("inf")

    log.info("trial %d COMPLETE  best_val=%.4f  stopped_epoch=%d  elapsed=%.0fs",
             trial.number, best_val, stopped_epoch, elapsed)

    # Save the trial's outputs alongside Optuna's own record so we have
    # provenance for the confirmation stage (top-3 needs its exact configs)
    trial_summary = {
        "task_id": task_id,
        "trial_number": trial.number,
        "sampled": sampled,
        "protocol": TRIAL_PROTOCOL,
        "seed": TRIAL_SEED,
        "best_val_loss": best_val,
        "stopped_epoch": stopped_epoch,
        "stopped_reason": stopped_reason,
        "elapsed_seconds": elapsed,
        "val_loss": val_history,
        "train_loss": result.get("train_loss", []),
    }
    with open(run_dir / "trial_summary.json", "w") as f:
        json.dump(trial_summary, f, indent=2, default=str)

    # cleanup for next trial
    del adapter, trainer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

    return best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(SEARCH_SPACES.keys()),
                   help=f"Which model to HPO. Available: {list(SEARCH_SPACES.keys())}")
    p.add_argument("--n-trials", type=int, default=30, help="Optuna trials to run (default 30)")
    p.add_argument("--study-name", default=None,
                   help="Optuna study name (default: hpo_{model}_v2)")
    p.add_argument("--storage", default=None,
                   help="Optuna storage URI (default: sqlite:///~/hpo/{study_name}.db)")
    p.add_argument("--hpo-root", default=None,
                   help="Where per-trial outputs land (default: runs/hpo_optuna_{model})")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: override protocol to 3 epochs, patience 999, train_hours_cap 8760 (1yr)")
    p.add_argument("--parallel", action="store_true",
                   help="Multi-worker parallel mode: skip startup guard (other workers' RUNNING "
                        "trials are live, not dead) and run exactly --n-trials per worker against "
                        "the shared study. Launch 3 workers each with --n-trials 10 for a 30-trial study.")
    args = p.parse_args()

    # Per-model defaults derived from --model
    if args.study_name is None:
        args.study_name = f"hpo_{args.model}_v2"
    if args.storage is None:
        args.storage = f"sqlite:///{Path.home()}/hpo/{args.study_name}.db"
    if args.hpo_root is None:
        args.hpo_root = f"runs/hpo_optuna_{args.model}"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    log = logging.getLogger("hpo_v2")

    # Smoke overrides: shorten trial time drastically for wiring verification
    if args.smoke:
        global TRIAL_PROTOCOL
        TRIAL_PROTOCOL = {
            **TRIAL_PROTOCOL,
            "training.epochs":                   3,
            "training.early_stopping_patience":  999,
            "data.train_hours_cap":              8760,   # 1 year
        }
        log.warning("SMOKE MODE: epochs=3, patience=999, train_hours_cap=8760 (1yr)")

    # Ensure storage dir exists
    if args.storage.startswith("sqlite:///"):
        db_path = Path(args.storage.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure hpo_root exists
    Path(args.hpo_root).mkdir(parents=True, exist_ok=True)
    (Path(args.hpo_root) / "runs").mkdir(exist_ok=True)

    log.info("=" * 70)
    log.info("HPO v2 (sequential Optuna TPE + MedianPruner) — %s on N=32 SAWS", args.model.upper())
    log.info("study_name=%s  storage=%s", args.study_name, args.storage)
    log.info("hpo_root=%s  n_trials=%d  smoke=%s", args.hpo_root, args.n_trials, args.smoke)
    log.info("SEARCH_SPACE (%s): %s", args.model, SEARCH_SPACES[args.model])
    log.info("TRIAL_PROTOCOL: %s", TRIAL_PROTOCOL)
    log.info("TRIAL_SEED=%d  SAMPLER_SEED=%d  PRUNER_N_WARMUP_STEPS=%d",
             TRIAL_SEED, SAMPLER_SEED, PRUNER_N_WARMUP_STEPS)
    log.info("=" * 70)

    # Create / resume study
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=SAMPLER_SEED)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=PRUNER_N_WARMUP_STEPS)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler, pruner=pruner,
        load_if_exists=True,
    )

    if args.parallel:
        # Multi-worker mode: SKIP startup guard (other workers' RUNNING trials
        # are live, not zombies). All workers share the SQLite study DB; the
        # callback terminates each worker once the study reaches n_trials complete.
        log.info("Parallel mode: startup guard SKIPPED; running until %d complete "
                 "trials across all workers", args.n_trials)

        def _stop_when_complete(study: optuna.Study,
                                trial: optuna.trial.FrozenTrial) -> None:
            n_done = sum(1 for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
            if n_done >= args.n_trials:
                study.stop()

        study.optimize(
            lambda t: run_one_trial(t, args),
            n_trials=None,   # unlimited per worker; callback stops at target
            callbacks=[_stop_when_complete],
            catch=(Exception,),
        )
    else:
        # Sequential mode: startup guard (any RUNNING trial is a zombie)
        n_stale = mark_stale_running_trials_failed(study)
        if n_stale > 0:
            log.warning("Startup guard: marked %d stale RUNNING trial(s) as FAILED "
                        "(sequential mode → any RUNNING at start is dead)", n_stale)

        finished_states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                           optuna.trial.TrialState.FAIL)
        n_finished = len(study.get_trials(deepcopy=False, states=finished_states))
        n_remaining = max(0, args.n_trials - n_finished)
        log.info("Existing finished trials: %d, target: %d, remaining: %d",
                 n_finished, args.n_trials, n_remaining)

        if n_remaining == 0:
            log.info("Study already at target. Nothing to run.")
        else:
            study.optimize(
                lambda t: run_one_trial(t, args),
                n_trials=n_remaining,
                catch=(Exception,),
            )

    # ----------------------------------------------------------------------
    # Post-run report — top trials, importances, dump full trials.csv
    # ----------------------------------------------------------------------
    log.info("=" * 70)
    log.info("HPO v2 complete")
    log.info("=" * 70)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    log.info("Trials: %d complete, %d pruned, %d failed",
             len(completed), len(pruned), len(failed))

    if not completed:
        log.warning("No completed trials to report on.")
        return 1

    # Best + top-3 (for the confirmation stage)
    completed_sorted = sorted(completed, key=lambda t: t.value)
    top3 = completed_sorted[:3]
    log.info("\n=== TOP 3 (for confirmation stage) ===")
    for i, t in enumerate(top3):
        log.info("  #%d  trial=%d  val_MAE=%.4f  params=%s",
                 i + 1, t.number, t.value, t.params)

    # Full trials.csv
    csv_path = Path(args.hpo_root) / f"{args.study_name}_trials.csv"
    study.trials_dataframe().to_csv(csv_path, index=False)
    log.info("Full trials table → %s", csv_path)

    # Top-3 config → JSON for the confirmation stage to consume
    top3_path = Path(args.hpo_root) / f"{args.study_name}_top3.json"
    with open(top3_path, "w") as f:
        json.dump({
            "model": args.model,
            "study_name": args.study_name,
            "storage": args.storage,
            "trial_seed": TRIAL_SEED,
            "sampler_seed": SAMPLER_SEED,
            "pruner_n_warmup_steps": PRUNER_N_WARMUP_STEPS,
            "search_space": SEARCH_SPACES[args.model],
            "trial_protocol": TRIAL_PROTOCOL,
            "top3": [
                {"trial_number": t.number, "val_mae": t.value, "params": t.params}
                for t in top3
            ],
        }, f, indent=2)
    log.info("Top-3 configs → %s (feeds scripts/hpo_confirm.py)", top3_path)

    # Parameter importances (fanova) — best-effort
    log.info("\n=== PARAMETER IMPORTANCES (fanova) ===")
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, imp in importances.items():
            log.info("  %-40s  %.3f", param, imp)
    except Exception as e:
        log.warning("could not compute importances: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
