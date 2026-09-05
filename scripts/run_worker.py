#!/usr/bin/env python3
"""Run_Plan_Spec §1.2 / §4 worker.

Loop:
    install SIGUSR1 handler → set stop_requested
    while True:
        if stop_requested or remaining_walltime < margin + EMA(seed_time):
            exit 0
        task = next_incomplete(manifest, claims, runs, my_jobid)
        if task is None: exit 0
        model, target, seed = parse(task)
        prepare_pipeline_and_trainer_for(model, target, seed, run_dir=runs/task/)
        try:
            result = trainer.train(stop_check=_stop_check)
        except TrainingDiverged:
            # FAILED_DIVERGED marker already written by trainer — keep claim,
            # move on (never re-attempt a divergent config, §5).
            continue
        if result.get("stopped_reason") == "signal":
            # Keep claim so successor resumes from state.pt.
            exit 0
        write_summary_json(run_dir, result)
        update_ema(model, epoch_time)

EMA seeded at full-cap per D3 refinement: AGCRN 18min, MTGODE 2.44h, CLCRN 2.6h.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from orchestration.manifest import (
    CLASSIFICATION_TOKEN,
    load as load_manifest,
    parse_task_id,
    summary_path,
    is_complete,
)
from orchestration.claim import next_incomplete_task, live_squeue_jobids


# ---------------------------------------------------------------------------
# sys.modules 'model' collision guard (production bug fix 2026-08-06)
# ---------------------------------------------------------------------------
# AGCRN, CLCRN, and MTGODE each vendor an upstream repo under a bare `model`
# package name and rely on `sys.path` injection to import it. Within a single
# Python process, once one adapter imports its `model` package, subsequent
# adapters that try `from model.XXX import ...` hit the WRONG package
# (whichever is in sys.modules['model']).
#
# In production this hit us when a worker process cycled from MTGODE →
# AGCRN (2 procs died on 1184528 at 10:03 SAST on Aug 6). The adjacency
# probe (scripts/probe_adjacency.py) already solved this — we lift its
# pattern verbatim here: snapshot sys.path at import time, then purge
# sys.modules['model'] + `model.*` and restore sys.path before each new
# adapter is instantiated.
_INITIAL_SYS_PATH = list(sys.path)


def _purge_shared_model_module() -> None:
    """Purge shared 'model' package from sys.modules + restore sys.path.

    Called before every task's adapter.build_model() so the freshly-selected
    adapter loads its OWN vendored `model` package rather than a prior
    adapter's leftover.
    """
    for key in list(sys.modules.keys()):
        if key == "model" or key.startswith("model."):
            del sys.modules[key]
    sys.path[:] = list(_INITIAL_SYS_PATH)


# EMA seed (in seconds), per Run_Plan_Spec §7 measured pilot data.
DEFAULT_EMA_SECONDS = {
    "agcrn":  18 * 60,           # 18 min
    "mtgode": int(2.44 * 3600),  # ~2.44 h
    "clcrn":  int(2.6 * 3600),   # ~2.6 h
}


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_STOP_REQUESTED = False


def _install_signal_handler():
    def handler(signum, frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        print(f"[worker] SIGUSR1 received at t={time.time():.0f}; "
              "graceful stop after current epoch", flush=True)
    signal.signal(signal.SIGUSR1, handler)


def _stop_check() -> bool:
    return _STOP_REQUESTED


# ---------------------------------------------------------------------------
# Walltime remaining (best-effort)
# ---------------------------------------------------------------------------

def _walltime_remaining_s() -> int | None:
    """Read remaining walltime from SLURM via `scontrol show job -o`.

    Returns None if not in a SLURM job or scontrol unavailable. Failure is
    non-fatal — the worker just doesn't self-limit on walltime (SIGUSR1 from
    #SBATCH --signal will still fire).
    """
    jobid = os.environ.get("SLURM_JOB_ID")
    if not jobid:
        return None
    try:
        out = subprocess.check_output(
            ["scontrol", "show", "job", "-o", jobid],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return None

    # Parse "TimeLimit=HH:MM:SS RunTime=HH:MM:SS" (or D-HH:MM:SS)
    def _parse_dur(s: str) -> int | None:
        if s in ("UNLIMITED", "NOT_SET"):
            return None
        try:
            days = 0
            if "-" in s:
                d, s = s.split("-", 1)
                days = int(d)
            parts = [int(x) for x in s.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0)
            h, m, sec = parts[-3], parts[-2], parts[-1]
            return days * 86400 + h * 3600 + m * 60 + sec
        except (ValueError, IndexError):
            return None

    limit = run = None
    for tok in out.split():
        if tok.startswith("TimeLimit="):
            limit = _parse_dur(tok.split("=", 1)[1])
        elif tok.startswith("RunTime="):
            run = _parse_dur(tok.split("=", 1)[1])
    if limit is None or run is None:
        return None
    return max(0, limit - run)


# ---------------------------------------------------------------------------
# EMA of per-model wall time
# ---------------------------------------------------------------------------

def _ema_path(runs_dir: Path) -> Path:
    return runs_dir / "ema_seconds.json"


def load_ema(runs_dir: Path) -> dict[str, float]:
    p = _ema_path(runs_dir)
    if not p.exists():
        return dict(DEFAULT_EMA_SECONDS)
    try:
        with open(p) as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_EMA_SECONDS)
    ema = dict(DEFAULT_EMA_SECONDS)
    ema.update({k: float(v) for k, v in saved.items() if isinstance(v, (int, float))})
    return ema


def save_ema(runs_dir: Path, ema: dict[str, float]) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    tmp = _ema_path(runs_dir).with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(ema, f)
    os.replace(tmp, _ema_path(runs_dir))


def update_ema(ema: dict[str, float], model: str, observed_seconds: float, alpha: float = 0.3) -> None:
    """Exponential moving average — α=0.3 gives a ~4-run half-life."""
    prev = ema.get(model, observed_seconds)
    ema[model] = (1 - alpha) * prev + alpha * observed_seconds


# ---------------------------------------------------------------------------
# Per-task training driver — reuses cli.py building blocks but with fixed
# per-task output_dir and single-seed single-target config
# ---------------------------------------------------------------------------

def _deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge `src` into `dst` (src wins). Returns dst for chaining.

    Used for HPO trial overrides: base model yaml is loaded first, then the
    per-trial yaml under `--hpo-configs-dir` is merged on top. Existing keys
    in dst are overwritten; nested dicts merged recursively so a trial can
    override e.g. `model.graph_constructor.k` without wiping the whole
    `graph_constructor` sub-block.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _prepare_and_run_task(
    task_id: str, runs_dir: Path, stop_check_fn,
    epochs_override: int | None = None,
    hpo_configs_dir: Path | None = None,
) -> tuple[str, float]:
    """Run one task to completion (or graceful stop, or divergence).

    Returns (result_state, elapsed_seconds) where result_state ∈
    {"complete", "stopped", "diverged"}.

    A "complete" run has summary.json written and the trainer's test eval
    finished. "stopped" preserves state.pt and the claim so a successor
    resumes. "diverged" has FAILED_DIVERGED marker; claim retained (§5).

    If `hpo_configs_dir` is provided and `{hpo_configs_dir}/{task_id}.yaml`
    exists, it is deep-merged onto the base model config. This is how the
    HPO manifest generator (scripts/build_hpo_manifest.py) injects per-trial
    hyperparameters without touching the manifest/claim/sweep machinery.
    """
    from cli import load_config           # noqa: PLC0415
    from cli import load_adapter          # noqa: PLC0415
    from cli import _build_loss           # noqa: PLC0415
    from preprocessing.pipeline import run_pipeline  # noqa: PLC0415
    from training.seed import seed_everything       # noqa: PLC0415
    from training.trainer import Trainer, TrainingDiverged  # noqa: PLC0415
    from evaluation.regression import ERA5_VAR_NAMES        # noqa: PLC0415

    model, target, seed = parse_task_id(task_id)
    is_classification = target == CLASSIFICATION_TOKEN

    # cli.load_config takes overrides as list[str] like ["training.epochs=3"];
    # for the worker we just mutate the loaded config directly.
    config = load_config(model, {})
    config["dataset"] = "saws"
    config.setdefault("data", {})
    if is_classification:
        # Phase-2 classification: multi-label per run (t2m/tp/rh/ws jointly).
        # No target_variable — the trainer subsets the model's per-variable
        # channel output to the label variables listed in the model yaml
        # (data.label_variables, defaults set per Phase-2 plan §2.1).
        config["task"] = "classification"
        config["data"].pop("target_variable", None)
    else:
        config["task"] = "regression"
        config["data"]["target_variable"] = target

    # HPO trial-config merge (deep). Applied BEFORE epochs_override so a --epochs
    # CLI flag still wins over a per-trial yaml (matches existing precedence).
    if hpo_configs_dir is not None:
        trial_cfg_path = Path(hpo_configs_dir) / f"{task_id}.yaml"
        if trial_cfg_path.exists():
            import yaml  # noqa: PLC0415
            with open(trial_cfg_path) as f:
                trial_overrides = yaml.safe_load(f) or {}
            _deep_merge(config, trial_overrides)
            print(f"[worker] HPO trial config merged from {trial_cfg_path}", flush=True)

    if epochs_override is not None:
        config.setdefault("training", {})["epochs"] = int(epochs_override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing — cache-hit path (all tasks share the same cache since
    # input_len/horizon/preproc keys are constant across the 120-task matrix
    # and across HPO trials — none of the sampled/pinned HPO knobs are in
    # `preproc_keys` inside compute_data_hash).
    train_loader, val_loader, test_loader, adj, meta = run_pipeline(
        config=config, force_rerun=False,
    )

    # HPO trial protocol: cap train samples to first N hours (Gaibie protocol).
    # Val + test loaders untouched. Post-pipeline so the shared cache stays hot.
    train_hours_cap = config.get("data", {}).get("train_hours_cap")
    if train_hours_cap is not None:
        from torch.utils.data import Subset, DataLoader   # noqa: PLC0415
        ds = train_loader.dataset
        # Sliding windows starting at t < train_hours_cap - (input_len + horizon)
        # keep the whole window inside the first `train_hours_cap` hours.
        max_start = int(train_hours_cap) - (ds.input_len + ds.horizon)
        if max_start < 1:
            raise ValueError(
                f"train_hours_cap={train_hours_cap} too small for "
                f"input_len+horizon={ds.input_len + ds.horizon}"
            )
        n_kept = min(max_start, len(ds))
        subset = Subset(ds, list(range(n_kept)))
        # Trainer._compute_loss and _evaluate_test read
        # `loader.dataset.target_variable_index` to slice the model's D-channel
        # output down to the single univariate target channel (Gaibie regime).
        # torch.utils.data.Subset does NOT proxy attributes to the underlying
        # dataset, so we mirror the univariate-related attrs onto the Subset
        # instance itself. Without this, MAE/MSE loss silently broadcasts a
        # (B,H,N,1) target across a (B,H,N,D) prediction — a wrong-loss bug
        # that only surfaces with train_hours_cap set. Found via HPO smoke
        # 2026-08-08.
        for attr in ("target_variable_index", "input_len", "horizon",
                     "task", "label_mode", "label_indices",
                     "features", "labels"):
            if hasattr(ds, attr):
                setattr(subset, attr, getattr(ds, attr))
        train_loader = DataLoader(
            subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=getattr(train_loader, "pin_memory", False),
            drop_last=getattr(train_loader, "drop_last", False),
        )
        print(f"[worker] HPO train_hours_cap={train_hours_cap}: "
              f"train samples {len(ds)} → {n_kept}", flush=True)

    config["meta"] = meta
    adj_device = adj.to(device) if isinstance(adj, torch.Tensor) else torch.from_numpy(adj).float().to(device)
    adj_cpu = adj_device.cpu()

    # Loss + adapter. For classification, _build_loss reads the training
    # loader's label tensor to compute per-variable pos_weight for WBCE.
    loss_fn = _build_loss(config["task"], train_loader, device)
    # sys.modules 'model' collision guard: purge stale vendored `model`
    # package from a previous task (bug hit in production 2026-08-06).
    _purge_shared_model_module()
    adapter_cls = load_adapter(model)
    seed_everything(seed)
    adapter = adapter_cls()
    adapter.build_model(config, adj_cpu)
    n_params = adapter.get_param_count()
    adapter.model.to(device)

    run_dir = runs_dir / task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    vpt_epsilon = float(config.get("evaluation", {}).get("vpt_epsilon", 0.3))

    trainer = Trainer(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        adj=adj_device,
        config=config,
        output_dir=run_dir,
        seed=seed,
        mlflow_enabled=False,
        norm_mean=meta["mean"],
        norm_std=meta["std"],
        thresholds=meta["thresholds"],
        static_thresholds=meta.get("static_thresholds"),
        var_names=meta.get("var_names", ERA5_VAR_NAMES),
        vpt_epsilon=vpt_epsilon,
    )

    t0 = time.time()
    try:
        result = trainer.train(stop_check=stop_check_fn)
    except TrainingDiverged as e:
        elapsed = time.time() - t0
        print(f"[worker] {task_id} DIVERGED at epoch={e.epoch} streak={e.streak}", flush=True)
        return "diverged", elapsed
    elapsed = time.time() - t0

    if result.get("stopped_reason") == "signal":
        print(f"[worker] {task_id} stopped by signal — state saved, claim retained", flush=True)
        return "stopped", elapsed

    # Write per-task summary.json — this is the completion marker checked
    # by is_complete() and by manifest_incomplete.py.
    #
    # test_metrics is task-shape-agnostic (whatever Trainer.train returned):
    # - regression: per-variable dicts + nrmse_per_horizon + scalar aggregates
    # - classification: per-variable {tau, f1, auc_roc, pr_auc, tp/fp/fn/tn, ...}
    # Aggregators in cli.py._aggregate_metrics already branch on shape.
    summary = {
        "task_id": task_id,
        "task": config["task"],
        "model": model,
        "target": target,   # "cls" for classification, variable name for regression
        "seed": seed,
        "n_params":         n_params,
        "n_train_samples":  len(train_loader.dataset),
        "n_val_samples":    len(val_loader.dataset),
        "n_test_samples":   len(test_loader.dataset),
        "stopped_epoch": result.get("stopped_epoch"),
        "train_loss": result.get("train_loss", []),
        "val_loss": result.get("val_loss", []),
        "grad_norm": result.get("grad_norm", []),
        "test_metrics": result.get("test_metrics", {}),
        "loss_type": result.get("loss_type"),
        "elapsed_seconds": elapsed,
        "completed_ts": time.time(),
    }
    tmp = run_dir / "summary.json.tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    os.replace(tmp, run_dir / "summary.json")

    print(f"[worker] {task_id} complete in {elapsed:.0f}s "
          f"(stopped_epoch={result.get('stopped_epoch')})", flush=True)
    return "complete", elapsed


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/manifest.json")
    ap.add_argument("--claims-dir", default="runs/claims")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--wallclock-margin", type=int, default=1500,
                    help="reserve this many seconds before walltime end "
                         "(default 1500 = 25 min; add EMA(model) on top)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config.training.epochs (for smoke tests)")
    ap.add_argument("--hpo-configs-dir", default=None,
                    help="if set, look for {hpo_configs_dir}/{task_id}.yaml and "
                         "deep-merge onto the base model config before training. "
                         "Written by scripts/build_hpo_manifest.py.")
    args = ap.parse_args()

    # Configure root logger so preprocessing.pipeline / preprocessing.dataset
    # / preprocessing.impute log.info() calls become visible in the worker log.
    # Without this the worker looks hung between "HPO trial config merged" and
    # the trainer's first "epoch=" line (typically several minutes on the SAWS
    # cache — cache load + WindowDataset construction + adapter build). Force
    # is set because the container image sometimes pre-configures a stderr
    # handler at WARNING; force=True installs ours regardless. (2026-08-08.)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    _install_signal_handler()
    my_jobid = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
    manifest_path = Path(args.manifest)
    claims_dir = Path(args.claims_dir)
    runs_dir = Path(args.runs_dir)

    print(f"[worker] starting jobid={my_jobid} manifest={manifest_path} "
          f"claims={claims_dir} runs={runs_dir} margin={args.wallclock_margin}s",
          flush=True)

    manifest = load_manifest(manifest_path)
    ema = load_ema(runs_dir)

    while True:
        if _stop_check():
            print("[worker] stop_requested True — exiting cleanly", flush=True)
            return 0

        remaining = _walltime_remaining_s()
        if remaining is not None:
            # Reserve wallclock_margin + EMA(worst-case model still in queue).
            # Simpler: reserve margin + max(EMA over all models) to guarantee
            # we can complete whatever we pick.
            need = args.wallclock_margin + int(max(ema.values()))
            if remaining < need:
                print(f"[worker] walltime {remaining}s < need {need}s "
                      f"(margin {args.wallclock_margin} + max EMA {int(max(ema.values()))}) "
                      "— exiting cleanly", flush=True)
                return 0

        task_id = next_incomplete_task(
            manifest, claims_dir, runs_dir, my_jobid,
        )
        if task_id is None:
            print("[worker] manifest drained — exiting", flush=True)
            return 0

        model, _target, _seed = parse_task_id(task_id)
        print(f"[worker] claimed {task_id} (model={model}, "
              f"ema={ema.get(model, 0):.0f}s)", flush=True)

        result, elapsed = _prepare_and_run_task(
            task_id, runs_dir, _stop_check,
            epochs_override=args.epochs,
            hpo_configs_dir=Path(args.hpo_configs_dir) if args.hpo_configs_dir else None,
        )

        if result == "complete":
            update_ema(ema, model, elapsed)
            save_ema(runs_dir, ema)
        elif result == "stopped":
            # Successor takes over; exit cleanly.
            return 0
        # 'diverged' → keep claim (never re-attempted), move on to next task


if __name__ == "__main__":
    sys.exit(main())
