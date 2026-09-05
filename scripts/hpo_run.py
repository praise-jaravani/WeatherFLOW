"""
Optuna hyperparameter sweep for a single model per specs/OPTUNA_HPO.md.

Per-trial workflow:
  1. Sample hyperparameters from the model's search space (TPE).
  2. Rewrite the model yaml with the sampled values.
  3. Invoke cli.py as a subprocess to train + evaluate 1 seed (seed = trial_number + 42).
  4. Parse the run's history.json for the best val_loss → the trial's objective.
  5. Save all trial artefacts (config, history, summary, stdout) under
     outputs/hpo/{model}/trials/trial_{N}/ so nothing is lost.

Storage: Optuna SQLite at outputs/hpo/{model}/study.db — resumable across
job restarts, and directly queryable for report generation.
"""

from __future__ import annotations

import os
import sys
_ROOT = "/home/praise/Computer_Science_Resources/Weatherflow"
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import argparse
import copy
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import optuna
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hpo_run")


# ---------------------------------------------------------------------------
# Search spaces (spec §Search spaces)
# ---------------------------------------------------------------------------

def sample_agcrn(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "data.input_len":   trial.suggest_categorical("data.input_len",   [12, 24, 48, 72]),
        "model.embed_dim":  trial.suggest_categorical("model.embed_dim",  [8, 10, 16, 32]),
        "model.cheb_k":     trial.suggest_categorical("model.cheb_k",     [2, 3]),
        "model.dropout":    trial.suggest_categorical("model.dropout",    [0.0, 0.1, 0.2, 0.3]),
    }


def sample_clcrn(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "data.input_len":            trial.suggest_categorical("data.input_len",            [12, 24, 48, 72]),
        "model.hidden_dim":          trial.suggest_categorical("model.hidden_dim",          [16, 32, 64, 128]),
        "training.learning_rate":    trial.suggest_categorical("training.learning_rate",    [0.0003, 0.001, 0.003, 0.01, 0.03, 0.05]),
        "model.k_neighbors":         trial.suggest_categorical("model.k_neighbors",         [10, 25, 40]),
        "model.embed_dim":           trial.suggest_categorical("model.embed_dim",           [8, 16, 32]),
    }


def sample_mtgode(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "data.input_len":                    trial.suggest_categorical("data.input_len",                    [12, 24, 48, 72]),
        "model.hidden_dim":                  trial.suggest_categorical("model.hidden_dim",                  [20, 40, 80]),
        "training.learning_rate":            trial.suggest_categorical("training.learning_rate",            [0.0003, 0.001, 0.003]),
        "model.graph_constructor.k":         trial.suggest_categorical("model.graph_constructor.k",         [5, 10, 20]),
        "ode.step_size_1":                   trial.suggest_categorical("ode.step_size_1",                   [0.2, 0.4, 0.6]),
    }


SAMPLERS = {
    "agcrn":  sample_agcrn,
    "clcrn":  sample_clcrn,
    "mtgode": sample_mtgode,
}


# ---------------------------------------------------------------------------
# Config surgery
# ---------------------------------------------------------------------------

def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """`_set_nested(d, "model.graph_constructor.k", 20)` sets d['model']['graph_constructor']['k']."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def build_trial_config(model_name: str, sampled: Dict[str, Any]) -> Dict[str, Any]:
    """Load the base yaml + apply sampled overrides in-memory."""
    base_path = Path("experiments/configs") / f"{model_name}.yaml"
    with open(base_path) as f:
        config = yaml.safe_load(f)
    for key, val in sampled.items():
        _set_nested(config, key, val)
    return config


# ---------------------------------------------------------------------------
# One trial: subprocess cli.py, parse history.json, return best val loss
# ---------------------------------------------------------------------------

def _cli_overrides_from_sampled(sampled: Dict[str, Any]) -> list[str]:
    args = []
    for k, v in sampled.items():
        args.extend([f"--config.{k}", str(v)])
    return args


def _find_latest_run_dir(model: str, dataset: str, task: str, since_ts: float) -> Path:
    """Find the cli.py-created outputs/{model}_{dataset}_{task}_YYYYMMDD_HHMMSS/ dir
    that was written after `since_ts`. This is how we locate the trial's own outputs."""
    root = Path("outputs")
    candidates = sorted(root.glob(f"{model}_{dataset}_{task}_*"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.stat().st_mtime >= since_ts:
            return c
    raise FileNotFoundError(f"no cli.py output dir found for {model}_{dataset}_{task}_* since {since_ts}")


def _find_latest_run_dir_or_none(model: str, dataset: str, task: str, since_ts: float):
    """Non-raising version — returns None if no match. Used for timeout salvage."""
    try:
        return _find_latest_run_dir(model, dataset, task, since_ts)
    except FileNotFoundError:
        return None


def run_one_trial(trial: optuna.trial.Trial, args) -> float:
    sampler = SAMPLERS[args.model]
    sampled = sampler(trial)
    log.info(f"[trial {trial.number}] sampled: {sampled}")

    trial_dir = Path("outputs/hpo") / args.model / "trials" / f"trial_{trial.number:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Persist the hyperparams before we run — if the trial dies we still know what was sampled
    (trial_dir / "hyperparams.json").write_text(json.dumps(sampled, indent=2))

    # Build the CLI invocation
    seed = 42 + trial.number
    epochs = args.smoke_epochs or 60
    cli_args = [
        sys.executable, "cli.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--task", "regression",
        "--runs", "1",
        "--seed-start", str(seed),
        "--mlflow-off",
        "--no-cache",   # ensure input_len change actually rebuilds the cache
        "--config.training.epochs", str(epochs),
    ] + _cli_overrides_from_sampled(sampled)

    log.info(f"[trial {trial.number}] launching cli.py (epochs={epochs}, seed={seed}, "
             f"timeout={args.trial_timeout_sec}s)")

    since_ts = time.time()
    stdout_log = trial_dir / "stdout.log"
    with open(stdout_log, "w") as f:
        try:
            proc = subprocess.run(
                cli_args, stdout=f, stderr=subprocess.STDOUT,
                timeout=args.trial_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - since_ts
            log.warning(f"[trial {trial.number}] EXCEEDED {args.trial_timeout_sec}s timeout "
                        f"(ran {elapsed:.0f}s); killed. Attempting to salvage partial history.")
            partial = _find_latest_run_dir_or_none(args.model, args.dataset, "regression", since_ts)

            # Salvage strategy 1: history.json (only exists if training completed
            # all epochs — usually NOT the case when we hit the timeout mid-training).
            if partial is not None:
                for fname in ["history.json"]:
                    matches = list(partial.rglob(fname))
                    if matches:
                        shutil.copy(matches[0], trial_dir / fname)
            hist_path = trial_dir / "history.json"
            if hist_path.exists():
                hist_raw = json.loads(hist_path.read_text())
                one = (hist_raw if "val_loss" in hist_raw
                       else next(iter(hist_raw.get("seed_histories", {}).values()), {}))
                vals = one.get("val_loss", [])
                if vals:
                    best_partial = float(min(vals))
                    log.warning(f"[trial {trial.number}] salvaged from history.json: "
                                f"val_loss = {best_partial:.5f} across {len(vals)} epochs")
                    return best_partial

            # Salvage strategy 2: parse train.log — written after every epoch,
            # so it survives mid-training kills. Format per trainer.py:
            #   'epoch=1 train_loss=1.798770 val_loss=1.673841 grad_norm=...'
            if partial is not None:
                train_logs = list(partial.rglob("train.log"))
                if train_logs:
                    log_text = train_logs[0].read_text()
                    val_losses = []
                    import re
                    for m in re.finditer(r"epoch=(\d+)\s+train_loss=([\d.eE+-]+)\s+val_loss=([\d.eE+-]+)", log_text):
                        try:
                            val_losses.append(float(m.group(3)))
                        except ValueError:
                            continue
                    # Copy the train.log into the trial dir for provenance
                    shutil.copy(train_logs[0], trial_dir / "train.log")
                    if val_losses:
                        best_partial = float(min(val_losses))
                        log.warning(f"[trial {trial.number}] salvaged from train.log: "
                                    f"val_loss = {best_partial:.5f} across {len(val_losses)} epochs")
                        # Also write a stub history.json so the reporting code
                        # can process this trial the same way as completed ones
                        (trial_dir / "history.json").write_text(json.dumps({
                            "train_loss": [float("nan")] * len(val_losses),  # unknown from log parse
                            "val_loss":   val_losses,
                            "stopped_epoch": len(val_losses),
                            "_salvaged_from": "train.log",
                            "_timeout_sec": args.trial_timeout_sec,
                        }))
                        return best_partial

            # Nothing salvageable — trial didn't complete any epochs before kill
            log.warning(f"[trial {trial.number}] no val_loss data to salvage; pruning trial")
            raise optuna.exceptions.TrialPruned(
                f"Trial exceeded {args.trial_timeout_sec}s and no salvageable epoch history"
            )

    if proc.returncode != 0:
        log.error(f"[trial {trial.number}] cli.py exited {proc.returncode}; see {stdout_log}")
        raise optuna.exceptions.TrialPruned(f"cli.py failed with rc={proc.returncode}")

    # Find + copy the run's outputs into the trial dir
    try:
        run_dir = _find_latest_run_dir(args.model, args.dataset, "regression", since_ts)
    except FileNotFoundError as e:
        log.error(str(e))
        raise optuna.exceptions.TrialPruned(str(e))

    for fname in ["history.json", "summary.json"]:
        src = run_dir / "seed_" / fname   # cli.py places these under seed_{seed}/
        # Actually cli.py's layout is per-seed; walk it:
        matches = list(run_dir.rglob(fname))
        if matches:
            shutil.copy(matches[0], trial_dir / fname)

    # Save the exact config used
    config_used = build_trial_config(args.model, sampled)
    with open(trial_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(config_used, f, sort_keys=False)

    # Extract objective: best val_loss from history.json
    hist_path = trial_dir / "history.json"
    if not hist_path.exists():
        raise optuna.exceptions.TrialPruned("no history.json produced")
    hist = json.loads(hist_path.read_text())

    # cli.py stores per-seed history under keys; the payload may be either a
    # dict-of-seeds or a single history. Handle both.
    if "seed_histories" in hist:
        one_seed_history = list(hist["seed_histories"].values())[0]
    elif "val_loss" in hist:
        one_seed_history = hist
    else:
        # try the outer list format
        one_seed_history = hist[0] if isinstance(hist, list) else hist

    val_losses = one_seed_history.get("val_loss", [])
    if not val_losses:
        raise optuna.exceptions.TrialPruned("no val_loss recorded in history")
    best_val = float(min(val_losses))
    log.info(f"[trial {trial.number}] best val_loss = {best_val:.5f}")
    return best_val


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(SAMPLERS.keys()))
    p.add_argument("--dataset", default="saws")
    p.add_argument("--n-trials", type=int, default=25,
                   help="Optuna TPE trials. Default 25 per Option C (2026-08-04): "
                        "fits in 24h HPC walltime with per-trial timeout margin.")
    p.add_argument("--trial-timeout-sec", type=int, default=3300,
                   help="Per-trial wall-time cap in seconds. Default 3300s (55 min). "
                        "If a trial exceeds this, the subprocess is killed; if it wrote "
                        "any val_loss values to history.json before the kill, the best of "
                        "those is reported as the objective. Otherwise the trial is pruned.")
    p.add_argument("--storage-uri", default=None,
                   help="Optuna storage URI. Default: sqlite:///outputs/hpo/{model}/study.db")
    p.add_argument("--sampler-seed", type=int, default=42,
                   help="TPE sampler seed for reproducibility")
    p.add_argument("--smoke-epochs", type=int, default=None,
                   help="Override epochs (for local smoke testing; production leaves at 60)")
    p.add_argument("--timeout-sec", type=int, default=None,
                   help="Optuna study timeout in seconds (None = no timeout)")
    args = p.parse_args()

    hpo_dir = Path("outputs/hpo") / args.model
    hpo_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage_uri or f"sqlite:///{hpo_dir}/study.db"
    study_name = f"hpo_{args.model}_saws"

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    log.info(f"Study: {study_name} @ {storage}")
    log.info(f"Existing trials: {len(study.trials)}, target: {args.n_trials}")

    remaining = args.n_trials - len(study.trials)
    if remaining <= 0:
        log.info("Study already at target trials; nothing to do.")
        return

    study.optimize(
        lambda t: run_one_trial(t, args),
        n_trials=remaining,
        timeout=args.timeout_sec,
        catch=(Exception,),   # a broken trial should not kill the study
    )

    # Persist the winner config for easy paste-in
    if study.best_trial is not None:
        winner_cfg = build_trial_config(args.model, study.best_trial.params)
        winner_path = hpo_dir / "winner_config.yaml"
        with open(winner_path, "w") as f:
            yaml.safe_dump(winner_cfg, f, sort_keys=False)
        log.info(f"Winner (val_loss={study.best_value:.5f}) → {winner_path}")


if __name__ == "__main__":
    main()
