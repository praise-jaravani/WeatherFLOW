#!/usr/bin/env python3
"""Pre-sampled random-search HPO manifest generator (Gaibie protocol).

Design: reviewer decision 2026-08-08 (see specs/HYPERPARAMETERS.md §HPO status).
Rejected: Optuna sequential TPE (sequential in a parallel-worker world) and
Optuna distributed (SQLite on NFS is a silent-corruption path).

This generator writes the exact artefacts §11 of HPC_Parallel_Worker_System.md
identified as missing: a manifest that the existing worker/claim/sweep
machinery consumes untouched, plus per-trial config yamls the worker deep-merges
onto the base model config.

Trial protocol (per reviewer + Gaibie 2024):
  - target = t2m (all trials)
  - epoch cap = 30, early_stopping_patience = 5
  - train_hours_cap = 17520 (first ~2 train years) — worker applies via a
    post-pipeline dataset truncation
  - seed(trial N) = 42 + N (unique per trial; deterministic)
  - task_id format = {model}_{target}_s{42+N}  (fits existing parse_task_id)

Search space (5 knobs — everything else pinned at paper-provenance values):
  training.learning_rate         ∈ {1e-4, 3e-4, 1e-3, 3e-3}
  training.batch_size            ∈ {32, 64, 128}
  model.dropout                  ∈ {0.1, 0.2, 0.3, 0.4}
  model.hidden_dim               ∈ {32, 64, 128}      # conv_channels only, decoupled from node_dim per audit
  model.graph_constructor.k      ∈ {10, 13, 20}       # 13 is N=32 density target

Pinned (do NOT sample — per 2026-08-08 audit + reviewer):
  data.input_len = 12                  # receptive-field R=25 at Jin's 1.0/0.25 solver; 48/72 silently truncate
  model.graph_constructor.node_dim = 40, alpha = 3
  ode.adjoint = false                  # Finding 0
  ode.cgp_alpha = 2.0                  # Finding 5 — paper Eq. 6
  ode.time_1/step_size_1/time_2/step_size_2 = 1.0/0.25/1.0/0.25

Outputs (all under --out-dir, default runs/hpo/):
  manifest.json          — list of task IDs, one per trial
  configs/{task_id}.yaml — per-trial config: base yaml + sampled overrides + protocol
  spec.json              — reproducibility record: sampler seed, ranges, sampled values

Usage:
  python scripts/build_hpo_manifest.py --model mtgode --n-trials 30 \\
      --sampler-seed 42 --out-dir runs/hpo

  # Smoke: 1 trial, reduced protocol
  python scripts/build_hpo_manifest.py --model mtgode --n-trials 1 \\
      --sampler-seed 42 --out-dir runs/hpo_smoke \\
      --epochs 3 --patience 999 --train-hours-cap 8760
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Task ID convention lives in the manifest module — reuse it so we don't drift.
from orchestration.manifest import make_task_id  # noqa: E402


# ---------------------------------------------------------------------------
# Search space per model. MTGODE is the only model in scope for this HPO
# (reviewer 2026-08-08: baselines adopt Gaibie's published SAWS optima).
# ---------------------------------------------------------------------------

SEARCH_SPACES = {
    "mtgode": {
        "training.learning_rate":        [1e-4, 3e-4, 1e-3, 3e-3],
        "training.batch_size":           [32, 64, 128],
        "model.dropout":                 [0.1, 0.2, 0.3, 0.4],
        "model.hidden_dim":              [32, 64, 128],
        "model.graph_constructor.k":     [10, 13, 20],
    },
}


def _set_nested(d: dict, dotted_key: str, value):
    """`_set_nested(d, 'model.graph_constructor.k', 20)` sets d['model']['graph_constructor']['k']."""
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def sample_trials(
    model: str,
    n_trials: int,
    sampler_seed: int,
) -> list[dict]:
    """Return a list of `n_trials` sampled hyperparameter dicts.

    Random search per Gaibie's protocol (reviewer decision 2026-08-08).
    Uses one seeded RNG so re-running with the same (model, n_trials, sampler_seed)
    triple produces identical trials — reproducibility for the paper's HPO section.
    """
    if model not in SEARCH_SPACES:
        raise ValueError(f"no HPO search space defined for model {model!r}")
    space = SEARCH_SPACES[model]
    rng = np.random.default_rng(sampler_seed)

    trials = []
    for _ in range(n_trials):
        sampled = {key: rng.choice(values).item() for key, values in space.items()}
        # Cast to native python types (numpy scalars serialise ugly)
        for k, v in list(sampled.items()):
            if hasattr(v, "item"):
                sampled[k] = v.item()
        trials.append(sampled)
    return trials


def build_trial_config(
    base_config: dict,
    sampled: dict,
    target: str,
    epochs: int,
    patience: int,
    train_hours_cap: int,
) -> dict:
    """Return a deep-copied config with the sampled hyperparams + trial protocol applied.

    Protocol overrides always win (they encode reviewer + Gaibie protocol) so a
    trial can never accidentally drift from the disclosed HPO regime.
    """
    cfg = copy.deepcopy(base_config)
    # 1. Sampled hyperparameters
    for key, value in sampled.items():
        _set_nested(cfg, key, value)
    # 2. Trial protocol (fixed for every trial)
    _set_nested(cfg, "data.target_variable",              target)
    _set_nested(cfg, "training.epochs",                   int(epochs))
    _set_nested(cfg, "training.early_stopping_patience",  int(patience))
    _set_nested(cfg, "data.train_hours_cap",              int(train_hours_cap))
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, choices=list(SEARCH_SPACES.keys()))
    ap.add_argument("--n-trials", type=int, required=True,
                    help="how many trials to pre-sample")
    ap.add_argument("--sampler-seed", type=int, default=42,
                    help="numpy RNG seed for the random search (default 42)")
    ap.add_argument("--out-dir", default="runs/hpo",
                    help="output directory (default runs/hpo)")
    ap.add_argument("--target", default="t2m",
                    help="target variable for all trials (default t2m — Gaibie)")
    ap.add_argument("--epochs", type=int, default=30,
                    help="training.epochs cap per trial (default 30 — Gaibie)")
    ap.add_argument("--patience", type=int, default=5,
                    help="early_stopping_patience per trial (default 5 — Gaibie)")
    ap.add_argument("--train-hours-cap", type=int, default=17520,
                    help="cap train samples to first N hours (default 17520 = 2 years — Gaibie)")
    ap.add_argument("--base-config", default=None,
                    help="path to base yaml (default: experiments/configs/{model}.yaml)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    configs_dir = out_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_config) if args.base_config else \
        REPO / "experiments" / "configs" / f"{args.model}.yaml"
    with open(base_path) as f:
        base = yaml.safe_load(f)

    trials_hparams = sample_trials(args.model, args.n_trials, args.sampler_seed)

    manifest = []
    for i, sampled in enumerate(trials_hparams):
        seed = 42 + i
        task_id = make_task_id(args.model, args.target, seed)
        manifest.append(task_id)
        cfg = build_trial_config(
            base, sampled,
            target=args.target,
            epochs=args.epochs,
            patience=args.patience,
            train_hours_cap=args.train_hours_cap,
        )
        cfg_path = configs_dir / f"{task_id}.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    # Persist manifest — the worker reads this file via existing load_manifest()
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Persist reproducibility record. This is what goes into the paper's HPO
    # section — every trial's sample is here, plus the sampler seed to reproduce.
    spec = {
        "model": args.model,
        "n_trials": args.n_trials,
        "sampler": "random_search",
        "sampler_seed": args.sampler_seed,
        "search_space": SEARCH_SPACES[args.model],
        "trial_protocol": {
            "target": args.target,
            "epochs": args.epochs,
            "patience": args.patience,
            "train_hours_cap": args.train_hours_cap,
        },
        "base_config": str(base_path.relative_to(REPO) if base_path.is_relative_to(REPO) else base_path),
        "trials": [
            {"task_id": tid, "seed": 42 + i, "sampled": s}
            for i, (tid, s) in enumerate(zip(manifest, trials_hparams))
        ],
    }
    spec_path = out_dir / "spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    print(f"wrote {len(manifest)} trials → {manifest_path}")
    print(f"per-trial configs → {configs_dir}/")
    print(f"reproducibility record → {spec_path}")
    print()
    print("First 3 trials:")
    for i in range(min(3, len(trials_hparams))):
        print(f"  trial {i} (task {manifest[i]}): {trials_hparams[i]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
