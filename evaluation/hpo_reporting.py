"""
HPO reporting — aggregate Optuna study data + per-trial artefacts into
tables and figures for the paper's methods/hyperparameters section.

Reads:
  outputs/hpo/{model}/study.db                 (Optuna SQLite)
  outputs/hpo/{model}/trials/trial_*/          (per-trial history + config)

Produces:
  outputs/hpo/{model}/report/
    trial_log.csv                    — all trials, ranked by val_loss
    winner_config.yaml               — best-trial config (paste-ready)
    learning_curves_all_trials.png   — overlaid per-trial val-loss curves
    parameter_importance.png         — Optuna fanova
    per_hyperparameter_scatter.png   — objective vs each knob's value

The report figures require matplotlib + optuna; the tabular outputs work
with pandas alone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

log = logging.getLogger(__name__)


def load_trial_log(model_dir: Path) -> pd.DataFrame:
    """
    Read all trial_*/history.json + hyperparams.json under model_dir/trials/
    and return a DataFrame ranked by best val_loss ascending.

    Each row is one trial with columns:
      trial_number, best_val_loss, n_epochs_run, stopped_epoch, wall_time_sec,
      plus every hyperparameter that was sampled (as its own column).
    """
    trials_dir = model_dir / "trials"
    if not trials_dir.exists():
        return pd.DataFrame()
    rows = []
    for tdir in sorted(trials_dir.iterdir()):
        if not tdir.is_dir():
            continue
        hp_path = tdir / "hyperparams.json"
        hist_path = tdir / "history.json"
        if not (hp_path.exists() and hist_path.exists()):
            log.warning(f"Skipping incomplete trial dir: {tdir.name}")
            continue
        hp = json.loads(hp_path.read_text())
        hist_raw = json.loads(hist_path.read_text())
        # cli.py's history format can be nested per-seed; pick the one series
        if "val_loss" in hist_raw:
            hist = hist_raw
        elif "seed_histories" in hist_raw:
            hist = next(iter(hist_raw["seed_histories"].values()))
        elif isinstance(hist_raw, list) and hist_raw:
            hist = hist_raw[0]
        else:
            log.warning(f"Skipping {tdir.name}: history.json format unrecognised")
            continue
        val_losses: List[float] = hist.get("val_loss", [])
        train_losses: List[float] = hist.get("train_loss", [])
        if not val_losses:
            continue
        row = {
            "trial_number":    int(tdir.name.replace("trial_", "")),
            "best_val_loss":   float(min(val_losses)),
            "best_epoch":      int(val_losses.index(min(val_losses))) + 1,
            "final_val_loss":  float(val_losses[-1]),
            "final_train_loss": float(train_losses[-1]) if train_losses else float("nan"),
            "n_epochs_run":    len(val_losses),
            "stopped_epoch":   int(hist.get("stopped_epoch", len(val_losses))),
        }
        row.update(hp)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("best_val_loss").reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def summarise_study(model_dir: Path) -> Dict[str, Any]:
    """
    Read the Optuna SQLite study and return a summary dict:
      n_trials, n_completed, best_value, best_params,
      value_stats {min, mean, median, max}, param_importance {param: importance}.
    """
    import optuna
    db = model_dir / "study.db"
    if not db.exists():
        return {}
    study_name = f"hpo_{model_dir.name}_saws"
    storage = f"sqlite:///{db}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        log.error(f"Could not load study {study_name}: {e}")
        return {}
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return {"n_trials": len(study.trials), "n_completed": 0}
    values = [t.value for t in completed]
    summary = {
        "n_trials":     len(study.trials),
        "n_completed":  len(completed),
        "best_value":   float(study.best_value),
        "best_params":  dict(study.best_params),
        "value_stats": {
            "min":    float(min(values)),
            "mean":   float(sum(values) / len(values)),
            "median": float(sorted(values)[len(values) // 2]),
            "max":    float(max(values)),
        },
    }
    # Parameter importance (fanova) — requires at least 2 completed trials
    if len(completed) >= 2:
        try:
            imp = optuna.importance.get_param_importances(study)
            summary["param_importance"] = {k: float(v) for k, v in imp.items()}
        except Exception as e:
            log.warning(f"Could not compute param importance: {e}")
            summary["param_importance"] = {}
    return summary


def plot_learning_curves(model_dir: Path, out_path: Path) -> None:
    """
    Overlay every trial's val-loss curve, colour by rank (best = red, others = grey).
    Requires matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_trial_log(model_dir)
    if df.empty:
        log.warning("No trial data to plot")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    trials_dir = model_dir / "trials"
    best_trial_id = int(df.iloc[0]["trial_number"])
    for _, row in df.iterrows():
        tid = int(row["trial_number"])
        hist_path = trials_dir / f"trial_{tid:02d}" / "history.json"
        if not hist_path.exists():
            continue
        raw = json.loads(hist_path.read_text())
        hist = raw if "val_loss" in raw else next(iter(raw.get("seed_histories", {}).values()), raw)
        vals = hist.get("val_loss", [])
        if not vals:
            continue
        color = "crimson" if tid == best_trial_id else "grey"
        alpha = 1.0 if tid == best_trial_id else 0.4
        lw    = 2.0 if tid == best_trial_id else 0.6
        label = f"trial {tid} (BEST)" if tid == best_trial_id else None
        ax.plot(range(1, len(vals) + 1), vals, color=color, alpha=alpha, linewidth=lw, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title(f"{model_dir.name.upper()} HPO trials — validation loss per epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_importance(model_dir: Path, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = summarise_study(model_dir)
    imp = summary.get("param_importance", {})
    if not imp:
        return
    keys = list(imp.keys())
    vals = [imp[k] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(keys, vals, color="steelblue")
    ax.set_xlabel("Importance (Optuna fANOVA)")
    ax.set_title(f"{model_dir.name.upper()} — hyperparameter importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_hyperparameter_scatter(model_dir: Path, out_path: Path) -> None:
    """One subplot per sampled hyperparameter: x=knob value, y=best_val_loss.
    Reveals which knobs drove the winning region."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load_trial_log(model_dir)
    if df.empty:
        return
    fixed_cols = {"rank", "trial_number", "best_val_loss", "best_epoch",
                  "final_val_loss", "final_train_loss", "n_epochs_run", "stopped_epoch"}
    knob_cols = [c for c in df.columns if c not in fixed_cols]
    if not knob_cols:
        return
    n = len(knob_cols)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    for i, knob in enumerate(knob_cols):
        ax = axes[i // cols][i % cols]
        ax.scatter(df[knob], df["best_val_loss"], c=df["rank"], cmap="viridis_r", s=40)
        # Mark the winner
        best_row = df.iloc[0]
        ax.scatter([best_row[knob]], [best_row["best_val_loss"]],
                   marker="*", s=200, color="crimson", edgecolor="black", zorder=5, label="best")
        ax.set_xlabel(knob)
        ax.set_ylabel("best val_loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"{model_dir.name.upper()} HPO — objective vs each knob")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def build_full_report(model: str, hpo_root: Path = Path("outputs/hpo")) -> Dict[str, Any]:
    """
    End-to-end report generation for one model.

    Writes:
      {hpo_root}/{model}/report/
        trial_log.csv
        study_summary.json
        learning_curves_all_trials.png
        parameter_importance.png
        per_hyperparameter_scatter.png

    Returns a dict of the summary for the paper's methods section.
    """
    model_dir = hpo_root / model
    report_dir = model_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Tabular outputs
    log_df = load_trial_log(model_dir)
    log_df.to_csv(report_dir / "trial_log.csv", index=False)
    summary = summarise_study(model_dir)
    (report_dir / "study_summary.json").write_text(json.dumps(summary, indent=2))

    # Figures
    try:
        plot_learning_curves(model_dir, report_dir / "learning_curves_all_trials.png")
    except Exception as e:
        log.warning(f"Learning curves plot failed: {e}")
    try:
        plot_parameter_importance(model_dir, report_dir / "parameter_importance.png")
    except Exception as e:
        log.warning(f"Param importance plot failed: {e}")
    try:
        plot_per_hyperparameter_scatter(model_dir, report_dir / "per_hyperparameter_scatter.png")
    except Exception as e:
        log.warning(f"Per-hyperparameter scatter failed: {e}")

    return summary
