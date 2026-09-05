"""
Generate HPO report artefacts for one or more models. Reads from
outputs/hpo/{model}/ and writes report tables + figures under
outputs/hpo/{model}/report/. Also writes a cross-model summary
outputs/hpo/summary.md ready to paste into the paper's methods section.
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

import pandas as pd

from evaluation.hpo_reporting import build_full_report, load_trial_log, summarise_study

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _fmt_winner_row(model: str, summary: dict, log_df: pd.DataFrame) -> str:
    """Markdown row for the cross-model summary table."""
    if not summary or "best_params" not in summary:
        return f"| {model} | (no completed trials) | — | — |"
    best = summary["best_params"]
    val_stats = summary.get("value_stats", {})
    best_val = summary.get("best_value", float("nan"))
    n_done = summary.get("n_completed", 0)
    n_total = summary.get("n_trials", 0)
    knob_str = ", ".join(f"{k}={v}" for k, v in sorted(best.items()))
    return (f"| {model} | val_loss = {best_val:.5f} | "
            f"{knob_str} | {n_done}/{n_total} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", default=["agcrn", "mtgode", "clcrn"])
    p.add_argument("--hpo-root", type=Path, default=Path("outputs/hpo"))
    args = p.parse_args()

    args.hpo_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for model in args.model:
        model_dir = args.hpo_root / model
        if not model_dir.exists():
            log.warning(f"No {model} directory found at {model_dir}; skipping")
            continue
        log.info(f"Building report for {model}…")
        summaries[model] = build_full_report(model, hpo_root=args.hpo_root)

    # Cross-model summary
    lines = [
        "# HPO Sweep Summary",
        "",
        "Per specs/OPTUNA_HPO.md — one comprehensive TPE sweep per model on the SAWS network.",
        "",
        "## Winners",
        "",
        "| Model | Best Objective (val_loss) | Winner Hyperparameters | Trials Complete |",
        "|---|---|---|---|",
    ]
    for model, summary in summaries.items():
        log_df = load_trial_log(args.hpo_root / model)
        lines.append(_fmt_winner_row(model, summary, log_df))

    # Per-model parameter importance
    lines.append("")
    lines.append("## Hyperparameter Importance (Optuna fanova)")
    lines.append("")
    for model, summary in summaries.items():
        imp = summary.get("param_importance", {})
        if not imp:
            continue
        lines.append(f"### {model.upper()}")
        lines.append("")
        lines.append("| Hyperparameter | Importance |")
        lines.append("|---|---|")
        for k, v in sorted(imp.items(), key=lambda x: -x[1]):
            lines.append(f"| `{k}` | {v:.3f} |")
        lines.append("")

    lines.append("## Per-model report artefacts")
    lines.append("")
    for model in summaries:
        lines.append(f"- **{model.upper()}**: [`report/trial_log.csv`](./{model}/report/trial_log.csv), "
                     f"[`learning_curves`](./{model}/report/learning_curves_all_trials.png), "
                     f"[`param_importance`](./{model}/report/parameter_importance.png), "
                     f"[`per_hyperparameter_scatter`](./{model}/report/per_hyperparameter_scatter.png), "
                     f"[`winner_config.yaml`](./{model}/winner_config.yaml)")

    (args.hpo_root / "summary.md").write_text("\n".join(lines))
    log.info(f"Wrote cross-model summary → {args.hpo_root / 'summary.md'}")


if __name__ == "__main__":
    main()
