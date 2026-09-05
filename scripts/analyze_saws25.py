#!/usr/bin/env python3
"""Aggregate SAWS 25-node campaign summaries + write per-model results.

Walks results/runs/<model>_<target>_s<seed>/summary.json and produces:
  - results/phase1_regression/<model>_regression_summary.json
    Per-model aggregate: mean/std/median per (target, metric) across seeds
  - results/plots/mae_bar.png             (per-model×target MAE with baselines)
  - results/plots/val_convergence.png     (val-loss curves per (model, target, seed))
  - results/plots/SUMMARY.md              (paste-ready tables + observations)

Idempotent: safe to re-run after every sync. Only completed tasks (summary.json
present) are aggregated; in-progress tasks (state.pt only) are ignored.

Usage:
  python scripts/analyze_saws25.py                # default paths
  python scripts/analyze_saws25.py --runs results/runs --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from orchestration.manifest import MODEL_ORDER, TARGET_ORDER, parse_task_id


def discover_completed(runs_dir: Path) -> List[dict]:
    """Return list of {model, target, seed, summary, task_id, per_station_path}."""
    out = []
    for d in sorted(runs_dir.iterdir()):
        sp = d / "summary.json"
        if not sp.exists():
            continue
        try:
            model, target, seed = parse_task_id(d.name)
        except ValueError:
            continue
        with open(sp) as f:
            s = json.load(f)
        out.append({
            "task_id": d.name,
            "model": model, "target": target, "seed": seed,
            "summary": s,
            "per_station": d / "per_station.npz" if (d / "per_station.npz").exists() else None,
        })
    return out


def _extract_scalar(m: dict, key: str) -> float | None:
    v = m.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def aggregate_per_model(records: List[dict]) -> Dict[str, dict]:
    """{model: {n_seeds, targets: {target: {metric: {mean,std,median,n,per_seed}}}}}"""
    by_model: Dict[str, list] = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)

    out: Dict[str, dict] = {}
    for model, recs in by_model.items():
        model_agg: Dict[str, dict] = {
            "model": model,
            "task": "regression",
            "dataset": "saws_25node",
            "n_records": len(recs),
            "targets": {},
            "epochs_actually_run": {},
        }
        # per-target
        by_target: Dict[str, list] = defaultdict(list)
        for r in recs:
            by_target[r["target"]].append(r)
        for target, trecs in by_target.items():
            seeds = sorted(r["seed"] for r in trecs)
            metrics = [r["summary"].get("test_metrics", {}) for r in trecs]

            # test_metrics keys we care about — some are scalars (mae_pooled_norm),
            # others are dicts keyed by target variable (mae={"t2m": 2.93}).
            # For dict-valued keys we extract m[key][target] which is the physical
            # per-variable metric for THIS univariate run's target.
            scalar_keys = {"mae_pooled_norm", "inference_ms_per_sample", "vpt_epsilon"}
            dict_keys = {"mae", "rmse", "mse", "smape",
                         "mae_extreme", "mae_nonextreme", "peak_intensity_error", "vpt"}

            metric_agg: Dict[str, dict] = {}

            def _agg(name: str, values: list):
                values = [v for v in values if v is not None and np.isfinite(v)]
                if not values:
                    return None
                return {
                    "mean": float(np.mean(values)),
                    "std":  float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "n":  len(values),
                    "per_seed": {str(s): v for s, v in zip(seeds, values)},
                }

            for k in scalar_keys:
                vals = [m.get(k) for m in metrics if isinstance(m, dict)]
                vals = [float(v) for v in vals if isinstance(v, (int, float))]
                if vals:
                    got = _agg(k, vals)
                    if got: metric_agg[k] = got

            # Physical metrics — keyed by target inside the dict
            for k in dict_keys:
                vals = []
                for m in metrics:
                    if not isinstance(m, dict): continue
                    sub = m.get(k, {})
                    if isinstance(sub, dict) and target in sub and isinstance(sub[target], (int, float)):
                        vals.append(float(sub[target]))
                got = _agg(k, vals)
                if got: metric_agg[k] = got

            stopped = [r["summary"].get("stopped_epoch") for r in trecs
                       if isinstance(r["summary"].get("stopped_epoch"), int)]
            best_val = [min(r["summary"].get("val_loss", []) or [float("nan")]) for r in trecs]
            best_val = [v for v in best_val if np.isfinite(v)]

            model_agg["targets"][target] = {
                "n_seeds": len(seeds),
                "seeds": seeds,
                "stopped_epoch": {
                    "mean": float(np.mean(stopped)) if stopped else None,
                    "min": min(stopped) if stopped else None,
                    "max": max(stopped) if stopped else None,
                },
                "best_val_loss": {
                    "mean": float(np.mean(best_val)) if best_val else None,
                    "std":  float(np.std(best_val, ddof=1)) if len(best_val) > 1 else None,
                },
                "test_metrics": metric_agg,
            }

        out[model] = model_agg
    return out


def load_baselines(baselines_dir: Path) -> Dict[str, dict]:
    b: Dict[str, dict] = {}
    for kind in ("persistence", "climatology"):
        p = baselines_dir / f"regression-{kind}.json"
        if p.exists():
            with open(p) as f:
                b[kind] = json.load(f)
    return b


def per_station_summary(records: List[dict]) -> Dict[str, dict]:
    """Aggregate per_station.npz arrays: {model: {target: {metric: (N,) mean-across-seeds}}}"""
    out: Dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    grouped: Dict[tuple, list] = defaultdict(list)
    for r in records:
        if r["per_station"] is None:
            continue
        grouped[(r["model"], r["target"])].append(r["per_station"])
    for (model, target), paths in grouped.items():
        # Each path holds arrays like mae, rmse, mae_extreme, mae_nonextreme, peak_intensity_error
        # shape (N, 1) for univariate
        acc: Dict[str, list] = defaultdict(list)
        for pp in paths:
            data = np.load(pp)
            for k in data.files:
                acc[k].append(data[k].squeeze(-1) if data[k].ndim == 2 else data[k])
        metric_out = {}
        for k, arrs in acc.items():
            stacked = np.stack(arrs)  # (n_seeds, N)
            metric_out[k] = {
                "mean_across_seeds": stacked.mean(axis=0).tolist(),
                "std_across_seeds": stacked.std(axis=0, ddof=1).tolist() if stacked.shape[0] > 1 else None,
                "n_seeds": int(stacked.shape[0]),
            }
        out[model][target] = metric_out
    return {m: dict(t) for m, t in out.items()}


def try_plot_mae_bar(agg: Dict[str, dict], baselines: Dict[str, dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    targets = list(TARGET_ORDER)
    models = [m for m in MODEL_ORDER if m in agg]
    if not models:
        return False

    x = np.arange(len(targets))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))

    # bars per model (best available metric — prefer mae over mae_pooled_norm)
    for i, m in enumerate(models):
        vals = []
        errs = []
        for tgt in targets:
            t_agg = agg[m]["targets"].get(tgt, {}).get("test_metrics", {})
            metric = t_agg.get("mae") or t_agg.get("mae_pooled_norm") or {}
            vals.append(metric.get("mean", np.nan))
            errs.append(metric.get("std", 0) or 0)
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, label=m.upper(), capsize=3)

    # baseline lines (physical units — mae from baselines JSON)
    for kind, style in [("persistence", "--"), ("climatology", ":")]:
        if kind not in baselines: continue
        bmae = baselines[kind].get("mae", {})
        vals = [bmae.get(tgt, np.nan) for tgt in targets]
        ax.plot(x, vals, style, color="grey", marker="_", markersize=15,
                label=f"baseline: {kind}")

    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("MAE (physical units)")
    ax.set_title("SAWS 25-node regression — MAE per (model, target) with baselines")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close()
    return True


def try_plot_val_convergence(records: List[dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    models = sorted({r["model"] for r in records})
    if not models:
        return False
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=False, squeeze=False)
    for j, m in enumerate(models):
        ax = axes[0, j]
        for r in records:
            if r["model"] != m: continue
            vl = r["summary"].get("val_loss", [])
            if not vl: continue
            ax.plot(range(1, len(vl) + 1), vl, alpha=0.6,
                    label=f"{r['target']}_s{r['seed']}")
        ax.set_title(m.upper())
        ax.set_xlabel("epoch")
        ax.set_ylabel("val loss")
        ax.grid(True, alpha=0.3)
        if any(r["model"] == m for r in records):
            ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("Validation-loss convergence per completed task")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close()
    return True


def write_summary_md(agg: Dict[str, dict], baselines: Dict[str, dict],
                     n_total: int, n_done: int, out_path: Path):
    lines = [
        "# SAWS 25-node regression — running results",
        "",
        f"**Auto-generated by `scripts/analyze_saws25.py`.**  ",
        f"**Progress:** {n_done}/{n_total} tasks complete ({100 * n_done / n_total:.1f}%)",
        "",
        "## Per-model × per-target MAE (physical units, mean ± std across completed seeds)",
        "",
    ]
    header = "| Model | " + " | ".join(TARGET_ORDER) + " |"
    sep = "|" + "|".join(["---"] * (len(TARGET_ORDER) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for m in MODEL_ORDER:
        if m not in agg:
            lines.append(f"| {m.upper()} | " + " | ".join(["—"] * len(TARGET_ORDER)) + " |")
            continue
        row = [m.upper()]
        for tgt in TARGET_ORDER:
            t_agg = agg[m]["targets"].get(tgt, {}).get("test_metrics", {})
            metric = t_agg.get("mae") or t_agg.get("mae_pooled_norm")
            if metric is None:
                row.append("—")
            else:
                n = metric["n"]
                row.append(f"{metric['mean']:.3f} ± {metric['std']:.3f} (n={n})")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Baselines (physical units, MAE)", "",
              "| Kind | " + " | ".join(TARGET_ORDER) + " |",
              sep]
    for kind in ("persistence", "climatology"):
        if kind not in baselines:
            continue
        bmae = baselines[kind].get("mae", {})
        row = [kind]
        for tgt in TARGET_ORDER:
            v = bmae.get(tgt)
            row.append(f"{v:.3f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Notes", "",
              "- Progress values include only tasks whose `summary.json` has landed.",
              "- Per-metric std uses `ddof=1` (Bessel-corrected sample std).",
              "- Baselines are seed-independent and were computed once via",
              "  `scripts/compute_baselines.py --dataset saws --skip-classification`.",
              ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    ap.add_argument("--total-tasks", type=int, default=120)
    args = ap.parse_args()

    records = discover_completed(args.runs)
    n_done = len(records)
    print(f"discovered {n_done} completed tasks in {args.runs}")

    if n_done == 0:
        print("no completions yet — nothing to aggregate")
        return 0

    agg = aggregate_per_model(records)
    baselines = load_baselines(args.out / "baselines")

    # Per-model aggregated JSONs
    p1 = args.out / "phase1_regression"
    p1.mkdir(parents=True, exist_ok=True)
    for model, adict in agg.items():
        with open(p1 / f"{model}_regression_summary.json", "w") as f:
            json.dump(adict, f, indent=2, default=str)
        print(f"  wrote {p1 / f'{model}_regression_summary.json'}  ({adict['n_records']} tasks)")

    # Per-station aggregated JSON (all models)
    station_agg = per_station_summary(records)
    if station_agg:
        with open(p1 / "per_station_across_seeds.json", "w") as f:
            json.dump(station_agg, f, indent=2, default=str)
        print(f"  wrote {p1 / 'per_station_across_seeds.json'}  ({len(station_agg)} model entries)")

    # Plots
    plots = args.out / "plots"
    ok = try_plot_mae_bar(agg, baselines, plots / "mae_bar.png")
    print(f"  {'wrote' if ok else 'skipped'} {plots / 'mae_bar.png'}")
    ok = try_plot_val_convergence(records, plots / "val_convergence.png")
    print(f"  {'wrote' if ok else 'skipped'} {plots / 'val_convergence.png'}")

    # Markdown summary
    write_summary_md(agg, baselines, args.total_tasks, n_done, plots / "SUMMARY.md")
    print(f"  wrote {plots / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
