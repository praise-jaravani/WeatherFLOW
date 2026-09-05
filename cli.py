"""
Single entry point for all WEATHER-FLOW experiments.

Usage:
  python cli.py --model <name> --dataset <saws|era5|both> --task <regression|classification> [options]

Available models: agcrn, clcrn, mtgode, climode, stlgru, litestgnn
(Diffusion models — DiffSTG, USTD — are maintained by Génevieve in her own
forks; register their adapters here when she integrates them.)

Optional flags:
  --runs N             Number of seeds to run (default 10, seeds 42–51)
  --horizon H          Forecast horizon in timesteps (overrides YAML)
  --no-cache           Force preprocessing to rerun even if cache is valid
  --mlflow-off         Disable MLflow tracking (useful for quick debug runs)
  --config.KEY VALUE   Override any YAML config value at runtime, e.g.:
                         --config.training.learning_rate 0.0005

Display: rich progress bars showing nested outer (run 1/10) and inner
(epoch 1/N) loops simultaneously, with live loss and ETA.
"""

import argparse
import importlib
import json
import sys
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console

ADAPTER_REGISTRY = {
    "agcrn":     "models.baselines.agcrn_adapter.AGCRNAdapter",
    "clcrn":     "models.baselines.clcrn_adapter.CLCRNAdapter",
    "mtgode":    "models.continuous_time.mtgode_adapter.MTGODEAdapter",
    "stgode":    "models.continuous_time.stgode_adapter.STGODEAdapter",
    "climode":   "models.continuous_time.climode_adapter.ClimODEAdapter",
    "stlgru":    "models.lightweight.stlgru_adapter.STLGRUAdapter",
    "litestgnn": "models.lightweight.litestgnn_adapter.LiteSTGNNAdapter",
    # Faithful reimplementation of Verma et al. ClimODE (ICLR 2024) —
    # DISTINCT from "climode" above (that key points to the archived
    # ClimODE-inspired advection graph-ODE; see results/archive/climode_inspired/).
    "climode_faithful": "models.continuous_time.climode_faithful_adapter.ClimODEFaithfulAdapter",
    # Diffusion adapters live in Génevieve's fork; add here when integrated:
    #   "diffstg": "models.diffusion.diffstg_adapter.DiffSTGAdapter",
    #   "ustd":    "models.diffusion.ustd_adapter.USTDAdapter",
}

console = Console()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WEATHER-FLOW: unified STGNN benchmarking framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, choices=list(ADAPTER_REGISTRY.keys()),
        help="Model name",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["saws", "era5", "both"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--task", required=True, choices=["regression", "classification"],
        help="Prediction task",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of seeds to run (seeds 42 through 42+runs-1)",
    )
    parser.add_argument(
        "--seed-start", type=int, default=42,
        help="First seed value; runs seeds seed-start through seed-start+runs-1",
    )
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="Override forecast horizon in timesteps",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force preprocessing to rerun even if cache is valid",
    )
    parser.add_argument(
        "--mlflow-off", action="store_true",
        help="Disable MLflow tracking",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=None, metavar="N",
        help="Subsample the graph to the first N nodes (for fast local smoke tests)",
    )

    # Parse known args; remaining --config.KEY VALUE pairs are handled below
    args, unknown = parser.parse_known_args()
    args.config_overrides = _parse_config_overrides(unknown)
    return args


def _parse_config_overrides(tokens: list) -> Dict[str, str]:
    """Extract --config.key value pairs from leftover argparse tokens."""
    overrides: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--config."):
            key = token[len("--config."):]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                overrides[key] = tokens[i + 1]
                i += 2
            else:
                console.print(f"[yellow]Warning:[/yellow] --config.{key} has no value, skipped")
                i += 1
        else:
            console.print(f"[yellow]Warning:[/yellow] Unknown argument '{token}', ignored")
            i += 1
    return overrides


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(model_name: str, overrides: Dict[str, str]) -> dict:
    """Load experiments/configs/<model>.yaml and apply CLI overrides."""
    import yaml  # noqa: PLC0415
    config_path = Path("experiments/configs") / f"{model_name}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, value in overrides.items():
        _set_nested(config, key.split("."), _coerce(value))
    return config


def _set_nested(d: dict, keys: list, value: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _coerce(value: str) -> Any:
    """Try to convert a string override to int, float, or bool."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

def load_adapter(model_name: str):
    """Dynamically import and return the adapter class for the given model."""
    if model_name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(ADAPTER_REGISTRY)}")
    module_path, class_name = ADAPTER_REGISTRY[model_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def make_output_dir(model: str, dataset: str, task: str) -> Path:
    """Create and return a timestamped output directory under outputs/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"{model}_{dataset}_{task}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Loss construction
# ---------------------------------------------------------------------------

def _build_loss(task: str, train_loader, device: torch.device) -> torch.nn.Module:
    from training.losses import mae_loss, compute_class_weights, wbce_loss  # noqa: PLC0415
    if task == "regression":
        # Uniform MAE across MTGODE / AGCRN / CLCRN per the 2026-08-08 audit
        # + loss decision. Provenance: MTGODE §4.4, AGCRN L1Loss default,
        # CLCRN masked_mae_loss default. See training/losses.py + HYPERPARAMETERS.md.
        return mae_loss()

    # Classification: derive per-variable class weights from training labels.
    # Per Phase-2 plan §3.2 pos_weight is a vector of length d_label.
    #
    # Read positive rates from the underlying (T, N, d) int8 label tensor on
    # the Dataset — the sliding-window loader would materialise a
    # (samples, H, N, d) float32 array (~40 GB at N=999). numpy's .mean()
    # promotes int8 to float64 internally without a full-array cast.
    ds = train_loader.dataset
    labels_full = ds.labels.numpy() if ds.labels is not None else None   # (T, N, d) int8
    if labels_full is None:
        raise RuntimeError("classification loader has no labels attached")
    if ds.label_indices is not None:
        labels_lab = labels_full[..., ds.label_indices.numpy()]
    else:
        labels_lab = labels_full
    if getattr(ds, "label_mode", "per_variable") == "any":
        labels_lab = labels_lab.max(axis=-1)                              # (T, N)
    w_0, w_1 = compute_class_weights(labels_lab)
    if np.ndim(w_1) == 0:
        w_1_arr = np.atleast_1d(np.asarray(w_1, dtype=float))
        pos_rate_pct = 100.0 / (1.0 + w_1_arr / float(w_0))
        console.print(
            f"  Class weights (scalar): w_0={float(w_0):.3f}  "
            f"w_1={float(w_1):.3f}  (pos rate={pos_rate_pct[0]:.2f}%)"
        )
    else:
        w_0_arr = np.asarray(w_0, dtype=float)
        w_1_arr = np.asarray(w_1, dtype=float)
        pos_rate_pct = 100.0 / (1.0 + w_1_arr / w_0_arr)
        console.print(f"  Per-variable class weights (d_label={len(w_1_arr)}):")
        var_names = None
        try:
            var_names = list(train_loader.dataset.label_indices.tolist())
        except AttributeError:
            var_names = None
        for i, (w1_i, pr_i) in enumerate(zip(w_1_arr, pos_rate_pct)):
            tag = f" [var-idx {var_names[i]}]" if var_names is not None else ""
            console.print(f"    ch {i}{tag}: w_1={w1_i:.3f}  pos_rate={pr_i:.2f}%")
    return wbce_loss(w_0, w_1, device)


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _aggregate_metrics(all_metrics: list) -> dict:
    """
    Aggregate per-seed regression metric dicts (compute_all output, 'per_station'
    already removed). Produces a per-variable summary, preserves the NRMSE curve,
    and stores raw per-seed arrays under 'per_seed' for Wilcoxon.

    Falls back to the old scalar aggregator for classification (no 'mae' dict).
    """
    if not all_metrics:
        return {}

    seed0 = all_metrics[0]

    # Classification fallback: all values are plain scalars
    if "mae" not in seed0 or not isinstance(seed0.get("mae"), dict):
        result = {}
        for k in seed0.keys():
            vals = [m[k] for m in all_metrics if k in m and isinstance(m[k], (int, float))]
            if vals:
                arr = np.array(vals, dtype=float)
                result[f"{k}_mean"] = float(arr.mean())
                result[f"{k}_std"] = float(arr.std())
        return result

    # Regression path: per-variable dicts
    PER_VAR = ["mae", "rmse", "mse", "smape", "mae_extreme",
               "mae_nonextreme", "peak_intensity_error", "vpt"]
    summary = {}
    for k in ("units", "vpt_epsilon", "report_vars"):
        if k in seed0:
            summary[k] = seed0[k]

    var_names = list(seed0.get("mae", {}).keys())

    per_variable = {v: {} for v in var_names}
    per_seed = {}
    for key in PER_VAR:
        if key not in seed0:
            continue
        per_seed[key] = {}
        for v in var_names:
            vals = np.array([m[key][v] for m in all_metrics], dtype=float)
            per_variable[v][f"{key}_mean"] = float(np.nanmean(vals))
            per_variable[v][f"{key}_std"] = float(np.nanstd(vals))
            per_seed[key][v] = vals.tolist()
    summary["per_variable"] = per_variable

    if "nrmse_per_horizon" in seed0:
        nrmse = {}
        for v in var_names:
            curves = np.array([m["nrmse_per_horizon"][v] for m in all_metrics], dtype=float)
            nrmse[v] = {"mean": np.nanmean(curves, axis=0).tolist(),
                        "std": np.nanstd(curves, axis=0).tolist()}
        summary["nrmse_per_horizon"] = nrmse

    skip = set(PER_VAR) | {"nrmse_per_horizon", "per_station",
                           "units", "vpt_epsilon", "report_vars"}
    for key, val in seed0.items():
        if key in skip or not isinstance(val, (int, float)):
            continue
        vals = np.array([m[key] for m in all_metrics], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(vals))
        summary[f"{key}_std"] = float(np.nanstd(vals))

    summary["per_seed"] = per_seed
    return summary


# ---------------------------------------------------------------------------
# Node subsampling for smoke tests
# ---------------------------------------------------------------------------

def _subsample_nodes(train_loader, val_loader, test_loader, adj, meta, max_nodes, config):
    """
    Rebuild dataloaders using only the first `max_nodes` nodes from cached arrays.

    Reads directly from the preprocessing cache so the pipeline doesn't need
    to be re-run. Does not modify the cache on disk.
    """
    from preprocessing.pipeline import load_cache   # noqa: PLC0415
    from preprocessing.dataset import build_dataloaders  # noqa: PLC0415

    N = min(max_nodes, int(meta["n_nodes"]))
    dataset_name = meta["dataset"]

    cached = load_cache(dataset_name)

    clip_sigma = float(config.get("preprocessing", {}).get("clip_sigma", 10.0))

    # Slice to first N nodes on the spatial axis (axis=1) and clip extremes
    X_tr = np.clip(cached["X_train"][:, :N, :], -clip_sigma, clip_sigma)
    X_va = np.clip(cached["X_val"]  [:, :N, :], -clip_sigma, clip_sigma)
    X_te = np.clip(cached["X_test"] [:, :N, :], -clip_sigma, clip_sigma)
    y_tr = cached["y_train"][:, :N, :]
    y_va = cached["y_val"]  [:, :N, :]
    y_te = cached["y_test"] [:, :N, :]

    adj_sub = cached["adj"][:N, :N].copy()

    # Re-normalise adjacency rows that lost neighbours due to slicing
    row_sums = adj_sub.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_sub /= row_sums

    new_meta = dict(meta)
    new_meta["n_nodes"] = N
    new_meta["coords"]  = cached["coords"][:N]
    new_meta["mean"]    = cached["mean"][:, :N, :]
    new_meta["std"]     = cached["std"] [:, :N, :]
    new_meta["thresholds"] = cached["thresholds"][:N, :]
    new_meta["static_thresholds"] = cached["thresholds"][:N, :]

    input_len  = config["data"]["input_len"]
    horizon    = config["data"]["horizon"]
    task       = config.get("task", "regression")
    batch_size = config["training"]["batch_size"]

    data_cfg = config.get("data", {})
    new_train, new_val, new_test = build_dataloaders(
        X_tr, X_va, X_te,
        y_tr, y_va, y_te,
        input_len=input_len, horizon=horizon, task=task,
        batch_size=batch_size, num_workers=0,
        label_mode=data_cfg.get("label_mode", "per_variable"),
        var_names=new_meta.get("var_names"),
        label_variables=data_cfg.get("label_variables", None),
    )

    return new_train, new_val, new_test, adj_sub, new_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.model, args.config_overrides)
    config["task"] = args.task
    config["dataset"] = args.dataset

    if args.horizon is not None:
        config["data"]["horizon"] = args.horizon

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console.rule("[bold]WEATHER-FLOW[/bold]")
    console.print(
        f"  model=[bold]{args.model}[/bold]  "
        f"dataset={args.dataset}  task={args.task}  "
        f"device={device}"
    )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    console.print("\n[bold]Preprocessing...[/bold]")
    from preprocessing.pipeline import run_pipeline, verify_cache_coverage  # noqa: PLC0415

    if not args.no_cache:
        coverage = verify_cache_coverage(config.get("dataset", "era5"))
        if coverage["status"] == "stale":
            n_missing = len(coverage["missing_from_cache"])
            first, last = coverage["missing_from_cache"][0], coverage["missing_from_cache"][-1]
            console.print(
                f"  [yellow]Warning: cache is stale — {n_missing} month(s) on disk not in cache "
                f"({first[0]}-{first[1]:02d} … {last[0]}-{last[1]:02d}). "
                f"Run with --no-cache to rebuild (recommended on HPC).[/yellow]"
            )
        elif coverage["status"] == "ok" and coverage["n_cached"] > 0:
            meta_info = {}
            try:
                from preprocessing.pipeline import load_cache  # noqa: PLC0415
                meta_info = load_cache(config.get("dataset", "era5"))["meta"]
            except Exception:
                pass
            dr = meta_info.get("date_range", {})
            console.print(
                f"  Cache covers {coverage['n_cached']} months "
                + (f"({dr.get('start', '?')} → {dr.get('end', '?')})" if dr else "")
            )

    train_loader, val_loader, test_loader, adj, meta = run_pipeline(
        config=config,
        force_rerun=args.no_cache,
    )

    # ------------------------------------------------------------------
    # Node subsampling (smoke test mode)
    # ------------------------------------------------------------------
    if args.max_nodes is not None:
        train_loader, val_loader, test_loader, adj, meta = _subsample_nodes(
            train_loader, val_loader, test_loader, adj, meta,
            args.max_nodes, config,
        )
        console.print(
            f"  [yellow]--max-nodes {args.max_nodes}:[/yellow] "
            f"subsampled to {meta['n_nodes']} nodes (smoke test mode)"
        )

    config["meta"] = meta
    adj_device = adj.to(device) if isinstance(adj, torch.Tensor) else torch.from_numpy(adj).float().to(device)
    adj_cpu = adj_device.cpu()

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------
    loss_fn = _build_loss(args.task, train_loader, device)

    # ------------------------------------------------------------------
    # Static computational metrics (before run loop)
    # ------------------------------------------------------------------
    console.print("\n[bold]Computing static metrics...[/bold]")
    adapter_cls = load_adapter(args.model)
    _probe_adapter = adapter_cls()
    _probe_model = _probe_adapter.build_model(config, adj_cpu)
    _probe_model.to(device)

    from evaluation.computational import record_static_metrics  # noqa: PLC0415
    sample_x, _ = next(iter(train_loader))
    sample_x = sample_x.to(device)
    static_metrics = record_static_metrics(_probe_model, sample_x)
    del _probe_adapter, _probe_model, sample_x
    torch.cuda.empty_cache() if device.type == "cuda" else None

    flops_str = f"{static_metrics['flops_G']:.3f}G" if static_metrics["flops_G"] is not None else "n/a"
    console.print(
        f"  params={static_metrics['params_M']:.3f}M  "
        f"flops={flops_str}"
    )

    # ------------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------------
    mlflow_enabled = not args.mlflow_off
    if mlflow_enabled:
        import mlflow  # noqa: PLC0415
        mlflow.set_tracking_uri("sqlite:///tracking/mlflow.db")
        experiment_name = f"{args.model}/{args.dataset}/{args.task}"
        mlflow.set_experiment(experiment_name)
        console.print(f"\n  MLflow experiment: [italic]{experiment_name}[/italic]")
    else:
        console.print("\n  MLflow: [yellow]disabled[/yellow]")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = make_output_dir(args.model, args.dataset, args.task)
    console.print(f"  Output dir: {output_dir}")

    # ------------------------------------------------------------------
    # Run loop (10 seeds)
    # ------------------------------------------------------------------
    from training.seed import seed_everything  # noqa: PLC0415
    from training.trainer import Trainer       # noqa: PLC0415
    from evaluation.regression import ERA5_VAR_NAMES  # noqa: PLC0415

    vpt_epsilon = float(config.get("evaluation", {}).get("vpt_epsilon", 0.3))

    seeds = list(range(args.seed_start, args.seed_start + args.runs))
    all_metrics = []
    all_histories = []

    console.rule(f"\n[bold]Running {args.runs} seed(s)[/bold]")

    for run_idx, seed in enumerate(seeds, 1):
        console.print(f"\n[bold cyan]Run {run_idx}/{args.runs}[/bold cyan]  seed={seed}")
        run_dir = output_dir / f"run_{seed:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(seed)

        adapter = adapter_cls()
        adapter.build_model(config, adj_cpu)
        adapter.model.to(device)

        def _run_one(mlflow_on: bool):
            return Trainer(
                adapter=adapter,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                loss_fn=loss_fn,
                adj=adj_device,
                config=config,
                output_dir=run_dir,
                seed=seed,
                mlflow_enabled=mlflow_on,
                norm_mean=meta["mean"],
                norm_std=meta["std"],
                thresholds=meta["thresholds"],
                static_thresholds=meta.get("static_thresholds"),
                var_names=meta.get("var_names", ERA5_VAR_NAMES),
                vpt_epsilon=vpt_epsilon,
            ).train()

        if mlflow_enabled:
            import mlflow  # noqa: PLC0415
            safe_static = {k: (v if v is not None else "n/a") for k, v in static_metrics.items()}
            with mlflow.start_run(run_name=f"seed_{seed}"):
                mlflow.log_params({
                    "seed": seed,
                    "model": args.model,
                    "dataset": args.dataset,
                    "task": args.task,
                    **safe_static,
                })
                result = _run_one(mlflow_on=True)
        else:
            result = _run_one(mlflow_on=False)

        # Save per-seed training history (loss curves, stopped epoch)
        history_payload = {
            "seed": seed,
            "train_loss": result.get("train_loss", []),
            "val_loss": result.get("val_loss", []),
            "grad_norm": result.get("grad_norm", []),
            "stopped_epoch": result.get("stopped_epoch"),
            "test_metrics": result.get("test_metrics", {}),
        }
        with open(run_dir / "history.json", "w") as _hf:
            json.dump(history_payload, _hf, indent=2)
        all_histories.append(history_payload)

        test_m = result.get("test_metrics", {})
        all_metrics.append(test_m)

        # Print quick per-seed scalar summary (pooled norm MAE + inference time)
        scalars = {k: v for k, v in test_m.items() if isinstance(v, (int, float))}
        _summary_line = "  " + "  ".join(
            f"{k}={v:.4f}" for k, v in scalars.items()
            if not k.endswith("_std")
        )
        console.print(_summary_line)

    # ------------------------------------------------------------------
    # Aggregate across seeds and save summary
    # ------------------------------------------------------------------
    summary = _aggregate_metrics(all_metrics)
    summary.update({
        "model":   args.model,
        "dataset": args.dataset,
        "task":    args.task,
        "seeds":   seeds,
        **{k: (v if v is not None else None) for k, v in static_metrics.items()},
    })

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------
    from training.plotting import generate_plots  # noqa: PLC0415
    console.print("\n[bold]Generating plots...[/bold]")
    generate_plots(output_dir, all_histories, args.task)

    console.rule("[bold green]Done[/bold green]")
    console.print(f"Summary  → {summary_path}")
    console.print(f"Plots    → {output_dir / 'plots'}")

    # Print key aggregate metrics
    if args.task == "regression":
        pv = summary.get("per_variable", {})
        report = summary.get("report_vars", list(pv.keys())[:1])
        for var in report:
            if var in pv and "mae_mean" in pv[var]:
                mae_mean = pv[var]["mae_mean"]
                mae_std = pv[var].get("mae_std", 0.0)
                console.print(f"  mae[{var}] = {mae_mean:.4f} ± {mae_std:.4f} (physical units)")
    else:
        if "f1_mean" in summary:
            console.print(f"  f1 = {summary['f1_mean']:.4f} ± {summary.get('f1_std', 0):.4f}")


if __name__ == "__main__":
    main()
