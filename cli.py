"""
Single entry point for all WEATHER-FLOW experiments.

Usage:
  python cli.py --model <name> --dataset <saws|era5|both> --task <regression|classification> [options]

Available models: agcrn, clcrn, mtgode, climode, stlgru, litestgnn, diffstg, ustd

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
    "climode":   "models.continuous_time.climode_adapter.ClimODEAdapter",
    "stlgru":    "models.lightweight.stlgru_adapter.STLGRUAdapter",
    "litestgnn": "models.lightweight.litestgnn_adapter.LiteSTGNNAdapter",
    "diffstg":   "models.diffusion.diffstg_adapter.DiffSTGAdapter",
    "ustd":      "models.diffusion.ustd_adapter.USTDAdapter",
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
    from training.losses import mse_loss, compute_class_weights, wbce_loss  # noqa: PLC0415
    if task == "regression":
        return mse_loss()
    # Classification: derive class weights from training labels
    all_labels = []
    for _, y in train_loader:
        all_labels.append(y.numpy())
    train_labels = np.concatenate(all_labels, axis=0).ravel()
    w_0, w_1 = compute_class_weights(train_labels)
    console.print(
        f"  Class weights: w_0={w_0:.3f}  w_1={w_1:.3f}  "
        f"(pos rate={1/(1+w_1/w_0)*100:.1f}%)"
    )
    return wbce_loss(w_0, w_1, device)


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _aggregate_metrics(all_metrics: list) -> dict:
    """Compute mean ± std across seeds for each metric key."""
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        arr = np.array(vals, dtype=float)
        result[f"{k}_mean"] = float(arr.mean())
        result[f"{k}_std"] = float(arr.std())
    return result


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

    input_len  = config["data"]["input_len"]
    horizon    = config["data"]["horizon"]
    task       = config.get("task", "regression")
    batch_size = config["training"]["batch_size"]

    new_train, new_val, new_test = build_dataloaders(
        X_tr, X_va, X_te,
        y_tr, y_va, y_te,
        input_len=input_len, horizon=horizon, task=task,
        batch_size=batch_size, num_workers=0,
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
    from preprocessing.pipeline import run_pipeline  # noqa: PLC0415
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

    console.print(
        f"  params={static_metrics['params_M']:.3f}M  "
        f"flops={static_metrics['flops_G']:.3f}G"
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

    # Extreme mask for stratified regression metrics (optional)
    extreme_mask: Optional[np.ndarray] = meta.get("test_extreme_mask")

    # ------------------------------------------------------------------
    # Run loop (10 seeds)
    # ------------------------------------------------------------------
    from training.seed import seed_everything  # noqa: PLC0415
    from training.trainer import Trainer       # noqa: PLC0415

    seeds = list(range(42, 42 + args.runs))
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
                extreme_mask=extreme_mask,
            ).train()

        if mlflow_enabled:
            import mlflow  # noqa: PLC0415
            with mlflow.start_run(run_name=f"seed_{seed}"):
                mlflow.log_params({
                    "seed": seed,
                    "model": args.model,
                    "dataset": args.dataset,
                    "task": args.task,
                    **static_metrics,
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

        _summary_line = "  " + "  ".join(
            f"{k}={v:.4f}" for k, v in test_m.items()
            if isinstance(v, float) and not k.endswith("_std")
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
        **static_metrics,
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
    primary = "mae_mean" if args.task == "regression" else "f1_mean"
    if primary in summary:
        console.print(f"  {primary.replace('_mean', '')} = {summary[primary]:.4f} ± {summary.get(primary.replace('mean','std'), 0):.4f}")


if __name__ == "__main__":
    main()
