"""
Generates summary plots at the end of a cli.py experiment run.

Called once after all seeds have completed. Reads per-seed histories and the
aggregated summary.json to produce:
  - Loss curves (train + val per epoch) for each seed, overlaid on one figure
  - NRMSE-vs-horizon curve averaged over seeds (regression only)
  - Bar chart of primary metrics (MAE/RMSE or F1/AUC) across seeds (boxplot)

Plots are saved to <output_dir>/plots/ as PNG files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)


def _matplotlib():
    """Import matplotlib in non-interactive mode."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    return plt


def plot_loss_curves(
    histories: List[Dict[str, Any]],
    plots_dir: Path,
) -> None:
    """
    Train and validation loss curves for all seeds on a single figure.

    Each seed is plotted as a semi-transparent line; a bold line shows the
    mean across seeds. A vertical dashed line marks the earliest early-stop.
    """
    plt = _matplotlib()

    fig, ax = plt.subplots(figsize=(9, 5))

    train_losses = [h["train_loss"] for h in histories if h.get("train_loss")]
    val_losses   = [h["val_loss"]   for h in histories if h.get("val_loss")]
    if not train_losses:
        plt.close(fig)
        return

    max_len = max(len(tl) for tl in train_losses)

    import numpy as np  # noqa: PLC0415

    def _pad(series_list):
        out = []
        for s in series_list:
            arr = np.array(s, dtype=float)
            pad = np.full(max_len - len(arr), np.nan)
            out.append(np.concatenate([arr, pad]))
        return np.array(out)

    train_arr = _pad(train_losses)
    val_arr   = _pad(val_losses)

    epochs = np.arange(1, max_len + 1)

    for i in range(len(train_arr)):
        ax.plot(epochs, train_arr[i], color="steelblue",  alpha=0.25, linewidth=0.8)
        ax.plot(epochs, val_arr[i],   color="darkorange", alpha=0.25, linewidth=0.8)

    ax.plot(epochs, np.nanmean(train_arr, axis=0), color="steelblue",
            linewidth=2.2, label="Train (mean)")
    ax.plot(epochs, np.nanmean(val_arr, axis=0),   color="darkorange",
            linewidth=2.2, label="Val (mean)")

    # Earliest early stop
    stop_epochs = [h.get("stopped_epoch") for h in histories if h.get("stopped_epoch") is not None]
    if stop_epochs:
        min_stop = min(stop_epochs)
        if min_stop < max_len:
            ax.axvline(min_stop, color="red", linestyle="--", alpha=0.5, label=f"Early stop (min={min_stop})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    log.info("Saved loss_curves.png")


def plot_nrmse_horizon(
    histories: List[Dict[str, Any]],
    plots_dir: Path,
) -> None:
    """
    NRMSE-vs-horizon curve averaged over seeds (regression only).

    Expects `history["test_metrics"]["nrmse_per_horizon"]` to be a list of
    per-horizon NRMSE values (length = H).
    """
    import numpy as np  # noqa: PLC0415

    curves = []
    for h in histories:
        nrmse = h.get("test_metrics", {}).get("nrmse_per_horizon")
        if nrmse:
            curves.append(np.array(nrmse, dtype=float))

    if not curves:
        return

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4))

    arr = np.array(curves)
    horizons = np.arange(1, arr.shape[1] + 1)
    mean_curve = arr.mean(axis=0)
    std_curve  = arr.std(axis=0)

    ax.plot(horizons, mean_curve, color="steelblue", linewidth=2, label="Mean NRMSE")
    ax.fill_between(horizons, mean_curve - std_curve, mean_curve + std_curve,
                    color="steelblue", alpha=0.2, label="±1 std")

    # VPT epsilon line for reference (epsilon = 0.5 by default)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="VPT threshold (ε=0.5)")

    ax.set_xlabel("Forecast horizon step")
    ax.set_ylabel("NRMSE")
    ax.set_title("NRMSE vs. Forecast Horizon")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "nrmse_horizon.png", dpi=150)
    plt.close(fig)
    log.info("Saved nrmse_horizon.png")


def plot_metric_boxplot(
    histories: List[Dict[str, Any]],
    plots_dir: Path,
    task: str,
) -> None:
    """
    Boxplot of per-seed test metrics (one box per metric).
    """
    import numpy as np  # noqa: PLC0415

    if task == "regression":
        metrics = ["mae", "rmse", "mape"]
    else:
        metrics = ["f1", "recall", "precision", "auc_roc"]

    data: Dict[str, List[float]] = {m: [] for m in metrics}
    for h in histories:
        tm = h.get("test_metrics", {})
        for m in metrics:
            val = tm.get(m) or tm.get(f"{m}_mean")
            if val is not None:
                data[m].append(float(val))

    data = {k: v for k, v in data.items() if v}
    if not data:
        return

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(max(5, len(data) * 1.5), 4))

    positions = list(range(1, len(data) + 1))
    bplot = ax.boxplot(list(data.values()), positions=positions,
                       patch_artist=True, notch=False)

    for patch in bplot["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([k.upper() for k in data.keys()])
    ax.set_ylabel("Value")
    ax.set_title(f"Per-seed Test Metrics ({task})")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(plots_dir / "metric_boxplot.png", dpi=150)
    plt.close(fig)
    log.info("Saved metric_boxplot.png")


def generate_plots(
    output_dir: Path,
    all_histories: List[Dict[str, Any]],
    task: str,
) -> None:
    """
    Generate all summary plots for a completed experiment run.

    Args:
        output_dir:    top-level output directory for the run (contains run_XX/).
        all_histories: list of history dicts returned by Trainer.train(), one per seed.
        task:          'regression' or 'classification'.
    """
    try:
        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_loss_curves(all_histories, plots_dir)

        if task == "regression":
            plot_nrmse_horizon(all_histories, plots_dir)

        plot_metric_boxplot(all_histories, plots_dir, task)

        log.info("Plots saved to %s", plots_dir)
    except Exception as exc:
        log.warning("Plot generation failed (non-fatal): %s", exc)
