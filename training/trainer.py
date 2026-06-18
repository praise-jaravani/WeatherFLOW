"""
Shared training loop. Every model in the pipeline runs through this Trainer.

The Trainer accepts a model adapter, DataLoaders, adjacency matrix, a config,
and an output directory. It runs the full training loop for a single seed.
The outer run loop in cli.py instantiates Trainer once per seed (10 times)
and aggregates results.

Responsibilities:
- Per-epoch train and validation passes
- Gradient clipping (max norm from config)
- Early stopping on validation loss (patience from config)
- Checkpoint saving: best model weights saved as checkpoint.pt
- Per-epoch metric logging to the terminal (rich) and MLflow
- Inference time measurement after best checkpoint is loaded
"""

import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from models.base_adapter import BaseAdapter
from training.scheduler import build_scheduler, clip_gradients
import evaluation.regression as regression
import evaluation.classification as classification
from evaluation.computational import record_inference_time

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        adapter: BaseAdapter,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: nn.Module,
        adj: torch.Tensor,
        config: Dict[str, Any],
        output_dir: Path,
        seed: int,
        mlflow_enabled: bool = True,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        thresholds: Optional[np.ndarray] = None,
        var_names: Optional[List[str]] = None,
        vpt_epsilon: float = 0.3,
    ):
        """
        Args:
            adapter:        fully-built model adapter (adapter.model is initialised).
            train_loader:   DataLoader for training split.
            val_loader:     DataLoader for validation split.
            test_loader:    DataLoader for test split.
            loss_fn:        loss module (MSELoss or BCEWithLogitsLoss).
            adj:            static adjacency matrix, shape (N, N), on any device.
            config:         merged experiment config dict.
            output_dir:     directory for this seed's checkpoint and logs.
            seed:           current random seed (used for display only).
            mlflow_enabled: if False, skip all mlflow.log_* calls.
            norm_mean:      (1, N, d) train z-score means; used to denorm test metrics.
            norm_std:       (1, N, d) train z-score stds; used to denorm test metrics.
            thresholds:     (N, d) extreme-event thresholds in PHYSICAL units.
            var_names:      list of d variable names (e.g. ERA5_VAR_NAMES).
            vpt_epsilon:    NRMSE threshold for VPT (fraction of a std).
        """
        self.adapter = adapter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.config = config
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.mlflow_enabled = mlflow_enabled

        self.norm_mean = norm_mean      # (1, N, d)
        self.norm_std = norm_std        # (1, N, d)
        self.thresholds = thresholds    # (N, d), PHYSICAL units
        self.var_names = var_names
        self.vpt_epsilon = vpt_epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adj = adj.to(self.device)

        adapter.model.to(self.device)
        self.loss_fn.to(self.device)

        train_cfg = config["training"]
        self.epochs = int(train_cfg["epochs"])
        self.patience = int(train_cfg["early_stopping_patience"])
        self.max_norm = float(train_cfg.get("gradient_clip_max_norm", 5.0))

        self.optimiser = torch.optim.Adam(
            adapter.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
        )
        self.scheduler = build_scheduler(self.optimiser, train_cfg)

        self.task = config.get("task", "regression")

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.checkpoint_path = self.output_dir / "checkpoint.pt"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._log_file = open(self.output_dir / "train.log", "w", buffering=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop for one seed.

        Returns:
            Dict with 'train_loss', 'val_loss' history lists, 'stopped_epoch',
            and 'test_metrics' dict from the best checkpoint.
        """
        history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "grad_norm": []}
        stopped_epoch = self.epochs

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=4,
        ) as progress:
            epoch_task = progress.add_task(
                f"[cyan]seed={self.seed}", total=self.epochs
            )

            for epoch in range(1, self.epochs + 1):
                train_loss, grad_norm = self._train_epoch()
                val_loss = self._validate_epoch()

                self.scheduler.step()
                self._save_checkpoint(val_loss)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["grad_norm"].append(grad_norm)

                desc = (
                    f"[cyan]seed={self.seed}[/cyan] "
                    f"ep={epoch}/{self.epochs} "
                    f"train={train_loss:.4f} val={val_loss:.4f}"
                )
                progress.update(epoch_task, advance=1, description=desc)

                log_line = (
                    f"epoch={epoch} train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} grad_norm={grad_norm:.4f}"
                )
                self._log_file.write(log_line + "\n")

                self._mlflow_log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss, "grad_norm": grad_norm},
                    step=epoch,
                )

                if self.patience_counter >= self.patience:
                    stopped_epoch = epoch
                    progress.update(
                        epoch_task,
                        description=f"[yellow]Early stop at epoch {epoch}[/yellow]",
                    )
                    break

        history["stopped_epoch"] = stopped_epoch
        history["test_metrics"] = self._evaluate_test()

        self._log_file.close()
        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self):
        """One training epoch. Returns (mean_train_loss, mean_grad_norm)."""
        self.adapter.model.train()
        total_loss = 0.0
        total_norm = 0.0
        n_batches = 0

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimiser.zero_grad()
            pred = self.adapter.forward(x, self.adj)

            # BCEWithLogitsLoss expects float targets; classification y is int8
            if self.task == "classification":
                y = y.float()

            loss = self.loss_fn(pred, y)
            loss.backward()
            norm = clip_gradients(self.adapter.model, self.max_norm)
            self.optimiser.step()

            total_loss += loss.item()
            total_norm += norm
            n_batches += 1

        return total_loss / max(n_batches, 1), total_norm / max(n_batches, 1)

    def _validate_epoch(self) -> float:
        """One validation epoch. Returns mean val loss."""
        total_loss = 0.0
        n_batches = 0

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.adapter.predict(x, self.adj)

            if self.task == "classification":
                y = y.float()

            loss = self.loss_fn(pred, y)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _build_extreme_mask(self, targets_norm: np.ndarray) -> Optional[np.ndarray]:
        """targets_norm: (samples, H, N, d). Returns (samples, H, N, d) bool, or None."""
        if self.thresholds is None or self.norm_mean is None or self.norm_std is None:
            return None
        m = self.norm_mean.reshape(1, 1, *self.norm_mean.shape[-2:])
        s = self.norm_std.reshape(1, 1, *self.norm_std.shape[-2:])
        targets_phys = targets_norm * s + m
        return targets_phys > self.thresholds[None, None, :, :]

    def _save_per_station(self, per_station: dict) -> None:
        """Save (N, d) per-station arrays for Phase 3 geographic stratification."""
        out_path = self.output_dir / "per_station.npz"
        np.savez(str(out_path), **{k: v for k, v in per_station.items() if isinstance(v, np.ndarray)})

    def _evaluate_test(self) -> Dict[str, Any]:
        """Load best checkpoint and evaluate on the test set."""
        if self.checkpoint_path.exists():
            state = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )
            self.adapter.model.load_state_dict(state)

        all_preds, all_targets = [], []

        for x, y in self.test_loader:
            x = x.to(self.device)
            pred = self.adapter.predict(x, self.adj)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

        preds = np.concatenate(all_preds, axis=0)    # (samples, H, N, d or no d)
        targets = np.concatenate(all_targets, axis=0)

        if self.task == "regression":
            mask = self._build_extreme_mask(targets)
            metrics = regression.compute_all(
                preds, targets,
                extreme_mask=mask,
                vpt_epsilon=self.vpt_epsilon,
                mean=self.norm_mean,
                std=self.norm_std,
                var_names=self.var_names,
            )
            per_station = metrics.pop("per_station")
            self._save_per_station(per_station)
        else:
            metrics = classification.compute_all(preds, targets)

        # Inference time (measured after best checkpoint loaded)
        infer = record_inference_time(
            self.adapter, self.test_loader, self.adj, self.device
        )
        metrics.update(infer)

        self._mlflow_log_metrics(
            {f"test/{k}": v for k, v in self._flatten_for_mlflow(metrics).items()}
        )

        summary_path = self.output_dir / "test_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _flatten_for_mlflow(self, metrics: dict, prefix: str = "") -> dict:
        """Flatten nested dicts to dot-separated keys for MLflow. Scalars only."""
        flat = {}
        for k, v in metrics.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)):
                flat[key] = v
            elif isinstance(v, dict):
                flat.update(self._flatten_for_mlflow(v, prefix=key))
        return flat

    def _save_checkpoint(self, val_loss: float) -> None:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            torch.save(self.adapter.model.state_dict(), self.checkpoint_path)
        else:
            self.patience_counter += 1

    def _mlflow_log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.mlflow_enabled:
            return
        try:
            import mlflow  # noqa: PLC0415
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass
