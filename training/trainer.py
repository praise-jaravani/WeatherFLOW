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
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
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


# Run_Plan_Spec §5: raised by Trainer when train_loss or grad_norm is
# non-finite for 3 consecutive epochs. The worker catches this, writes the
# FAILED_DIVERGED marker, and moves on. Never retried — one bad config
# doesn't get to burn 10× GPU-hours in a loop.
class TrainingDiverged(RuntimeError):
    def __init__(self, epoch: int, streak: int, last_loss: float, last_norm: float):
        super().__init__(
            f"Training diverged at epoch {epoch}: "
            f"{streak} consecutive non-finite epochs "
            f"(last train_loss={last_loss}, grad_norm={last_norm})"
        )
        self.epoch = epoch
        self.streak = streak


def _epoch_shuffle_seed(base_seed: int, epoch: int) -> int:
    """Deterministic per-epoch seed for shuffling + dropout.

    Run_Plan_Spec §1.1: derive shuffle seed as f(base_seed, epoch) at each
    epoch start so a resumed run picks up the identical stochastic trajectory
    as an uninterrupted run. Uses hashlib for stability across Python
    versions (Python's built-in hash() is salted per process).
    """
    import hashlib
    payload = f"{int(base_seed)}:{int(epoch)}".encode()
    digest = hashlib.sha256(payload).digest()
    # Take first 8 bytes as an unsigned int, mask to 63 bits so it fits
    # torch.manual_seed's accepted range (int64, non-negative).
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


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

        # Phase 2: label variable names (subset of var_names). Falls back to
        # var_names if not set — appropriate for regression and for the legacy
        # all-8 label case.
        self.label_variables = config.get("data", {}).get("label_variables") or var_names
        # Index tensor for subsetting model output channels to label channels.
        # Model outputs (B, H, N, n_vars); labels are (B, H, N, d_label). We
        # subset the pred's trailing axis to line the two up. When
        # label_variables == var_names this is a no-op mask.
        if var_names is not None and self.label_variables is not None:
            try:
                _idx = [var_names.index(v) for v in self.label_variables]
                self._label_idx = torch.tensor(_idx, dtype=torch.long)
            except ValueError:
                self._label_idx = None
        else:
            self._label_idx = None

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
        # Run_Plan_Spec §1.1: full training state for atomic resume.
        # Distinct from checkpoint.pt (best model weights only).
        self.state_path = self.output_dir / "state.pt"
        # §5: streak of consecutive non-finite train epochs; resets on any finite epoch.
        self.diverged_streak = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log file opened in append mode when resuming an existing run,
        # write mode for a fresh start. Decided in train() once we know
        # whether a state.pt exists.
        self._log_file = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        stop_check: Optional[Callable[[], bool]] = None,
        epoch_callback: Optional[Callable[[int, float, float], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full training loop for one seed.

        Args:
            stop_check: optional callable returning True if a graceful stop
                has been requested (SIGUSR1 handler in the worker). Polled
                at every epoch boundary — if True, checkpoint then return
                the partial history with 'stopped_reason'='signal'. The
                worker's next link then resumes from state.pt.
            epoch_callback: optional callable invoked at the END of each
                completed epoch with signature ``(epoch, val_loss, best_val_loss)``.
                Returns True if training should stop (e.g. Optuna's median-pruner
                has decided this trial is a loser). When True, the trainer sets
                'stopped_reason'='pruned', checkpoints, and returns early. Used
                by HPO v2 (`scripts/hpo_run_optuna.py`); production paths pass None.

        Returns:
            Dict with 'train_loss', 'val_loss' history lists, 'stopped_epoch',
            and 'test_metrics' dict from the best checkpoint. If stop_check
            fires, returns early with 'stopped_reason'='signal' and no
            'test_metrics' key — successor link will complete the run.
            If epoch_callback returns True, returns early with
            'stopped_reason'='pruned' and no 'test_metrics' key.

        Raises:
            TrainingDiverged: after 3 consecutive non-finite epochs (§5).
        """
        # Record which loss function actually ran, so the archived history
        # is self-describing (per §1 of the 2026-07-28 review directive).
        loss_type = (
            f"native:{type(self.adapter).__name__}"
            if self._uses_native_loss()
            else type(self.loss_fn).__name__
        )
        default_history: Dict[str, Any] = {
            "loss_type": loss_type,
            "train_loss": [], "val_loss": [], "grad_norm": [],
        }

        # Run_Plan_Spec §1.1: resume from state.pt if present.
        start_epoch, history = self._load_full_state(default_history)
        # Open log in append mode on resume, write mode on fresh start.
        log_mode = "a" if start_epoch > 1 else "w"
        self._log_file = open(self.output_dir / "train.log", log_mode, buffering=1)
        if start_epoch > 1:
            self._log_file.write(f"# RESUMED from epoch {start_epoch - 1}\n")

        stopped_epoch = self.epochs
        stopped_reason: Optional[str] = None

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
            # Advance the progress bar to reflect completed epochs when resuming.
            if start_epoch > 1:
                progress.update(epoch_task, advance=start_epoch - 1)

            for epoch in range(start_epoch, self.epochs + 1):
                # Run_Plan_Spec §1.1: seed torch/numpy/python at each epoch
                # start with f(base_seed, epoch). This makes the epoch's
                # dataloader shuffle order + dropout patterns identical
                # between a control run and an interrupted-resumed run.
                # Applied uniformly (including epoch 1) so the two runs
                # follow the same reseeding schedule.
                self._seed_epoch(epoch)

                # Graceful-stop check BEFORE work begins for the epoch.
                # If SIGUSR1 fired between epochs, exit here — last state
                # was already checkpointed, successor resumes from epoch.
                if stop_check is not None and stop_check():
                    stopped_reason = "signal"
                    stopped_epoch = epoch - 1
                    break

                # Exposed on `self` so the native-loss hook can compute
                # its epoch-dependent `var_coeff` (upstream cosine-annealed).
                self._epoch = epoch
                train_loss, grad_norm = self._train_epoch()
                val_loss = self._validate_epoch()

                self.scheduler.step()
                self._save_checkpoint(val_loss)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["grad_norm"].append(grad_norm)

                # §5 divergence guard: 3 consecutive non-finite epochs → raise.
                if not (math.isfinite(train_loss) and math.isfinite(grad_norm)):
                    self.diverged_streak += 1
                    if self.diverged_streak >= 3:
                        # Persist state so the marker is consistent with disk state.
                        self._save_full_state(epoch, history)
                        (self.output_dir / "FAILED_DIVERGED").write_text(
                            f"epoch={epoch} streak={self.diverged_streak} "
                            f"train_loss={train_loss} grad_norm={grad_norm}\n"
                        )
                        self._log_file.write(
                            f"# DIVERGED at epoch={epoch} streak={self.diverged_streak}\n"
                        )
                        self._log_file.close()
                        raise TrainingDiverged(
                            epoch=epoch, streak=self.diverged_streak,
                            last_loss=train_loss, last_norm=grad_norm,
                        )
                else:
                    self.diverged_streak = 0

                # Run_Plan_Spec §1.1: atomic per-epoch state save.
                # This is the load-bearing piece of the C1 guarantee —
                # a timeout mid-next-epoch loses at most the next epoch,
                # never this one.
                self._save_full_state(epoch, history)

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

                # HPO v2 hook: give an external caller (Optuna) a chance to
                # report val progress and prune the trial. Fires AFTER state
                # is saved so a pruned trial still leaves a resumable state.pt
                # on disk (safe if the caller wants to inspect it). Reports
                # best_val_loss (monotone) not this epoch's val_loss so that
                # cosine-annealing restart bumps at epochs 10, 30, ... don't
                # get a recovering trial spuriously killed by MedianPruner.
                if epoch_callback is not None:
                    try:
                        should_stop = bool(epoch_callback(epoch, val_loss, self.best_val_loss))
                    except Exception:
                        # Callback errors must not tank the training run —
                        # log and continue. Optuna's TrialPruned exception
                        # will still surface via the caller if raised outside.
                        should_stop = False
                    if should_stop:
                        stopped_epoch = epoch
                        stopped_reason = "pruned"
                        progress.update(
                            epoch_task,
                            description=f"[magenta]Pruned at epoch {epoch}[/magenta]",
                        )
                        self._log_file.write(f"# PRUNED at epoch={epoch}\n")
                        break

                if self.patience_counter >= self.patience:
                    stopped_epoch = epoch
                    progress.update(
                        epoch_task,
                        description=f"[yellow]Early stop at epoch {epoch}[/yellow]",
                    )
                    break

        history["stopped_epoch"] = stopped_epoch
        # Run_Plan_Spec §1.1: graceful stop path — successor link resumes
        # from state.pt and will run this test eval when training truly ends.
        if stopped_reason == "signal":
            history["stopped_reason"] = "signal"
            self._log_file.write(f"# STOPPED_BY_SIGNAL at end_of_epoch={stopped_epoch}\n")
            self._log_file.close()
            return history

        # End-of-training full-val pass (§1.1): if we've been running on
        # a subsampled val set for checkpoint selection, do one full-val
        # pass at the end so `final_val_loss` in the history reflects
        # the entire validation split. No-op when subsampling is off
        # (force_full just re-runs the same full pass).
        if self._val_subsample_indices() is not None:
            history["final_val_loss_full"] = self._validate_epoch(force_full=True)
        history["test_metrics"] = self._evaluate_test()

        self._log_file.close()
        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-model native-loss hook (guarded, additive, default-off).
    # ------------------------------------------------------------------
    #
    # Adapters that need to train under their published objective (e.g.
    # climode_faithful's Gaussian NLL) opt in via:
    #   adapter.uses_native_loss = True
    #   adapter.forward(x, adj) → non-Tensor object (tuple, dict, ...)
    #   adapter.native_loss(pred_object, target, **kwargs) → scalar
    #
    # Every other adapter's path is byte-identical to before — the
    # `getattr(..., False)` default returns False and the shared
    # `self.loss_fn(pred, y)` branch runs unchanged. Tested in
    # tests/test_native_loss_hook.py.
    # ------------------------------------------------------------------
    def _uses_native_loss(self) -> bool:
        return bool(getattr(self.adapter, "uses_native_loss", False))

    def _native_loss_kwargs(self) -> Dict[str, Any]:
        """
        Build the kwargs passed to `adapter.native_loss(...)`. Currently
        just `var_coeff` for climode_faithful's Gaussian NLL prior on σ
        (upstream train_region.py: `var_coeff = 2 * lr` after epoch 0,
        `1e-3` at epoch 0). Extendable for other native losses.
        """
        current_lr = self.optimiser.param_groups[0]["lr"]
        var_coeff = 1e-3 if self._epoch <= 0 else 2.0 * current_lr
        return {"var_coeff": var_coeff}

    def _train_epoch(self):
        """One training epoch. Returns (mean_train_loss, mean_grad_norm)."""
        self.adapter.model.train()
        total_loss = 0.0
        total_norm = 0.0
        n_batches = 0
        native = self._uses_native_loss()

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimiser.zero_grad()
            pred = self.adapter.forward(x, self.adj)

            if native:
                # Adapter owns the loss and the pred format. Skip
                # classification subsetting — adapters using native_loss
                # are regression-only (see FAITHFULNESS.md notes).
                loss = self.adapter.native_loss(pred, y, **self._native_loss_kwargs())
            else:
                # BCEWithLogitsLoss expects float targets; classification y is int8
                if self.task == "classification":
                    y = y.float()
                    pred = self._subset_pred_to_label_channels(pred)
                elif self.task == "regression":
                    # Univariate regime (Gaibie 2024): if the dataset sliced y
                    # to a single target channel, slice pred to the matching
                    # channel so the loss only backprops through that variable.
                    tv_idx = getattr(self.train_loader.dataset, "target_variable_index", None)
                    if tv_idx is not None and pred.shape[-1] != y.shape[-1]:
                        pred = pred[..., tv_idx:tv_idx + 1]
                loss = self.loss_fn(pred, y)

            loss.backward()
            norm = clip_gradients(self.adapter.model, self.max_norm)
            self.optimiser.step()

            total_loss += loss.item()
            total_norm += norm
            n_batches += 1

        return total_loss / max(n_batches, 1), total_norm / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Subsampled-validation hook (§1.1 of the 2026-07-28 directive).
    # Guarded, default-off: unless the config has training.val_subsample_n
    # OR the adapter declares val_subsample_n, the full val_loader runs.
    #
    # Deterministic subsample: batch indices selected once at first call
    # via a torch.Generator seeded from (val_subsample_seed default 12345)
    # so the same subset is used every epoch and every seed's run.
    # ------------------------------------------------------------------
    def _val_subsample_config(self) -> tuple[int | None, int]:
        """Return (subsample_n, seed) or (None, seed) to disable subsampling."""
        # Adapter overrides win
        n = getattr(self.adapter, "val_subsample_n", None)
        if n is None:
            n = self.config.get("training", {}).get("val_subsample_n") if hasattr(self, "config") else None
        seed = getattr(self.adapter, "val_subsample_seed", None) or 12345
        if hasattr(self, "config"):
            seed = self.config.get("training", {}).get("val_subsample_seed", seed)
        return (int(n) if n else None, int(seed))

    def _val_subsample_indices(self) -> Optional[list]:
        """Cached deterministic indices into the val_loader's batch stream."""
        n, seed = self._val_subsample_config()
        if n is None:
            return None
        if getattr(self, "_val_sub_cache", None) is not None:
            return self._val_sub_cache
        n_val = len(self.val_loader.dataset)
        if n_val <= n:
            self._val_sub_cache = list(range(n_val))
        else:
            g = torch.Generator()
            g.manual_seed(seed)
            self._val_sub_cache = torch.randperm(n_val, generator=g)[:n].tolist()
        return self._val_sub_cache

    def _validate_epoch(self, force_full: bool = False) -> float:
        """One validation epoch. Returns mean val loss.

        Uses a subsampled window subset when configured (via
        `training.val_subsample_n`), giving a stable checkpoint-selection
        signal at a fraction of the cost. `force_full=True` bypasses the
        subsample — used for the end-of-training final eval.
        """
        total_loss = 0.0
        n_batches = 0
        native = self._uses_native_loss()

        sub_indices = None if force_full else self._val_subsample_indices()
        if sub_indices is not None:
            # Build a light-weight subset view of the same DataLoader dataset
            subset = torch.utils.data.Subset(self.val_loader.dataset, sub_indices)
            loader = torch.utils.data.DataLoader(
                subset, batch_size=self.val_loader.batch_size,
                shuffle=False, num_workers=0, pin_memory=False,
            )
        else:
            loader = self.val_loader

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            if native:
                # Adapters using native_loss must expose a training-mode
                # forward that returns the object native_loss expects
                # (e.g. climode_faithful returns (μ, σ) from forward).
                # Wrap in no_grad since we're in the val loop.
                with torch.no_grad():
                    pred = self.adapter.forward(x, self.adj)
                    loss = self.adapter.native_loss(pred, y, **self._native_loss_kwargs())
            else:
                pred = self.adapter.predict(x, self.adj)
                if self.task == "classification":
                    y = y.float()
                    pred = self._subset_pred_to_label_channels(pred)
                elif self.task == "regression":
                    # Univariate regime — mirror the _train_epoch slicing so
                    # train_loss and val_loss remain comparable in scale.
                    tv_idx = getattr(self.val_loader.dataset, "target_variable_index", None)
                    if tv_idx is not None and pred.shape[-1] != y.shape[-1]:
                        pred = pred[..., tv_idx:tv_idx + 1]
                loss = self.loss_fn(pred, y)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _build_extreme_mask(self, targets_norm: np.ndarray) -> Optional[np.ndarray]:
        """targets_norm: (samples, H, N, d). Returns (samples, H, N, d) bool, or None."""
        if self.thresholds is None or self.norm_mean is None or self.norm_std is None:
            return None
        # calendar_hour_relative thresholds are (N, D, 366, 24) — cannot broadcast
        # against (samples, H, N, d) without per-sample timestamp indexing. Skip.
        if self.thresholds.ndim != 2:
            return None
        m = self.norm_mean.reshape(1, 1, *self.norm_mean.shape[-2:])
        s = self.norm_std.reshape(1, 1, *self.norm_std.shape[-2:])
        targets_phys = targets_norm * s + m
        return targets_phys > self.thresholds[None, None, :, :]

    def _save_per_station(self, per_station: dict) -> None:
        """Save (N, d) per-station arrays for Phase 3 geographic stratification."""
        out_path = self.output_dir / "per_station.npz"
        np.savez(str(out_path), **{k: v for k, v in per_station.items() if isinstance(v, np.ndarray)})

    def _subset_pred_to_label_channels(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Subset a (B, H, N, n_vars) tensor to (B, H, N, d_label) via the
        `_label_idx` computed once at construction. No-op when the subset is
        the full var set. Trailing-axis-only.
        """
        if self._label_idx is None or pred.dim() != 4:
            return pred
        idx = self._label_idx.to(pred.device)
        if idx.shape[0] == pred.shape[-1] and torch.equal(idx, torch.arange(pred.shape[-1], device=pred.device)):
            return pred
        return pred.index_select(-1, idx)

    def _collect_val_predictions(self) -> tuple:
        """
        Run the loaded checkpoint over the validation set and return
        (val_probs, val_targets) as numpy arrays. Used only by classification —
        Phase-2 plan §4.3 requires validation-selected decision thresholds.
        Adapters return LOGITS (per §3.1 no adapter changes); the sigmoid is
        applied here in the trainer.
        """
        val_probs, val_targets = [], []
        for x, y in self.val_loader:
            x = x.to(self.device)
            logits = self.adapter.predict(x, self.adj)
            logits = self._subset_pred_to_label_channels(logits)
            val_probs.append(torch.sigmoid(logits).cpu().numpy())
            val_targets.append(y.numpy())
        return np.concatenate(val_probs, axis=0), np.concatenate(val_targets, axis=0)

    def _evaluate_test(self) -> Dict[str, Any]:
        """Load best checkpoint and evaluate on the test set."""
        # TODO(perf): stream test eval. At 999 nodes the accumulated all_preds +
        # all_targets reach ~35 GB peak (45K samples × 12H × 999N × 8d × 4B × 2),
        # forcing --mem=128G. Refactor to compute metrics chunk-by-chunk so we
        # never hold the full test prediction set in RAM. See specs/ for context.
        import resource
        def _rss_gib():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        print(f"[mem] {_rss_gib():.2f} GiB entering _evaluate_test", flush=True)
        if self.checkpoint_path.exists():
            state = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )
            self.adapter.model.load_state_dict(state)

        all_preds, all_targets = [], []

        for x, y in self.test_loader:
            x = x.to(self.device)
            pred = self.adapter.predict(x, self.adj)
            # Classification: adapters return LOGITS; subset to label channels
            # and sigmoid here (§3.1).
            if self.task == "classification":
                pred = self._subset_pred_to_label_channels(pred)
                pred = torch.sigmoid(pred)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

        preds = np.concatenate(all_preds, axis=0)    # (samples, H, N, d or no d)
        targets = np.concatenate(all_targets, axis=0)
        # Free the intermediate list of chunks now that the concatenated
        # arrays own the data. Each chunk list held ~13 GB at N=999.
        del all_preds, all_targets
        print(f"[mem] {_rss_gib():.2f} GiB after test preds+targets concat  "
              f"(preds={preds.nbytes/1e9:.1f}GB targets={targets.nbytes/1e9:.1f}GB)", flush=True)

        if self.task == "regression":
            # Univariate regime (Gaibie 2024): the collected `targets` always
            # has shape (S, H, N, 1) when `target_variable_index` is set.
            # Two adapter-output shapes are possible for preds:
            #   (a) (S, H, N, D)  — adapter emits full-D (MTGODE, AGCRN, CLCRN
            #       — the univariate slice happens at loss time). Need to
            #       slice preds AND stats to the target channel.
            #   (b) (S, H, N, 1)  — adapter emits univariate directly (STGODE
            #       — the univariate slice happens INSIDE the adapter).
            #       preds already correct shape; only stats need slicing.
            # In both cases, we end with univariate preds + univariate stats.
            tv_idx = getattr(self.test_loader.dataset, "target_variable_index", None)
            mean_ch = self.norm_mean.shape[-1] if self.norm_mean is not None else preds.shape[-1]
            needs_slice = (tv_idx is not None
                           and (preds.shape[-1] != targets.shape[-1]
                                or preds.shape[-1] < mean_ch))
            if needs_slice:
                # (a) full-D adapter: slice preds to target channel
                if preds.shape[-1] > 1:
                    preds = preds[..., tv_idx:tv_idx + 1]
                # (a) + (b): always slice stats to target channel
                eval_mean = self.norm_mean[..., tv_idx:tv_idx + 1] if self.norm_mean is not None else None
                eval_std  = self.norm_std[...,  tv_idx:tv_idx + 1] if self.norm_std  is not None else None
                eval_thresholds = self.thresholds[..., tv_idx:tv_idx + 1] if self.thresholds is not None else None
                eval_var_names  = [self.var_names[tv_idx]] if self.var_names is not None else None
                # Also swap in the sliced thresholds for _build_extreme_mask
                saved_thresholds, saved_mean, saved_std = self.thresholds, self.norm_mean, self.norm_std
                self.thresholds, self.norm_mean, self.norm_std = eval_thresholds, eval_mean, eval_std
                try:
                    mask = self._build_extreme_mask(targets)
                finally:
                    self.thresholds, self.norm_mean, self.norm_std = saved_thresholds, saved_mean, saved_std
            else:
                eval_mean = self.norm_mean
                eval_std = self.norm_std
                eval_var_names = self.var_names
                mask = self._build_extreme_mask(targets)
            metrics = regression.compute_all(
                preds, targets,
                extreme_mask=mask,
                vpt_epsilon=self.vpt_epsilon,
                mean=eval_mean,
                std=eval_std,
                var_names=eval_var_names,
            )
            per_station = metrics.pop("per_station")
            self._save_per_station(per_station)
        else:
            # Phase-2 plan §4.3: decision thresholds selected on the VALIDATION
            # split by max-F1, per label variable. Never on 0.5, never on test.
            val_preds, val_targets = self._collect_val_predictions()
            label_vars = self.label_variables or [f"var{i}" for i in range(val_preds.shape[-1])]
            taus = classification.select_thresholds(val_preds, val_targets, label_vars)
            # Release the validation accumulators (~26 GB at N=999) before
            # compute_all — they're no longer needed once taus is chosen and
            # they'd otherwise sit alive alongside test preds/targets for the
            # duration of compute_all, pushing peak RSS toward the 96G limit.
            print(f"[mem] {_rss_gib():.2f} GiB after collect_val + select_thresholds", flush=True)
            del val_preds, val_targets
            metrics = classification.compute_all(
                preds, targets, var_names=label_vars, taus=taus,
            )
            print(f"[mem] {_rss_gib():.2f} GiB after compute_all", flush=True)
            per_station = metrics.pop("per_station")
            # per_station comes back as a dict of (N, d_label) arrays; save as npz.
            self._save_per_station(per_station)

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

    # ------------------------------------------------------------------
    # Run_Plan_Spec §1.1: per-epoch checkpoint-and-resume
    # ------------------------------------------------------------------

    def _seed_epoch(self, epoch: int) -> None:
        """Deterministic per-epoch reseed of torch/numpy/python.

        Called at every epoch start (including epoch 1) so that a control
        run and an interrupted-resumed run follow the same reseed schedule
        and therefore produce bit-identical epoch outputs under
        cudnn.deterministic=True.

        DataLoader shuffle uses torch's global generator by default, so
        reseeding here fixes the shuffle order for the epoch about to run.
        """
        s = _epoch_shuffle_seed(self.seed, epoch)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        np.random.seed(s & 0xFFFF_FFFF)  # numpy wants uint32
        random.seed(s)

    def _save_full_state(self, epoch: int, history: Dict[str, Any]) -> None:
        """Atomic write of full training state to state.pt.

        Writes state.pt.tmp then os.replace — atomic on POSIX and cluster
        filesystems (Lustre, GPFS). A crash mid-write leaves either the
        old state.pt intact or the new one — never a torn file.
        """
        state = {
            "epoch": int(epoch),
            "model_state": self.adapter.model.state_dict(),
            "optimizer_state": self.optimiser.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": float(self.best_val_loss),
            "patience_counter": int(self.patience_counter),
            "diverged_streak": int(self.diverged_streak),
            "history": history,
            "seed": int(self.seed),
            "rng_state": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda_all": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else []
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        tmp = self.output_dir / "state.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.state_path)

    def _load_full_state(
        self, default_history: Dict[str, Any]
    ) -> tuple[int, Dict[str, Any]]:
        """Load state.pt if it exists; return (start_epoch, history).

        Fresh start returns (1, default_history). Resume returns
        (last_completed_epoch + 1, restored_history). Restores model,
        optimizer, scheduler, best-val, patience counter, divergence
        streak, and RNG states (torch CPU + cuda-all, numpy, python).
        """
        if not self.state_path.exists():
            return 1, default_history

        # Load with map_location='cpu' so RNG ByteTensors stay on CPU
        # (torch.set_rng_state rejects non-CPU tensors). Model + optimizer
        # tensors are moved back to self.device explicitly below.
        state = torch.load(
            self.state_path, map_location="cpu", weights_only=False
        )
        self.adapter.model.load_state_dict(state["model_state"])
        self.adapter.model.to(self.device)
        self.optimiser.load_state_dict(state["optimizer_state"])
        # optimizer.load_state_dict does not move tensors — walk and move.
        for opt_state in self.optimiser.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(self.device)
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.best_val_loss = float(state["best_val_loss"])
        self.patience_counter = int(state["patience_counter"])
        self.diverged_streak = int(state.get("diverged_streak", 0))

        rng = state["rng_state"]
        # RNG ByteTensors must be on CPU for set_rng_state; on GPU for cuda.
        torch.set_rng_state(rng["torch_cpu"].cpu() if hasattr(rng["torch_cpu"], "cpu") else rng["torch_cpu"])
        if torch.cuda.is_available() and rng["torch_cuda_all"]:
            # torch.cuda.set_rng_state_all accepts CPU ByteTensors and moves them.
            torch.cuda.set_rng_state_all([t.cpu() if hasattr(t, "cpu") else t for t in rng["torch_cuda_all"]])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

        start_epoch = int(state["epoch"]) + 1
        history = state.get("history", default_history)
        return start_epoch, history

    def _mlflow_log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.mlflow_enabled:
            return
        try:
            import mlflow  # noqa: PLC0415
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass
