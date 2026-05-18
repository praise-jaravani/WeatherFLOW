"""
Shared training loop. Every model in the pipeline runs through this Trainer.

The Trainer accepts a model adapter, DataLoaders, a config, and an output
directory. It runs the training loop for a single seed. The outer run loop in
cli.py instantiates Trainer ten times (seeds 42–51) and aggregates results.

Responsibilities:
- Per-epoch train and validation passes
- Gradient clipping (max norm from config)
- Early stopping on validation loss (patience from config)
- Checkpoint saving: best model weights saved as checkpoint.pt
- Per-epoch metric logging to the terminal (rich/tqdm) and MLflow
- ETA computation updated after each epoch

The Trainer never calls model internals directly — it calls adapter.forward()
during training and adapter.predict() during validation and test evaluation.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader

from models.base_adapter import BaseAdapter


class Trainer:
    def __init__(
        self,
        adapter: BaseAdapter,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        output_dir: Path,
        seed: int,
    ):
        raise NotImplementedError

    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop for one seed.

        Returns:
            Dictionary of per-epoch metrics and final test metrics.
        """
        raise NotImplementedError

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns mean train loss."""
        raise NotImplementedError

    def _validate_epoch(self, epoch: int) -> float:
        """Run one validation epoch. Returns mean val loss."""
        raise NotImplementedError

    def _evaluate_test(self) -> Dict[str, float]:
        """Load best checkpoint and evaluate on the test set."""
        raise NotImplementedError

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model weights if val_loss is the best seen so far."""
        raise NotImplementedError
