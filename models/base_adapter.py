"""
Abstract base class that every model adapter must implement.

This contract is the only interface the training loop (training/trainer.py)
ever uses. The trainer has no knowledge of any model's internals beyond these
four methods.

All adapters inherit from BaseAdapter and override these methods. The adapter
pattern preserves the original cloned repository code untouched — it is the
bridge between each model's internal conventions and the shared pipeline.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAdapter(ABC):

    @abstractmethod
    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> nn.Module:
        """
        Instantiate and return the model from the cloned repo.

        Args:
            config: merged experiment config dict.
            graph: adjacency matrix tensor of shape (N, N).

        Returns:
            The configured nn.Module.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training. Called with model in train mode, gradients enabled.

        Args:
            x: input tensor of shape (batch, T, N, d).
            adj: adjacency matrix of shape (N, N).

        Returns:
            Predictions of shape (batch, H, N, d) for regression
            or (batch, H, N) for classification.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass at evaluation time. Called with model in eval mode,
        inside torch.no_grad().

        Separate from forward() because some models differ between training
        and inference — diffusion models run the full denoising chain at test
        time; ODE models may use tighter solver tolerances at evaluation.

        Args:
            x: input tensor of shape (batch, T, N, d).
            adj: adjacency matrix of shape (N, N).

        Returns:
            Predictions of shape (batch, H, N, d) for regression
            or (batch, H, N) for classification.
        """
        raise NotImplementedError

    @abstractmethod
    def get_param_count(self) -> int:
        """Return total trainable parameter count."""
        raise NotImplementedError
