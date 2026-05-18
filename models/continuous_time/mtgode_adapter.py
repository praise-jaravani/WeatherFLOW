"""
Adapter (method) for MTGODE — training and prediction logic.

The key distinction between forward() and predict() for MTGODE:
  - forward() (training): uses looser ODE solver tolerances (rtol=1e-3, atol=1e-4)
    for faster backpropagation through the adjoint method.
  - predict() (evaluation): uses tighter tolerances (rtol=1e-5, atol=1e-6)
    for higher accuracy on the test set.

Both paths call modules/ode_integrator.py's train_integrate() and eval_integrate()
respectively, ensuring consistent solver configuration.
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.continuous_time.mtgode_model import MTGODEModel


class MTGODEAdapter(BaseAdapter):

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> MTGODEModel:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Training pass — loose ODE tolerances."""
        raise NotImplementedError

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Evaluation pass — tight ODE tolerances, no gradient."""
        raise NotImplementedError

    def get_param_count(self) -> int:
        raise NotImplementedError
