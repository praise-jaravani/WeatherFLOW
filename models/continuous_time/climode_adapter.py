"""
Adapter (method) for ClimODE — training and prediction logic.

Like MTGODE, ClimODE uses an ODE solver internally, so forward() and predict()
differ in their solver tolerances:
  - forward() (training): loose tolerances via ode_integrator.train_integrate()
  - predict() (evaluation): tight tolerances via ode_integrator.eval_integrate()

ClimODE's velocity optimisation (Optim_velocity) adds learnable transport
parameters that are part of the model's parameter count.
"""

import torch
from typing import Dict, Any

from models.base_adapter import BaseAdapter
from models.continuous_time.climode_model import ClimODEModel


class ClimODEAdapter(BaseAdapter):

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> ClimODEModel:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Training pass — loose ODE tolerances."""
        raise NotImplementedError

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Evaluation pass — tight ODE tolerances, no gradient."""
        raise NotImplementedError

    def get_param_count(self) -> int:
        raise NotImplementedError
