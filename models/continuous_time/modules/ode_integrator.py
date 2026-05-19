"""
Shared ODE integration utilities for MTGODE and ClimODE.

Wraps torchdiffeq's odeint and odeint_adjoint with shared configuration logic
so both models use consistent solver settings. Solver type, step size, and
tolerances are read from the experiment config rather than hardcoded.

Training vs. evaluation: ODE solvers can use looser tolerances during training
for speed and tighter tolerances at evaluation time for accuracy. This module
exposes separate train_integrate() and eval_integrate() functions to make that
distinction explicit and match the forward()/predict() separation in the adapters.
"""

import torch
import torch.nn as nn


def train_integrate(
    func: nn.Module,
    y0: torch.Tensor,
    t_span: torch.Tensor,
    method: str = "dopri5",
    rtol: float = 1e-3,
    atol: float = 1e-4,
    adjoint: bool = True,
) -> torch.Tensor:
    """
    Integrate an ODE during training. Uses adjoint method by default for
    memory-efficient backpropagation through the solver.

    Args:
        func: the ODE function (nn.Module with forward(t, y)).
        y0: initial state tensor.
        t_span: 1-D tensor of time points to evaluate at.
        method: torchdiffeq solver name ('dopri5', 'euler', 'rk4', etc.).
        rtol, atol: relative and absolute tolerances (used for adaptive solvers).
        adjoint: use odeint_adjoint for backprop (True) or odeint (False).

    Returns:
        Tensor of states at each time in t_span, shape (len(t_span), *y0.shape).
    """
    from torchdiffeq import odeint, odeint_adjoint  # noqa: PLC0415

    kwargs = {"method": method, "rtol": rtol, "atol": atol}

    if adjoint:
        return odeint_adjoint(func, y0, t_span, **kwargs)
    return odeint(func, y0, t_span, **kwargs)


def eval_integrate(
    func: nn.Module,
    y0: torch.Tensor,
    t_span: torch.Tensor,
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> torch.Tensor:
    """
    Integrate an ODE at evaluation time. Uses tighter tolerances than training
    for higher accuracy. No adjoint needed (no backprop at eval).

    Returns:
        Tensor of states at each time in t_span, shape (len(t_span), *y0.shape).
    """
    from torchdiffeq import odeint  # noqa: PLC0415

    return odeint(func, y0, t_span, method=method, rtol=rtol, atol=atol)
