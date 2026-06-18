# WeatherFLOW — Phase 2 Classification Head (MTGODE + ClimODE)

**Problem:** every adapter's `forward()`/`predict()` returns `(B, H, N, d)`, but classification labels are `(B, H, N)` — one binary "extreme / not" per node per horizon step. Running `--task classification` raises:
```
ValueError: Target size ([512, 12, 10]) must be the same as input size ([512, 12, 10, 8])
```
**Fix:** a task-conditional output head that collapses the `d` (variable) dimension to a single per-node logit. Regression path is untouched; classification adds a learned `Linear(d, 1)` over the existing output. Logits are returned from `forward()` (for a numerically-stable `BCEWithLogitsLoss`), and `predict()` applies the sigmoid for evaluation.

This keeps the upstream model repos untouched — the head lives in the wrapper layer, consistent with the adapter pattern.

---

## 1. Shared head — `models/heads.py` (new file)

```python
"""Task-conditional output heads shared across all model wrappers."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Collapse a per-variable forecast (B, H, N, d) into one binary logit per
    node per horizon step (B, H, N).

    Weights are shared across batch, horizon, and nodes — the same linear map
    is applied at every (b, h, n). Returns raw LOGITS (no sigmoid); the loss
    (BCEWithLogitsLoss) applies the sigmoid internally for numerical stability,
    and predict() applies it explicitly for evaluation.
    """

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, N, d) -> (B, H, N)
        return self.proj(x).squeeze(-1)
```

> A single `Linear(d, 1)` is the minimal correct choice. If you later want more capacity, swap `self.proj` for a small MLP (`Linear(d, h) → ReLU → Linear(h, 1)`); the interface is unchanged.

---

## 2. ClimODE — `models/continuous_time/climode_model.py`

**`__init__`** — after `d`, `H`, `T` are read, add the task and head:
```python
from models.heads import ClassificationHead   # at top of file

# ... inside __init__, after hidden_dim / ode_cfg are set:
self.task = config.get("task", "regression")
self.head = ClassificationHead(d) if self.task == "classification" else None
```

**`forward`** — replace the final return block:
```python
        # out: (H+1, B, N, d) — skip t=0 (initial condition)
        out = out[1:]                  # (H, B, N, d)
        out = out.permute(1, 0, 2, 3)  # (B, H, N, d)

        if self.task == "classification":
            return self.head(out)      # (B, H, N) logits
        return out                     # (B, H, N, d)
```

**`models/continuous_time/climode_adapter.py`** — two edits.

`build_model`, in the `merged` dict, pass the task through:
```python
merged = {
    **config.get("model", {}),
    "ode":       config.get("ode", {}),
    "input_dim": config["meta"]["n_vars"],
    "horizon":   config["data"]["horizon"],
    "input_len": config["data"]["input_len"],
    "task":      config.get("data", {}).get("task", "regression"),   # <-- add
}
```

`predict`, apply the sigmoid for classification:
```python
def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    self.model.eval()
    with torch.no_grad():
        out = self.model(x)
        if getattr(self.model, "task", "regression") == "classification":
            out = torch.sigmoid(out)      # (B, H, N) probabilities
        return out
```
(Leave `forward()` returning logits — the WBCE loss needs logits, not probabilities.)

---

## 3. MTGODE — `models/continuous_time/mtgode_model.py`

I'm working from the wrapper's described interface (output reshaped `(B, H×d, N, 1) → (B, H, N, d)`), not its source — adapt placement to the actual code.

**`__init__`** — same two lines as ClimODE:
```python
from models.heads import ClassificationHead   # at top

# after d / H are known in __init__:
self.task = config.get("task", "regression")
self.head = ClassificationHead(d) if self.task == "classification" else None
```

**`forward`** — immediately after the line that produces the `(B, H, N, d)` tensor (the final reshape), before returning:
```python
        out = out.reshape(B, H, N, d)     # <-- your existing final reshape

        if self.task == "classification":
            return self.head(out)         # (B, H, N) logits
        return out                        # (B, H, N, d)
```

**`models/continuous_time/mtgode_adapter.py`** — same as ClimODE: add `"task"` to the config you pass into `MTGODEModel`, and sigmoid in `predict()` for classification.

> When you add the head to STLGRU / Lite-STGNN / DiffSTG / USTD later, it's the same three touch-points: import the head, set `self.task` + `self.head` in `__init__`, branch at the end of `forward`, sigmoid in `predict`. Worth telling the teammates the pattern so all eight are consistent.

---

## 4. Loss dependency — verify before running

The head returns **logits**, so the classification loss must be `BCEWithLogitsLoss`, not `BCELoss` (which expects probabilities). Check `training/losses.py`. The weighted-BCE for class imbalance is expressed via `pos_weight`:
```python
import torch
import torch.nn as nn

def weighted_bce_loss(pos_weight: torch.Tensor) -> nn.Module:
    # pos_weight up-weights the positive (extreme) class.
    # A standard choice: pos_weight = (#negative / #positive) in the TRAIN set.
    # With a 95th-percentile threshold that's ~ 0.95 / 0.05 = 19, but compute
    # it empirically per variable-collapsed label set rather than assuming 19.
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```
- Feed `forward()` output (logits) straight to this loss — do **not** sigmoid first.
- Compute `pos_weight` from the actual train-label class balance, not a hard-coded constant. Pass it from the pipeline (you already compute labels there).
- If `losses.py` currently applies its own sigmoid + `BCELoss`, either switch it to `BCEWithLogitsLoss` (recommended) or have `predict()`/the trainer handle sigmoid consistently — just don't double-sigmoid.

---

## 5. Tests to add — `tests/test_adapters.py`

```python
# For AGCRN, MTGODE, ClimODE (CLCRN skipped locally):

def test_classification_output_shape(adapter, config):
    """Classification forward returns (B, H, N), not (B, H, N, d)."""
    cfg = {**config, "data": {**config["data"], "task": "classification"}}
    model = adapter.build_model(cfg, graph)
    out = adapter.forward(x, adj)          # x: (B, T, N, d)
    assert out.shape == (B, H, N)

def test_classification_predict_is_probability(adapter, config):
    """predict() in classification mode returns values in [0, 1]."""
    # ... build with task=classification ...
    p = adapter.predict(x, adj)
    assert p.min() >= 0.0 and p.max() <= 1.0

def test_regression_unchanged_by_head(adapter, config):
    """Default (regression) still returns (B, H, N, d)."""
    out = adapter.forward(x, adj)
    assert out.shape == (B, H, N, d)

def test_classification_loss_runs(adapter, config):
    """One WBCE training step on logits decreases loss, no shape error."""
    # build classification model, BCEWithLogitsLoss(pos_weight), one step.
```

---

## 6. Verification (local smoke, once wired)

```bash
CUDA_VISIBLE_DEVICES="" python3 cli.py \
  --model mtgode --dataset era5 --task classification \
  --runs 1 --max-nodes 10 --mlflow-off \
  --config.training.epochs 3 --config.training.batch_size 512
```
- [ ] No `ValueError` on output/target shape.
- [ ] Loss is finite and decreasing.
- [ ] `summary.json` (classification) carries precision / recall / F1 / AUC-ROC.
- [ ] `predict()` outputs ∈ [0, 1].
- [ ] Same for ClimODE; AGCRN as the baseline sanity check.

---

## Scope note

This unblocks the *shape* mismatch for your two models (MTGODE, ClimODE) and AGCRN. It does **not** decide *how* the `(B, H, N)` binary label is defined from per-variable thresholds — that's a `labels.py` / dataset question the team should confirm (which variable's extremity defines a node's label, or whether it's "any variable extreme"). The head works regardless of that choice; just make sure the label definition is deliberate and documented, since it's the operational definition of "extreme event" for the entire Phase 2 result.
