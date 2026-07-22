"""
Phase-2 classification unit tests.

The Phase-2 plan (§5) says: no adapters change; the only shape work is that
the trainer subsets model outputs on the trailing (variable) axis before
comparing against per-variable labels. These tests exercise that contract at
the boundaries where the plan can go wrong.

Coverage:

  1. dataset.STGNNDataset per_variable mode returns (H, N, d_label) labels
     and subsets by label_indices correctly.
  2. dataset.STGNNDataset "any" mode returns (H, N) labels (backwards compat).
  3. losses.compute_class_weights returns a (d_label,) vector for per-var
     labels and a scalar for the flat ANY case.
  4. losses.wbce_loss builds BCEWithLogitsLoss with a (d_label,) pos_weight.
  5. classification.select_thresholds returns per-variable τ ∈ [0.05, 0.95]
     and prefers the F1-optimal value on a designed signal.
  6. classification.compute_all handles:
     - matching (samples, H, N, d_label) shapes
     - zero-positive stations (NaN in per_station, coverage counter updated)
     - per_variable dict + macro dict populated
  7. labels.apply_tp_wet_hour_rule fires only when dry-hour fraction > 90%.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from preprocessing.dataset import STGNNDataset, build_dataloaders
from preprocessing.labels import (
    apply_tp_wet_hour_rule,
    compute_thresholds,
    label_all_splits,
)
from training.losses import compute_class_weights, wbce_loss
from evaluation.classification import (
    compute_all,
    select_thresholds,
)


# --------------------------------------------------------------------------- #
# Test data helpers
# --------------------------------------------------------------------------- #

VAR_NAMES = ["t2m", "d2m", "tp", "u10", "v10", "sp", "ws10", "rh"]
LABEL_VARS = ["t2m", "d2m", "tp", "ws10", "rh", "sp"]              # §2.4 default
LABEL_INDICES = [VAR_NAMES.index(v) for v in LABEL_VARS]           # [0,1,2,6,7,5]

T, N, D = 200, 10, len(VAR_NAMES)
INPUT_LEN, HORIZON = 12, 12


def _make_features_and_labels():
    """Random-ish physical-units features and their derived binary labels."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((T, N, D)).astype(np.float32)
    # 95th percentile threshold per (station, variable) on features.
    thresholds = np.percentile(features, 95.0, axis=0).astype(np.float32)
    labels = (features > thresholds[None, :, :]).astype(np.int8)
    return features, labels, thresholds


# --------------------------------------------------------------------------- #
# dataset.py — per_variable vs any label modes
# --------------------------------------------------------------------------- #

def test_per_variable_label_shape():
    features, labels, _ = _make_features_and_labels()
    ds = STGNNDataset(
        features=features, labels=labels,
        input_len=INPUT_LEN, horizon=HORIZON,
        task="classification",
        label_mode="per_variable",
        label_indices=LABEL_INDICES,
    )
    x, y = ds[0]
    assert x.shape == (INPUT_LEN, N, D)
    assert y.shape == (HORIZON, N, len(LABEL_VARS)), y.shape
    assert y.dtype == torch.float32


def test_per_variable_label_content_matches_indices():
    features, labels, _ = _make_features_and_labels()
    ds = STGNNDataset(
        features=features, labels=labels,
        input_len=INPUT_LEN, horizon=HORIZON,
        task="classification",
        label_mode="per_variable",
        label_indices=LABEL_INDICES,
    )
    _, y = ds[5]
    expected = labels[5 + INPUT_LEN : 5 + INPUT_LEN + HORIZON][..., LABEL_INDICES]
    np.testing.assert_array_equal(y.numpy().astype(np.int8), expected)


def test_any_label_mode_still_returns_2d():
    features, labels, _ = _make_features_and_labels()
    ds = STGNNDataset(
        features=features, labels=labels,
        input_len=INPUT_LEN, horizon=HORIZON,
        task="classification",
        label_mode="any",
        label_indices=LABEL_INDICES,
    )
    _, y = ds[0]
    assert y.shape == (HORIZON, N)


def test_regression_path_unchanged():
    features, _, _ = _make_features_and_labels()
    ds = STGNNDataset(
        features=features, labels=None,
        input_len=INPUT_LEN, horizon=HORIZON,
        task="regression",
    )
    x, y = ds[0]
    assert x.shape == (INPUT_LEN, N, D)
    assert y.shape == (HORIZON, N, D)


def test_build_dataloaders_classification_smoke():
    """Loader iterates without crashing at per-variable classification."""
    features, labels, _ = _make_features_and_labels()
    tr, va, te = build_dataloaders(
        features, features, features,
        labels,   labels,   labels,
        input_len=INPUT_LEN, horizon=HORIZON,
        task="classification", batch_size=4, num_workers=0,
        label_mode="per_variable",
        var_names=VAR_NAMES,
        label_variables=LABEL_VARS,
    )
    for loader in (tr, va, te):
        x, y = next(iter(loader))
        assert y.shape[-1] == len(LABEL_VARS)


# --------------------------------------------------------------------------- #
# losses.py — vector pos_weight
# --------------------------------------------------------------------------- #

def test_compute_class_weights_per_variable_vector():
    rng = np.random.default_rng(1)
    # 4-D labels with a known positive rate per channel
    labels = rng.random((100, 12, 20, 3)) < np.array([0.05, 0.10, 0.20])
    labels = labels.astype(np.int8)
    w_0, w_1 = compute_class_weights(labels)
    assert isinstance(w_0, np.ndarray) and w_0.shape == (3,)
    assert isinstance(w_1, np.ndarray) and w_1.shape == (3,)
    # w_1 should be higher where positives are rarer
    assert w_1[0] > w_1[1] > w_1[2]


def test_compute_class_weights_scalar_case():
    labels = (np.random.default_rng(2).random(1000) < 0.05).astype(np.int8)
    w_0, w_1 = compute_class_weights(labels)
    assert isinstance(w_0, float)
    assert isinstance(w_1, float)
    assert w_1 > 15  # roughly 19 for a 5% rate


def test_wbce_loss_pos_weight_shape():
    w_0 = np.ones(3, dtype=np.float32)
    w_1 = np.array([19.0, 9.0, 4.0], dtype=np.float32)
    loss = wbce_loss(w_0, w_1, torch.device("cpu"))
    assert loss.pos_weight.shape == (3,)
    np.testing.assert_allclose(loss.pos_weight.cpu().numpy(), [19.0, 9.0, 4.0])


def test_wbce_loss_broadcasts_over_batch_and_horizon():
    """A single loss step doesn't raise on (B, H, N, d_label) logits."""
    w_0 = np.ones(2, dtype=np.float32)
    w_1 = np.array([19.0, 9.0], dtype=np.float32)
    loss_fn = wbce_loss(w_0, w_1, torch.device("cpu"))
    logits = torch.randn(4, 12, 10, 2, requires_grad=True)
    targets = torch.randint(0, 2, (4, 12, 10, 2), dtype=torch.float32)
    loss = loss_fn(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


# --------------------------------------------------------------------------- #
# labels.py — tp wet-hour rule
# --------------------------------------------------------------------------- #

def test_tp_wet_hour_rule_fires_when_mostly_dry():
    """Station 0 is 95% dry (rule fires); station 1 is 50% dry (rule doesn't)."""
    rng = np.random.default_rng(3)
    N_ = 2
    T_ = 5000
    var_names = ["tp"]
    tp = np.zeros((T_, N_, 1), dtype=np.float32)
    # Station 0: 5% wet, 95% dry — but with meaningfully wet values
    is_wet_s0 = rng.random(T_) < 0.05
    tp[is_wet_s0, 0, 0] = rng.exponential(scale=5e-4, size=is_wet_s0.sum()).astype(np.float32) + 1e-4
    # Station 1: 50% wet
    is_wet_s1 = rng.random(T_) < 0.5
    tp[is_wet_s1, 1, 0] = rng.exponential(scale=5e-4, size=is_wet_s1.sum()).astype(np.float32) + 1e-4

    naive = compute_thresholds(tp)
    updated, trigger = apply_tp_wet_hour_rule(tp, naive, var_names)

    assert trigger[0] is True or trigger[0] == True  # station 0 fires
    assert trigger[1] is False or trigger[1] == False  # station 1 does not
    # At station 0, override should raise the threshold above trace
    assert updated[0, 0] > naive[0, 0]


def test_label_all_splits_returns_five_tuple():
    features, _, _ = _make_features_and_labels()
    tr, va, te = features[:100], features[100:150], features[150:]
    result = label_all_splits(tr, va, te, var_names=VAR_NAMES)
    assert len(result) == 5
    y_tr, y_va, y_te, thresholds, tp_trigger = result
    assert y_tr.shape == tr.shape
    assert thresholds.shape == (N, D)
    assert tp_trigger.shape == (N,)


# --------------------------------------------------------------------------- #
# evaluation/classification.py — thresholds + metrics
# --------------------------------------------------------------------------- #

def test_select_thresholds_prefers_higher_f1():
    """Constructed: variable 0 wants τ near 0.7; variable 1 near 0.3."""
    rng = np.random.default_rng(4)
    samples, H, N_, d = 30, 12, 8, 2
    val_targets = rng.integers(0, 2, size=(samples, H, N_, d)).astype(np.int8)
    val_probs = np.zeros_like(val_targets, dtype=np.float32)
    # Variable 0: positives get scores near 0.9, negatives near 0.5
    val_probs[..., 0] = np.where(val_targets[..., 0] == 1, 0.9, 0.5)
    # Variable 1: positives near 0.4, negatives near 0.2
    val_probs[..., 1] = np.where(val_targets[..., 1] == 1, 0.4, 0.2)
    taus = select_thresholds(val_probs, val_targets, ["v0", "v1"])
    assert 0.5 <= taus[0] <= 0.95
    assert 0.05 <= taus[1] <= 0.5


def test_compute_all_shape_and_content():
    rng = np.random.default_rng(5)
    samples, H, N_, d = 40, 12, 15, 3
    targets = rng.integers(0, 2, size=(samples, H, N_, d)).astype(np.int8)
    probs = rng.random((samples, H, N_, d)).astype(np.float32)
    taus = np.full(d, 0.5, dtype=np.float32)
    var_names = ["a", "b", "c"]
    out = compute_all(probs, targets, var_names=var_names, taus=taus)
    assert set(out.keys()) >= {
        "per_variable", "macro", "taus", "label_variables",
        "per_station", "baseline_always_negative",
    }
    for v in var_names:
        for m in ("precision", "recall", "f1", "csi", "auc_roc", "pr_auc", "accuracy"):
            assert m in out["per_variable"][v]
    # per_station arrays
    for m in ("precision", "recall", "f1", "csi", "auc_roc", "pr_auc", "accuracy"):
        assert out["per_station"][m].shape == (N_, d)


def test_compute_all_nan_skip_zero_positive_stations():
    """Stations with no positives must show NaN in per_station.f1, not crash."""
    samples, H, N_, d = 30, 12, 5, 1
    targets = np.zeros((samples, H, N_, d), dtype=np.int8)
    # Only station 2 has any positives
    targets[:5, :, 2, 0] = 1
    probs = np.full_like(targets, 0.6, dtype=np.float32)
    out = compute_all(probs, targets, var_names=["v"], taus=np.array([0.5]))
    per_station_f1 = out["per_station"]["f1"][:, 0]
    assert np.isnan(per_station_f1[0])
    assert np.isnan(per_station_f1[1])
    assert not np.isnan(per_station_f1[2])
    # Coverage counter reflects it
    assert out["per_variable"]["v"]["station_coverage"] == 1
