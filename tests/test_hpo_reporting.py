"""
Tests for evaluation.hpo_reporting — the tabular aggregation code. Figures are
not asserted here; a smoke of the full pipeline covers those integration-style.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.hpo_reporting import load_trial_log, summarise_study, build_full_report


def _make_fake_trial(trials_dir: Path, trial_id: int, val_loss_curve: list[float],
                     hyperparams: dict) -> None:
    """Create a fake trial dir with the minimum artefacts load_trial_log needs."""
    tdir = trials_dir / f"trial_{trial_id:02d}"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "hyperparams.json").write_text(json.dumps(hyperparams))
    (tdir / "history.json").write_text(json.dumps({
        "train_loss": [v * 1.1 for v in val_loss_curve],   # fake train, slightly worse
        "val_loss":   val_loss_curve,
        "stopped_epoch": len(val_loss_curve),
    }))


class TestLoadTrialLog:

    def test_empty_dir_returns_empty_df(self, tmp_path):
        df = load_trial_log(tmp_path)
        assert df.empty

    def test_single_trial_produces_one_row(self, tmp_path):
        _make_fake_trial(tmp_path / "trials", 0,
                         val_loss_curve=[1.0, 0.8, 0.6, 0.5, 0.55],
                         hyperparams={"data.input_len": 12, "model.hidden_dim": 32})
        df = load_trial_log(tmp_path)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["trial_number"] == 0
        assert row["best_val_loss"] == pytest.approx(0.5)
        assert row["best_epoch"] == 4   # index 3 → 1-based 4
        assert row["final_val_loss"] == pytest.approx(0.55)
        assert row["n_epochs_run"] == 5
        assert row["data.input_len"] == 12
        assert row["model.hidden_dim"] == 32

    def test_multiple_trials_ranked_by_val_loss(self, tmp_path):
        trials = tmp_path / "trials"
        _make_fake_trial(trials, 0, [1.0, 0.9, 0.8],       {"data.input_len": 12})   # best 0.8
        _make_fake_trial(trials, 1, [1.0, 0.7, 0.5],       {"data.input_len": 48})   # best 0.5 → winner
        _make_fake_trial(trials, 2, [1.0, 0.85, 0.65],     {"data.input_len": 24})   # best 0.65
        df = load_trial_log(tmp_path)
        assert list(df["rank"]) == [1, 2, 3]
        assert df.iloc[0]["trial_number"] == 1   # winner
        assert df.iloc[0]["data.input_len"] == 48
        assert list(df["best_val_loss"]) == [0.5, 0.65, 0.8]

    def test_incomplete_trial_skipped(self, tmp_path):
        trials = tmp_path / "trials"
        # complete trial
        _make_fake_trial(trials, 0, [0.5], {"x": 1})
        # incomplete: only hyperparams, no history
        (trials / "trial_01").mkdir(parents=True)
        (trials / "trial_01" / "hyperparams.json").write_text('{"x": 2}')
        df = load_trial_log(tmp_path)
        assert len(df) == 1

    def test_nested_history_format_handled(self, tmp_path):
        """cli.py's history.json can be either flat or {"seed_histories": {seed: hist}}."""
        trials = tmp_path / "trials"
        tdir = trials / "trial_00"
        tdir.mkdir(parents=True)
        (tdir / "hyperparams.json").write_text('{"model.dropout": 0.1}')
        (tdir / "history.json").write_text(json.dumps({
            "seed_histories": {
                "42": {
                    "train_loss": [1.0, 0.8],
                    "val_loss":   [1.1, 0.9],
                    "stopped_epoch": 2,
                }
            }
        }))
        df = load_trial_log(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["best_val_loss"] == pytest.approx(0.9)


class TestFullReportBuildE2E:
    """End-to-end: fake trials → build_full_report → files exist + tables correct.
    Doesn't assert figure pixel content; just that the files land."""

    def test_report_files_produced(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "mtgode"
        trials_dir = model_dir / "trials"
        _make_fake_trial(trials_dir, 0, [1.0, 0.7], {"data.input_len": 12})
        _make_fake_trial(trials_dir, 1, [1.0, 0.5], {"data.input_len": 48})
        _make_fake_trial(trials_dir, 2, [1.0, 0.6], {"data.input_len": 24})

        # No study.db; summarise_study will return {}. That's OK — the report
        # builder handles missing SQLite.
        summary = build_full_report("mtgode", hpo_root=tmp_path)
        # Study summary is empty (no SQLite) but the CSV should still be there
        report_dir = model_dir / "report"
        assert (report_dir / "trial_log.csv").exists()
        assert (report_dir / "study_summary.json").exists()
        # Figures generated from trial files, not SQLite:
        assert (report_dir / "learning_curves_all_trials.png").exists()
        assert (report_dir / "per_hyperparameter_scatter.png").exists()
        # Trial log has 3 rows, ranked
        df = pd.read_csv(report_dir / "trial_log.csv")
        assert len(df) == 3
        assert df.iloc[0]["data.input_len"] == 48
