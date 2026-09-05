#!/usr/bin/env python3
"""Rebuild results_upload/ from local runs/ + manifest_all.json.

Regenerates:
  - CAMPAIGN_status.json  (snapshot: completed/pending/diverged lists, per-cell counts)
  - CAMPAIGN_all_runs.csv (one row per completed task, 79 cols)
  - MODEL_{agcrn,clcrn,mtgode,stgode}.json  (nested per-cell seed dumps)
  - 00_INDEX_results.md    (updated completion stats)

Idempotent. Baselines / HPO / FIGURE / PHASE2 files untouched.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/praise/Computer_Science_Resources/Weatherflow")
RUNS = REPO / "runs"
MANIFEST = RUNS / "manifest_all.json"
OUT = REPO / "results_upload"

CSV_REG_METRICS = ("mae", "rmse", "mse", "smape", "mae_extreme", "mae_nonextreme", "vpt")
CSV_CLS_METRICS = ("f1", "pr_auc", "precision", "recall", "csi", "auc_roc", "accuracy", "tau")
CSV_CLS_MACRO   = ("f1", "pr_auc", "precision", "recall", "csi", "auc_roc")
REG_TARGETS = ("t2m", "sp", "rh", "ws")
CLS_LABELS  = ("t2m", "tp", "rh", "ws")


def csv_header() -> list[str]:
    cols = ["task_id","model","target","seed","task_type","stopped_epoch","loss_type",
            "n_train","n_val","n_test","n_params",
            "val_loss_last","val_loss_min","completed_ts_utc"]
    for v in REG_TARGETS:
        for m in CSV_REG_METRICS:
            cols.append(f"reg_{v}_{m}")
    for v in CLS_LABELS:
        for m in CSV_CLS_METRICS:
            cols.append(f"cls_{v}_{m}")
    for m in CSV_CLS_MACRO:
        cols.append(f"cls_macro_{m}")
    return cols


def _get_reg_metric(tm, var, metric):
    d = tm.get(metric) or {}
    return d.get(var)


def _get_cls_metric(tm, var, metric):
    pv = (tm.get("per_variable") or {}).get(var, {}) or {}
    return pv.get(metric)


def _get_macro(tm, metric):
    m = tm.get("macro", {}) or {}
    return m.get(metric)


def summary_to_csv_row(summary: dict) -> list:
    row = {}
    task_id = summary.get("task_id", "?")
    parts = task_id.split("_")
    model = parts[0]
    target = parts[1]
    seed = int(parts[2].lstrip("s"))
    task_type = "classification" if target == "cls" else "regression"

    row["task_id"] = task_id
    row["model"] = model
    row["target"] = target
    row["seed"] = seed
    row["task_type"] = task_type
    row["stopped_epoch"] = summary.get("stopped_epoch", "")
    row["loss_type"] = summary.get("loss_type", "")
    row["n_train"] = summary.get("n_train_samples", "")
    row["n_val"]   = summary.get("n_val_samples", "")
    row["n_test"]  = summary.get("n_test_samples", "")
    row["n_params"] = summary.get("n_params", "")

    val_loss = summary.get("val_loss") or []
    row["val_loss_last"] = val_loss[-1] if val_loss else ""
    row["val_loss_min"]  = min(val_loss) if val_loss else ""

    ts = summary.get("completed_ts", "")
    if isinstance(ts, (int, float)):
        row["completed_ts_utc"] = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        row["completed_ts_utc"] = ""

    tm = summary.get("test_metrics", {}) or {}
    if task_type == "regression":
        for v in REG_TARGETS:
            for m in CSV_REG_METRICS:
                row[f"reg_{v}_{m}"] = _get_reg_metric(tm, v, m)
    else:
        for v in CLS_LABELS:
            for m in CSV_CLS_METRICS:
                row[f"cls_{v}_{m}"] = _get_cls_metric(tm, v, m)
        for m in CSV_CLS_MACRO:
            row[f"cls_macro_{m}"] = _get_macro(tm, m)

    # produce full column set in header order
    header = csv_header()
    return [row.get(c, "") for c in header]


def is_diverged(rundir: Path) -> bool:
    marker = rundir / "FAILED_DIVERGED"
    if marker.exists():
        return True
    log = rundir / "train.log"
    if log.exists():
        try:
            with open(log) as f:
                for line in f:
                    if "DIVERGED" in line:
                        return True
        except Exception:
            pass
    return False


def main():
    manifest = json.load(open(MANIFEST))
    all_tasks = list(manifest)

    completed, diverged, pending = [], [], []
    summaries = {}
    for tid in all_tasks:
        rd = RUNS / tid
        sp = rd / "summary.json"
        if sp.exists():
            try:
                summaries[tid] = json.load(open(sp))
                completed.append(tid)
            except Exception:
                pending.append(tid)
        elif rd.exists() and is_diverged(rd):
            diverged.append(tid)
        else:
            pending.append(tid)

    print(f"[rebuild] {len(all_tasks)} manifest / {len(completed)} completed / "
          f"{len(diverged)} diverged / {len(pending)} pending")

    # ---- CAMPAIGN_status.json ----
    counts_by_mt = {}
    for tid in all_tasks:
        parts = tid.split("_")
        key = f"{parts[0]}_{parts[1]}"
        counts_by_mt.setdefault(key, {"completed": 0, "diverged": 0, "pending": 0})
    for tid in completed: counts_by_mt[f"{tid.split('_')[0]}_{tid.split('_')[1]}"]["completed"] += 1
    for tid in diverged:  counts_by_mt[f"{tid.split('_')[0]}_{tid.split('_')[1]}"]["diverged"]  += 1
    for tid in pending:   counts_by_mt[f"{tid.split('_')[0]}_{tid.split('_')[1]}"]["pending"]   += 1

    reg_c = sum(1 for t in completed if "_cls_" not in t)
    reg_d = sum(1 for t in diverged  if "_cls_" not in t)
    reg_p = sum(1 for t in pending   if "_cls_" not in t)
    cls_c = sum(1 for t in completed if "_cls_" in t)
    cls_d = sum(1 for t in diverged  if "_cls_" in t)
    cls_p = sum(1 for t in pending   if "_cls_" in t)

    status = {
        "snapshot_time_utc": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_tasks": len(all_tasks),
        "completed": sorted(completed),
        "diverged":  sorted(diverged),
        "pending":   sorted(pending),
        "counts_by_model_target": counts_by_mt,
        "counts_by_type": {
            "regression":     {"completed": reg_c, "diverged": reg_d, "pending": reg_p},
            "classification": {"completed": cls_c, "diverged": cls_d, "pending": cls_p},
        },
    }
    with open(OUT / "CAMPAIGN_status.json", "w") as f:
        json.dump(status, f, indent=2)
    print(f"[rebuild] wrote CAMPAIGN_status.json ({len(completed)} completed / {len(pending)} pending)")

    # ---- CAMPAIGN_all_runs.csv ----
    header = csv_header()
    with open(OUT / "CAMPAIGN_all_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for tid in sorted(completed):
            w.writerow(summary_to_csv_row(summaries[tid]))
    print(f"[rebuild] wrote CAMPAIGN_all_runs.csv ({len(completed)} rows, {len(header)} cols)")

    # ---- MODEL_{model}.json ----
    for model in ("agcrn", "clcrn", "mtgode", "stgode"):
        cells = {}
        for tid in all_tasks:
            parts = tid.split("_")
            if parts[0] != model: continue
            cell = parts[1]
            cells.setdefault(cell, {"n_seeds_completed": 0, "n_seeds_diverged": 0,
                                    "diverged_seeds": [], "seeds": []})
            if tid in summaries:
                cells[cell]["seeds"].append(summaries[tid])
                cells[cell]["n_seeds_completed"] += 1
            elif tid in diverged:
                seed = int(parts[2].lstrip("s"))
                cells[cell]["diverged_seeds"].append(seed)
                cells[cell]["n_seeds_diverged"] += 1
        for cell in cells:
            cells[cell]["seeds"].sort(key=lambda s: s.get("seed", 0))
            cells[cell]["diverged_seeds"].sort()
        # order cell keys: cls last if present, reg targets in default order
        ordered = {}
        for k in ("cls",):
            if k in cells: ordered[k] = cells.pop(k)
        for k in ("t2m", "sp", "rh", "ws"):
            if k in cells: ordered[k] = cells.pop(k)
        # anything left (unexpected)
        for k in sorted(cells):
            ordered[k] = cells[k]
        # write cls FIRST so old readers work; per_target after
        final = {"model": model, "cells": dict((k, ordered[k]) for k in ordered)}
        with open(OUT / f"MODEL_{model}.json", "w") as f:
            json.dump(final, f, indent=2)
        n_done = sum(c["n_seeds_completed"] for c in final["cells"].values())
        n_div  = sum(c["n_seeds_diverged"] for c in final["cells"].values())
        print(f"[rebuild] wrote MODEL_{model}.json ({n_done} completed, {n_div} diverged)")

    return status


if __name__ == "__main__":
    main()
