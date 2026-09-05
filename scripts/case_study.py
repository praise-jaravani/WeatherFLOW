"""
Section 4.5.3 case study: the December 2024 heatwave.

Two stages, so the expensive part runs once on HPC and plotting iterates anywhere.

    # on HPC (needs checkpoints + cache)
    python scripts/case_study.py extract

    # anywhere (needs only the .npz written above)
    python scripts/case_study.py plot

FORECAST ALIGNMENT
------------------
Sample i takes input hours [i, i+w-1] and predicts hours [i+w, i+w+H-1].
Horizon step h (1-indexed) therefore targets absolute hour  i + w - 1 + h.
To plot a fixed-lead series over target hours [T0, T1] we need samples
    i = T - (w - 1) - h        for T in [T0, T1]
so extraction must reach back LEAD_MAX + w - 1 hours before the event start.
Plotting X_test[T0:T1] against predictions indexed by T would be wrong.
"""

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# sys.modules 'model' collision guard — same pattern as run_worker.py.
# AGCRN, CLCRN, MTGODE, STGODE each vendor a bare 'model' package; iterating
# through them in one process leaves stale entries that break later imports.
_INITIAL_SYS_PATH = list(sys.path)

def _purge_model_module() -> None:
    for key in list(sys.modules.keys()):
        if key == "model" or key.startswith("model."):
            del sys.modules[key]
    sys.path[:] = list(_INITIAL_SYS_PATH)

# ── event: December 2024 heatwave ────────────────────────────────────────────
EVENT = "dec2024_heatwave"
T0, T1 = 8208, 8336              # inclusive test-partition indices (2024-12-08T09 → 2024-12-13T17)
# Primary focal station: Makatini Research Centre (node 16) — network-highest
# observed t2m (41.2 °C), onset visible at hour 53, 53% exceedance rate, zero
# exceedance in first 6 hours. Bethlehem WO (node 2) was the original pick but
# was rejected: event already underway at hour 5, 81% exceedance rate, and
# positive-class imbalance makes single-station cls metrics uninterpretable.
STATIONS = {16: "Makatini Research Centre", 10: "Ficksburg", 27: "Royal Natal NP"}
W, H = 12, 48
LEADS = [6, 12, 24, 48]          # fixed-lead series to extract
LEAD_MAIN = 24                   # lead shown in Panels A and B

# median-MAE (regression) / median-PR-AUC (classification) seeds over all completed seeds.
# Computed from CAMPAIGN_all_runs.csv, 2026-08-31.
# Regression: median of 10 seeds (9 for CLCRN t2m — s50 diverged, excluded).
# Classification: median of 10 seeds by macro PR-AUC (rank 5 of 10 chosen).
# AGCRN cls s50 rank 5, CLCRN cls s47 rank 5, MTGODE cls s51 rank 6.
REG_SEED = {"agcrn": 50, "clcrn": 49, "mtgode": 44, "stgode": 45}
CLS_SEED = {"agcrn": 50, "clcrn": 47, "mtgode": 51}

OUT = ROOT / "results" / f"CASE_{EVENT}.npz"


def _get_adapter(model_name: str):
    if model_name == "agcrn":
        from models.baselines.agcrn_adapter import AGCRNAdapter
        return AGCRNAdapter
    if model_name == "clcrn":
        from models.baselines.clcrn_adapter import CLCRNAdapter
        return CLCRNAdapter
    if model_name == "mtgode":
        from models.continuous_time.mtgode_adapter import MTGODEAdapter
        return MTGODEAdapter
    if model_name == "stgode":
        from models.continuous_time.stgode_adapter import STGODEAdapter
        return STGODEAdapter
    raise KeyError(f"unknown model: {model_name!r}")


# ── extraction (HPC) ─────────────────────────────────────────────────────────

def extract():
    import torch, yaml
    from preprocessing.pipeline import load_cache

    cached = load_cache("saws")
    X_test = np.asarray(cached["X_test"])                 # (T, N, D)
    y_ch   = np.asarray(cached["y_test_ch"])              # (T, N, D) exceedance labels
    mean, std = np.asarray(cached["mean"]), np.asarray(cached["std"])
    adj = np.asarray(cached["adj"])
    var_names = list(cached.get("var_names") or cached["meta"]["var_names"])
    t2m = var_names.index("t2m")
    print(f"var_names={var_names}  t2m index={t2m}")

    lead_max = max(LEADS)
    i_lo = T0 - (W - 1) - lead_max
    i_hi = T1 - (W - 1) - min(LEADS)
    assert i_lo >= 0, "event too close to the start of the test split"
    print(f"samples needed: {i_lo}..{i_hi}  ({i_hi - i_lo + 1} windows)")

    # observed truth and labels over the event, in physical units
    obs = X_test[T0:T1 + 1, :, t2m] * std[0, :, t2m] + mean[0, :, t2m]
    lab = y_ch[T0:T1 + 1, :, t2m].astype(np.uint8)

    store = {"obs_t2m": obs, "label_t2m": lab,
             "T0": T0, "T1": T1, "leads": np.array(LEADS), "lead_main": LEAD_MAIN,
             "station_ids": np.array(list(STATIONS)),
             "station_names": np.array(list(STATIONS.values()))}

    def windows(lo, hi):
        """Stack input windows for sample indices lo..hi inclusive."""
        return np.stack([X_test[i:i + W] for i in range(lo, hi + 1)])   # (S, W, N, D)

    Xw = torch.tensor(windows(i_lo, i_hi), dtype=torch.float32)
    adj_t = torch.tensor(adj, dtype=torch.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj_t = adj_t.to(dev)

    def run(model, task, seed):
        cfg = yaml.safe_load(open(ROOT / f"experiments/configs/{model}.yaml"))
        cfg.setdefault("data", {})["task"] = task
        # build_model needs config["meta"] — reproduce the same dict the training worker uses
        run_meta = dict(cached["meta"])
        run_meta.update({
            "coords":            cached["coords"],
            "mean":              cached["mean"],
            "std":               cached["std"],
            "thresholds":        cached["thresholds"],
            "static_thresholds": cached["thresholds"],
        })
        cfg["meta"] = run_meta
        target = "t2m" if task == "regression" else "cls"
        ckpt = ROOT / "runs" / f"{model}_{target}_s{seed}" / "checkpoint.pt"
        if not ckpt.exists():
            print(f"  MISSING {ckpt}"); return None
        _purge_model_module()
        adapter = _get_adapter(model)()
        adapter.build_model(cfg, adj)
        state = torch.load(ckpt, map_location=dev, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        adapter.model.load_state_dict(state); adapter.model.to(dev).eval()
        outs = []
        with torch.no_grad():
            for k in range(0, len(Xw), 64):
                outs.append(adapter.predict(Xw[k:k + 64].to(dev), adj_t).cpu().numpy())
        return np.concatenate(outs, 0)                    # (S, H, N, d)

    for m, seed in REG_SEED.items():
        p = run(m, "regression", seed)
        if p is None: continue
        if p.shape[-1] > 1: p = p[..., t2m:t2m + 1]
        p = p[..., 0] * std[0, :, t2m] + mean[0, :, t2m]  # (S, H, N) physical
        for lead in LEADS:
            # target hour T <- sample (T - (W-1) - lead), horizon step lead
            idx = np.arange(T0, T1 + 1) - (W - 1) - lead - i_lo
            store[f"reg_{m}_lead{lead}"] = p[idx, lead - 1, :]
        print(f"  {m} regression seed {seed}: ok")

    for m, seed in CLS_SEED.items():
        p = run(m, "classification", seed)
        if p is None: continue
        prob = 1.0 / (1.0 + np.exp(-p[..., 0]))           # t2m is label index 0
        for lead in LEADS:
            idx = np.arange(T0, T1 + 1) - (W - 1) - lead - i_lo
            store[f"cls_{m}_lead{lead}"] = prob[idx, lead - 1, :]
        tau_summary = json.load(open(ROOT / "runs" / f"{m}_cls_s{seed}" / "summary.json"))
        lv = tau_summary["test_metrics"].get("label_variables", ["t2m", "tp", "rh", "ws"])
        store[f"tau_{m}"] = float(tau_summary["test_metrics"]["taus"][lv.index("t2m")])
        print(f"  {m} classification seed {seed}: ok, tau={store[f'tau_{m}']:.3f}")

    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **store)
    print(f"wrote {OUT}")


# ── plotting (anywhere) ──────────────────────────────────────────────────────

def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.labelsize": 8, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})

    d = np.load(OUT, allow_pickle=False)
    lead = int(d["lead_main"])
    hours_dt64 = np.datetime64("2016-01-01T00:00") + (
        np.arange(int(d["T0"]), int(d["T1"]) + 1) + 52603 + 17534
    ).astype("timedelta64[h]")
    # .tolist() on datetime64[ms] yields Python datetime objects reliably;
    # .astype("O") does not and caused the MAXTICKS warning in the original plot.
    hours = hours_dt64.astype("datetime64[ms]").tolist()
    sid = list(d["station_ids"]); snames = list(d["station_names"])
    node = sid[0]; name = snames[0]                      # primary station
    n = sid.index(node)

    COL = {"agcrn": "#2E7D32", "clcrn": "#1565C0",
           "mtgode": "#EF6C00", "stgode": "#C62828"}

    fig, ax = plt.subplots(3, 1, figsize=(3.33, 5.4), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1.5, 1.2], "hspace": 0.18})

    # Panel A — observed vs predicted (Gaibie device)
    lab = d["label_t2m"][:, node].astype(bool)
    ax[0].fill_between(hours, 0, 1, where=lab, transform=ax[0].get_xaxis_transform(),
                       color="0.88", zorder=0, label="exceedance hours")
    ax[0].plot(hours, d["obs_t2m"][:, node], color="k", lw=1.4, label="observed")
    for m in COL:
        k = f"reg_{m}_lead{lead}"
        if k in d: ax[0].plot(hours, d[k][:, node], color=COL[m], lw=0.9, label=m.upper())
    ax[0].set_ylabel("t2m (°C)")
    ax[0].legend(ncol=2, frameon=False, loc="upper left")
    ax[0].set_title(f"{name} — {lead} h lead", fontsize=8, loc="left")

    # Panel B — exceedance probability vs frozen threshold
    ax[1].fill_between(hours, 0, 1, where=lab, transform=ax[1].get_xaxis_transform(),
                       color="0.88", zorder=0)
    for m in ("agcrn", "clcrn", "mtgode"):
        k = f"cls_{m}_lead{lead}"
        if k in d:
            ax[1].plot(hours, d[k][:, node], color=COL[m], lw=0.9)
            ax[1].axhline(float(d[f"tau_{m}"]), color=COL[m], ls=":", lw=0.7)
    ax[1].set_ylabel("P(exceedance)"); ax[1].set_ylim(0, 1)

    # Panel C — MAE against lead time, over the event (Pasche device)
    leads = [int(x) for x in d["leads"]]
    for m in COL:
        errs = [np.abs(d[f"reg_{m}_lead{L}"][:, node] - d["obs_t2m"][:, node]).mean()
                for L in leads if f"reg_{m}_lead{L}" in d]
        if errs: ax[2].plot(leads[:len(errs)], errs, "o-", color=COL[m], ms=3, lw=0.9)
    ax[2].set_xlabel("lead time (h)"); ax[2].set_ylabel("event MAE (°C)")
    ax[2].set_xticks(leads)

    for a in ax[:2]:
        a.xaxis.set_major_locator(mdates.DayLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate(rotation=0, ha="center")

    out = ROOT / "figures" / "fig7_case_study.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "plot"])
    a = ap.parse_args()
    extract() if a.stage == "extract" else plot()
