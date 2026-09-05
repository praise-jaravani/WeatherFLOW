"""
Find candidate extreme events in the test partition for the Section 4.5.3 case study.

Data-first selection: rather than picking an event from news reports and hoping the
network captured it, this scans the calendar-relative exceedance labels already
computed by the pipeline and ranks episodes by how many stations were simultaneously
in exceedance and for how long. You then match the winner to the SAWS record for a name.

Also runs a separate LOWER-tail scan on t2m, because the Phase-2 labels are upper-tail
only and cannot represent a cold event (e.g. the 19-22 Sep 2024 snow event). A cold
event can still serve as a Phase-1-only case study.

Usage:
    python scripts/find_case_study_events.py                 # top 10 per variable
    python scripts/find_case_study_events.py --min-stations 6 --top 15
    python scripts/find_case_study_events.py --probe         # just show cache keys and exit

Outputs results/CASE_STUDY_candidates.csv and prints a ranked table.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.pipeline import load_cache

# Record origin and split sizes from Section 3.1. Verified in the script below.
RECORD_START = np.datetime64("2016-01-01T00:00")
N_TRAIN_HOURS, N_VAL_HOURS, N_TEST_HOURS = 52603, 17534, 17535
LABEL_VARS = ["t2m", "tp", "rh", "ws"]


# ── cache access ─────────────────────────────────────────────────────────────

def probe(cached):
    """Print every key with its shape so the label array can be identified."""
    print("\nCACHE CONTENTS")
    for k, v in sorted(cached.items()):
        if isinstance(v, np.ndarray):
            print(f"  {k:28s} ndarray {str(v.shape):22s} {v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k:28s} {type(v).__name__}[{len(v)}]  first={v[0] if len(v) else None!r}")
        else:
            print(f"  {k:28s} {type(v).__name__}  {str(v)[:60]}")
    print()


def get_test_labels(cached):
    """
    Return (T_test, N, n_label_vars) uint8 exceedance labels for the test split.

    The key name is not assumed. If none of the candidates is present, --probe
    will show what is actually in the cache.
    """
    for key in ("labels_test", "y_labels_test", "labels_ch_test", "exceed_test", "y_test_ch"):
        if key in cached:
            print(f"  using labels key: '{key}'")
            return np.asarray(cached[key])

    # Fall back to a full-record label array sliced to the test span.
    for key in ("labels", "labels_ch", "exceedance_labels"):
        if key in cached:
            arr = np.asarray(cached[key])
            if arr.shape[0] == N_TRAIN_HOURS + N_VAL_HOURS + N_TEST_HOURS:
                print(f"  using labels key: '{key}' (sliced to test span)")
                return arr[N_TRAIN_HOURS + N_VAL_HOURS:]

    raise KeyError(
        "No label array found. Run with --probe and add the correct key to "
        "get_test_labels(). Expected shape (T_test, N, 4) matching "
        f"({N_TEST_HOURS}, n_stations, {len(LABEL_VARS)})."
    )


def get_station_names(cached, n_stations):
    for key in ("station_names", "stations", "node_names", "station_ids"):
        if key in cached:
            names = list(cached[key])
            if len(names) == n_stations:
                return names
    print("  WARNING: no station names in cache; using indices")
    return [f"station_{i:02d}" for i in range(n_stations)]


def test_timestamps(n_hours):
    """Hourly timestamps for the test split, derived from the record origin."""
    offset = N_TRAIN_HOURS + N_VAL_HOURS
    ts = RECORD_START + np.arange(offset, offset + n_hours).astype("timedelta64[h]")
    return ts


# ── episode detection ────────────────────────────────────────────────────────

def find_episodes(count_series, timestamps, min_stations, min_hours=3, max_gap=6):
    """
    Group hours whose simultaneous-exceedance count meets min_stations into episodes,
    bridging gaps of up to max_gap hours (a heatwave dips overnight but is one event).
    """
    hot = count_series >= min_stations
    if not hot.any():
        return []

    idx = np.flatnonzero(hot)
    episodes, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap:
            episodes.append((start, prev))
            start = i
        prev = i
    episodes.append((start, prev))

    out = []
    for a, b in episodes:
        duration = b - a + 1
        if duration < min_hours:
            continue
        seg = count_series[a:b + 1]
        out.append(dict(
            start=timestamps[a], end=timestamps[b],
            duration_h=int(duration),
            peak_stations=int(seg.max()),
            mean_stations=float(seg.mean()),
            station_hours=int(seg.sum()),      # ranking statistic
            start_idx=int(a), end_idx=int(b),
        ))
    return out


def lower_tail_scan(x_test, var_idx, timestamps, station_names, pct=5.0, min_stations=6):
    """
    Upper-tail labels cannot represent a cold event. This finds hours where many
    stations fall below their own test-period 5th percentile, for a Phase-1-only
    case study (e.g. the September 2024 cold and snow event).
    """
    v = x_test[..., var_idx]                              # (T, N)
    thr = np.nanpercentile(v, pct, axis=0)                # per station
    below = (v < thr[None, :]).sum(axis=1)
    return find_episodes(below, timestamps, min_stations)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-stations", type=int, default=5,
                    help="simultaneous stations in exceedance to count as an event hour")
    ap.add_argument("--min-hours", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=6,
                    help="hours of dip tolerated within one episode")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--probe", action="store_true", help="print cache keys and exit")
    args = ap.parse_args()

    cached = load_cache("saws")
    if args.probe:
        probe(cached)
        return

    labels = get_test_labels(cached)
    T, N, V = labels.shape
    print(f"labels: {labels.shape}  (expected T={N_TEST_HOURS}, V={len(LABEL_VARS)})")
    if T != N_TEST_HOURS:
        print(f"  WARNING: T={T} != {N_TEST_HOURS}; timestamps may be misaligned")

    # resolve label-variable indices from the cache var_names (avoids wd/ws confusion)
    cache_var_names = list(cached.get("meta", {}).get("var_names", [])
                           or cached.get("var_names", []))
    if cache_var_names and V == len(cache_var_names):
        label_indices = []
        for lv in LABEL_VARS:
            if lv in cache_var_names:
                label_indices.append(cache_var_names.index(lv))
            else:
                raise KeyError(f"Label variable '{lv}' not found in cache var_names: {cache_var_names}")
        print(f"  resolved label indices from var_names {cache_var_names}: {dict(zip(LABEL_VARS, label_indices))}")
    else:
        label_indices = list(range(len(LABEL_VARS)))
        print(f"  WARNING: using default indices {label_indices} (var_names not matched)")

    ts = test_timestamps(T)
    names = get_station_names(cached, N)
    print(f"test span: {ts[0]} to {ts[-1]}")

    # sanity: recomputed positive rates should match PHASE2_labels_ch_meta.json
    print("\nrecomputed test positive rates (compare to PHASE2_labels_ch_meta.json):")
    for j, v in zip(label_indices, LABEL_VARS):
        print(f"  {v:4s} {labels[..., j].mean():.5f}")

    rows = []
    for j, var in zip(label_indices, LABEL_VARS):
        counts = labels[..., j].sum(axis=1)               # stations in exceedance per hour
        eps = find_episodes(counts, ts, args.min_stations, args.min_hours, args.max_gap)
        eps.sort(key=lambda e: e["station_hours"], reverse=True)
        print(f"\n{'='*78}\n{var}  —  {len(eps)} episodes at >= {args.min_stations} "
              f"simultaneous stations\n{'='*78}")
        for e in eps[:args.top]:
            top = labels[e["start_idx"]:e["end_idx"] + 1, :, j].sum(axis=0)
            worst = [names[i] for i in np.argsort(-top)[:5] if top[i] > 0]
            print(f"  {str(e['start'])[:16]} -> {str(e['end'])[:16]}  "
                  f"{e['duration_h']:4d} h  peak {e['peak_stations']:2d} stn  "
                  f"{e['station_hours']:5d} stn-h")
            print(f"      most affected: {', '.join(worst)}")
            rows.append(dict(variable=var, tail="upper", **e,
                             top_stations="|".join(worst)))

    # lower tail on t2m, for a Phase-1-only cold-event case study
    if "X_test" in cached:
        var_names = list(cached.get("var_names", [])
                         or cached.get("meta", {}).get("var_names", []))
        if "t2m" in var_names:
            print(f"\n{'='*78}\nt2m LOWER tail (below per-station 5th percentile) "
                  f"— Phase 1 only\n{'='*78}")
            eps = lower_tail_scan(np.asarray(cached["X_test"]), var_names.index("t2m"),
                                  ts, names, min_stations=args.min_stations)
            eps.sort(key=lambda e: e["station_hours"], reverse=True)
            for e in eps[:args.top]:
                print(f"  {str(e['start'])[:16]} -> {str(e['end'])[:16]}  "
                      f"{e['duration_h']:4d} h  peak {e['peak_stations']:2d} stn  "
                      f"{e['station_hours']:5d} stn-h")
                rows.append(dict(variable="t2m", tail="lower", **e, top_stations=""))

    out = ROOT / "results" / "CASE_STUDY_candidates.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nNext: pick one episode, confirm it against the SAWS record for a name, "
          "and pass its start/end to the plotting script.")


if __name__ == "__main__":
    main()
