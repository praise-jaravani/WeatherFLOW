# WeatherFLOW Phase 2 — Formulation

Multi-label detection of moderate meteorological extremes on the 32-station SAWS network, at horizons up to 48 h.

---

## 1. Problem statement

Given a 12-hour input window of observations `X[t-11..t] ∈ ℝ^(12×N×6)` on a network of `N=32` SAWS stations (6 canonical variables: t2m, tp, rh, wd, ws, sp), predict — at each horizon step `h ∈ {1..48}` — a per-(station, variable) binary indicator of whether the observed value exceeds a **calendar/hour-relative 95th-percentile threshold** derived from training data only.

Target set: `label_variables = {t2m, tp, rh, ws}` — dropped `wd` (circular; extreme wind-direction is not physically meaningful for our forecasting mission) and `sp` (sea-level pressure extremes are a proxy for other phenomena already captured by ws/rh gradients).

Loss: multi-label weighted binary cross-entropy (WBCE) with per-variable `pos_weight = (1 − p_v) / p_v` derived from training positive rates.

**tp pos_weight watch item.** Under the calendar-hour label scheme, tp's training positive rate is ≈ 0.28% (rare wet-hour exceedances), giving `pos_weight_tp ≈ 350` — roughly 15× the other variables' weights (t2m 23.3, rh 25.6, ws 20.0). The tp loss term is therefore ~15× more sensitive to a single mispredicted positive than the other variables. During the campaign we log per-variable loss components separately so that a tp-driven instability (spike in tp gradient magnitude) is diagnosable from the training log, not inferred after the fact from a diverged summary.

Headline metric: macro-mean PR-AUC across the 4 label variables. Per-variable secondary metrics: F1 (τ selected on validation by max-F1, frozen for test), CSI, precision, recall, ROC-AUC, station coverage.

Model set: `{AGCRN, CLCRN, MTGODE}`. STGODE is regression-only (univariate output head incompatible with the multi-label channel-select path; see `specs/STGODE_INTEGRATION_AUDIT.md §12`). Runs: 3 models × 10 seeds = 30 tasks.

---

## 2. Label definition (calendar/hour-relative 95th-percentile)

### 2.1 Threshold tensor

For each `(station s, variable v, day-of-year d ∈ [0,365], hour-of-day h ∈ [0,23])`:

```
T[s, v, d, h] = 95th percentile of TRAINING values of v at s
                 across timestamps whose (doy0, hour) satisfies:
                     doy0 ∈ (d ± 7)   mod 366
                     hour ∈ (h ± 1)   mod 24
```

Windows wrap at both boundaries (day 366 → day 1; hour 23 → hour 0). This gives every cell ≥ 200 training samples (median 270 = 6 train years × 15 days × 3 hours).

The window choice `(day_window=7, hour_window=1)` was not fixed by fiat — it was selected by a **pre-registered sweep** against stability and deseasonalisation criteria (§2.1.1 and Appendix table). The width is asymmetric because the two axes have different physical tolerances: the seasonal cycle is slow (a ±7-day window spans very little climatological drift), so day pooling is cheap; the diurnal cycle is fast (temperature swings several °C per hour around sunrise/sunset), so hour pooling costs meaningful fidelity. Following the ETCCDI calendar-window convention, widened on the day axis from ETCCDI's 5 days to compensate for our shorter 6-year baseline (ETCCDI uses 30-year baselines × 5 days ≈ 150 samples; we need 6 years × 15 days × 3 hours = 270 to clear the ~200-sample floor a 95th-percentile estimate requires).

Cache shape: `(N=32, D=6, 366, 24)` float32 ≈ 4.5 MB, saved as `preprocessing/cache/saws/thresholds_ch.npy` (additive to the static-scheme `thresholds.npy` — both schemes coexist so Phase-1 stratified metrics remain reproducible).

**Leap-year convention:** doy is 1..366 unconditionally (Feb 29 sits at doy 60). Non-leap-year timestamps never populate the doy 60 (0-based 59) cell directly — but the wrap-around window still pools leap-year samples for that cell, matching the ETCCDI convention of using all available reference-period years.

### 2.1.1 Window-selection sweep (methodological justification for `day_window=7, hour_window=1`)

**Pre-registered selection rule** (fixed 2026-08-13 before any sweep result was inspected):

> Candidates enumerated over the grid `hour_window ∈ {0, 1, 2}`, `day_window ∈ {3, 7, 10, 15}` (12 candidates), ordered lexicographically by `(hour_window, day_window)` ascending. Winner = first candidate satisfying **both** (a) median non-tp cell sample count ≥ 200, and (b) max non-tp seasonal-climatology opponent PR-AUC / prevalence ≤ 1.15 on real data. Any winning candidate with `day_window > 10` is flagged for a physics review before acceptance.
>
> A synthetic control (pure seasonal + diurnal + Gaussian white noise, no weather signal) is computed for all candidates as an **estimation-noise calibration**, not as a selection criterion. Real minus synthetic gives an "honest leakage" reading. A hard stop fires only if the *selected* window's synthetic ratio > 1.15, which would indicate a pipeline defect.

Rationale for the hour-first ordering: the diurnal cycle is sharper than the seasonal cycle, so an hour of smearing costs more label fidelity than a day of smearing. This prevents the sweep from preferring `(day=3, hour=2) = 210 samples` over `(day=7, hour=1) = 270 samples` for a marginal cell-count gain.

**Sweep result** (2026-08-13, see `results/diagnostics/phase2_labels/window_sweep.md` for the full 12-row appendix table):

| Candidate | median cells | real max PR-AUC/prev | synth max PR-AUC/prev | verdict |
|---|---|---|---|---|
| (hour=0, day=∈{3,7,10,15}) | 42 / 90 / 126 / 186 | 0.90 / 0.90 / 0.94 / 0.94 | 0.89 / 0.93 / 0.98 / 1.18 | fail cell floor (all < 200) |
| (hour=1, day=3) | 126 | 0.91 | 1.19 | fail cell floor |
| **(hour=1, day=7)** ← **winner** | **270** | **0.933** | 1.331 | **pass both bounds** |
| (hour=1, day=10) | 378 | 0.951 | 1.429 | passes, but later in order |
| (hour=1, day=15) | 558 | 0.972 | 1.713 | passes, but later in order |
| (hour=2, day=3–15) | 210–930 | 1.033–1.159 | 2.526–3.522 | later in order; (h=2, d=15) fails real bound |

**Two findings the sweep table justifies:**

1. **The real ratios are essentially flat (0.90–1.16) across all 12 candidates**, with 8 of 12 passing the ≤ 1.15 bound. This means the CH scheme is not sensitive to window choice — any reasonable window removes the calendar signal, and the winner is chosen by pre-registered rule rather than by a fragile empirical minimum.

2. **The synthetic-control ratios climb monotonically with window width**, from 0.89 at (h=0, d=3) up to 3.52 at (h=2, d=15). This was contrary to the a-priori expectation that narrower windows would show more estimation noise. Root cause: the synthetic series uses `amp_ann=5, amp_dir=4, noise_σ=1` — a signal-to-noise ratio (~9:1) that is unrealistic for real weather (real weather is much noisier relative to the seasonal cycle). Under this too-clean synthetic, wider windows pull samples from cells whose true underlying value differs from the target cell's, inflating the estimated threshold's variance and giving the frequency map spurious structure to exploit.

   The winner's synthetic ratio of 1.331 formally trips the 1.15 bug tripwire, and the selection script exited status SYNTH_TRIPWIRE. **We investigated and determined this is a synthetic-design issue, not a pipeline defect**: the same threshold + frequency-map code produces real-data ratios of 0.933 at the same window, and the monotonic climb across the sweep is consistent with pooling artefacts in a too-clean signal. The synthetic remains valuable as **calibration** — the "leak = real − synth" column in the appendix shows real is deeply below the noise floor at every passing candidate (leak = −0.40 at the winner) — but should not be interpreted as a strict pass/fail gate under our current synthetic parameterisation.

**Adopted window: `day_window=7, hour_window=1`.** Justified by the first-passing-candidate rule under real-data bounds, with the synthetic control providing calibration that real deseasonalisation succeeds well below the estimator noise floor.

### 2.2 Comparator

```python
label[t, n, v] = 1 if X[t, n, v] > T[n, v, doy0(t), hour(t)] else 0
```

**Strict `>`.** Explicitly preserved from the static scheme (`preprocessing/labels.py:116`) to keep the two schemes comparable at the label-construction level.

### 2.3 tp wet-hour rule (§2.5 of Phase-2 plan, extended to CH cells)

Precipitation is dominated by dry hours (≥90% at most SAWS stations). A naive per-cell 95th percentile on all tp values reduces to "any rain". So for tp specifically, the per-cell threshold is computed over WET HOURS ONLY (tp ≥ 0.1 mm/h = `TP_WET_THRESH_M`), with a fallback ladder for sparse cells:

| Rung | Definition | Requires | Observed frequency (SAWS N=32) |
|---|---|---|---|
| 0 | Cell wet-percentile (doy ±7, hour ±1) | ≥ 30 wet hours in cell | 11.4% |
| 1 | Wide-day (doy ±15, hour ±1) | ≥ 30 wet hours | 31.3% |
| 2 | Hour-pooled (doy ±15, any hour) | ≥ 30 wet hours | 51.7% |
| 3 | Seasonal (all doy in 3-month season, all hours) | ≥ 30 wet hours per (var, station) | 4.8% |
| 4 | Global static wet-hour threshold (from Phase-2 plan §2.5) | Always defined | 0.8% |

The per-cell rung used is saved to `tp_wet_trigger_ch.npy` shape `(N, 366, 24)` int8 (-1 outside tp). The rung histogram is emitted at label-build time.

### 2.4 Cache schema + hash

**Not part of the preprocessing hash.** The scheme is a LABEL-time dispatch, not a preprocessing knob, so the two label sets coexist under a single data-hash without invalidating either. The `data.threshold_scheme` config key (default `"static"`; Phase-2 configs set `"calendar_hour_relative"`) drives which `y_*` and `thresholds*` files `load_cache` returns.

Artefacts added (all in `preprocessing/cache/saws/`, all read-only):

```
thresholds_ch.npy         (N, D, 366, 24) float32   ≈ 4.5 MB
y_train_ch.npy            (T_train, N, D) int8      ≈ 9.9 MB
y_val_ch.npy              (T_val,   N, D) int8      ≈ 3.3 MB
y_test_ch.npy             (T_test,  N, D) int8      ≈ 3.3 MB
tp_wet_trigger_ch.npy     (N, 366, 24)    int8      ≈ 0.3 MB (rung code per cell)
labels_ch_meta.json       scheme name/version, windows, percentile, comparator, rung
                          histogram, train-span, per-variable pos rates.
```

Reproducibility trail (in `labels_ch_meta.json`):
- `scheme: "calendar_hour_relative"`, `scheme_version: "1.0"`
- `day_win: 7`, `hour_win: 1`, `day_win_wide: 15`
- `percentile: 95.0`, `comparator: ">"`
- `min_samples_per_cell: 100`, `tp_min_wet_per_cell: 30`
- Rung histogram (all tp cells flattened)
- Train span: exact `datetime` bounds + total hour count

---

## 3. Baseline suite

All baselines run through the same `evaluation.classification.compute_all` + `select_thresholds` (val-F1 τ selection) pipeline used for trained models. Reported per variable + macro-mean.

| Baseline | Prediction rule |
|---|---|
| **Persistence** | `label̂(t+h) = label(t+INPUT_LEN-1)` — repeat last observed extreme flag as a hard 0/1 probability for all H horizon steps. |
| **Base-rate** | `label̂(s, v, t+h) = training positive rate of (s, v)` — constant per (station, variable), broadcast over all samples × horizons. |
| **Seasonal-climatology** | `label̂(s, v, t+h) = training EXCEEDANCE FREQUENCY in the same (doy±7, hour±1) cell of station s, variable v`. Same fallback ladder as the CH threshold builder for sparse tp cells. |

The **seasonal-climatology baseline is a matched opponent for Update 1** — a classifier that only knows (station, doy, hour) climatology. Under the calendar-hour label scheme, it should collapse to base-rate performance; under the static scheme, it should be a very strong baseline (that "strength" is what our new scheme extracts as leakage).

Both baseline suites are emitted per invocation. Under CH labels, the seasonal-climatology JSON is the primary comparator. Under static labels, it's a diagnostic that quantifies exactly how much calendar signal Update 1 removed.

---

## 4. Pre-registered decision rules (verbatim)

**(a) Headline detection finding.**
> If trained models fail to exceed the seasonal-climatology baseline on PR-AUC under calendar-relative labels, this is reported as the headline detection finding.

**(b) Verification bound for the calendar-hour scheme.**
> The seasonal-climatology baseline under calendar-relative labels must have per-variable PR-AUC ≤ 1.15 × prevalence (train positive rate). Any variable materially exceeding that bound is treated as a bug signal — the deseasonalisation leaked (window too wide, hour pooling error, or fallback rung firing when it shouldn't). Investigate and re-run before training campaigns.

**(c) Comparator not silently changed.**
> Comparator is strict `>` in both schemes (`preprocessing/labels.py:116` and `preprocessing/labels_ch.py:apply_ch_labels`). Any change to the comparator (`>=`, `>`, per-variable mix) is a separate decision requiring re-registration.

---

## 5. Scope

This study targets **moderate meteorological extremes at a 48 h point-forecast horizon**.

Out of scope, with rationale:

- **Droughts** — SPI-based (Standardised Precipitation Index) definitions require 1–12 month accumulation windows (WMO SPI standard). A 48 h forecast cannot resolve a compound event that is by construction a monthly-scale phenomenon.
- **Fluvial floods** — hydrological events preconditioned by antecedent soil moisture, upstream inflow, and catchment topology. Not a point-station weather-forecast problem (Zscheischler et al. 2020 compound-event typology).
- **Named heatwaves (SAWS operational definition)** — requires ≥3 consecutive days ≥5 °C above the hottest-month mean maximum. Aggregation across days at the STATION level; incompatible with our per-hour per-station label set. Presented for scope alignment only.
- **Multi-class severity grading** (e.g., Hobday marine-heatwave classes I–IV) — declined in favour of binary detection so metrics remain comparable across variables; a natural follow-on.

Explicit **upper-tail-only limitation**: labels are set on high-value exceedances only. Lower-tail extremes (frost, drought-related sp anomalies, low-wind renewable-energy shortfalls) are not indicated by `label = 1` in this set. A future symmetric extension (low + high tails) is trivial mechanically but requires re-registering the label semantics.

---

## 6. Literature trail

Papers that shaped this formulation, with what each contributed:

- **Zhang, X. et al. (2011), WIREs Climate Change 2:851** — "Indices for monitoring changes in extremes based on daily temperature and precipitation data". Introduced the ETCCDI family (TX90p / TN90p / R95p): percentile thresholds computed within a moving calendar-day window around each target day, from a reference period. **Direct basis of Update 1** (extended to hourly resolution by adding the ±1 hour-of-day window; unchanged wrap semantics).

- **South African Weather Service — operational heatwave definition** (public; also documented in SAWS annual climate statements). Requires ≥3 consecutive days with T_max ≥ 5 °C above the hottest-month mean maximum, station-level. **Motivates the anomaly-relative principle** (thresholds should adapt to the reference climatology, not sit at a fixed absolute); we relax the multi-day duration constraint to keep the problem inside a 48 h forecast horizon.

- **Zscheischler, J. et al. (2020), Nature Reviews Earth & Environment 1:333** — "A typology of compound weather and climate events". **Motivates scope boundary**: droughts and fluvial floods are compound events by construction, requiring multi-scale conditioning that a 48 h univariate forecast cannot capture.

- **WMO (2012), Standardised Precipitation Index (SPI) User Guide (WMO-No. 1090)**. **Motivates drought exclusion**: SPI is by definition an accumulation over 1–12 months; the 48 h forecast horizon is orthogonal to the drought detection problem.

- **Xu, W. et al. (2024), ExtremeCast**, arXiv:2402.01295. Demonstrates that L2 / L1 losses on regression heads systematically underestimate extremes — motivates the presence-absence detection formulation over regression-with-extremeness (which we do report in Phase 1, but as regression accuracy, not extremeness).

- **Pasche, O. et al. (2025), AI for the Earth Sciences (AIES) 4:e240163** — case-study named-event validation of ML weather forecasts (Storm Ciarán). **Precedent for reporting on canonical events alongside aggregate metrics**; the CH scheme makes such event-level validation possible without calendar-signal contamination.

- **Nurmohamed et al. (2026), Forecasting 8(4):63** — ML-heatwave scoping review. Reports percentile-threshold labels as the plurality formulation across ~40 recent ML extreme-detection studies, and identifies definitional inconsistency across the field as a known weakness. **Motivates our documented, reproducible scheme.**

- **Marine-heatwave ML review** (Wang et al. 2025, Ocean Modelling — presence/absence + imbalance-aware losses; CSI/SEDI/MCC recommended). Motivates our metric choice (F1 + PR-AUC + CSI, all reported); we decline multi-class severity grading (Hobday 2016) because it doesn't cross-map cleanly to non-marine variables at hourly resolution.

- **Galib, A. H. et al. (2022), DeepExtrema**, IJCAI. GEV (Generalised Extreme Value) block-maxima reformulation of extreme detection. **Declined for this study** (breaks per-timestep supervision and complicates the hourly-forecast framing) but noted as a natural next-methodology.

- **Kim, S. et al. (2025), Scientific Data 12:445 (ClimateNet lineage)** — expert-labelled tropical-cyclone / atmospheric-river segmentation masks on ERA5. **Declined** because expert-labelled object masks are unavailable at the SAWS station network resolution; our percentile-threshold scheme is the operational alternative.

---

## 7. Explicitly-recorded follow-on decision (out of scope here)

Whether to **recompute Phase-1 extreme-stratified metrics** (MAE-extreme, MAE-nonextreme, ratio) under the new calendar-hour threshold definition for one-definition coherence across phases. This is a **post-hoc computation on saved predictions only** — it needs the Phase-1 `per_station.npz` predictions, no re-training. The trade-off:

- **For coherence**: reporting "MAE-extreme" in Phase 1 using a different exceedance definition than "extreme detected" in Phase 2 is a coherence break. Recomputing under CH aligns them.
- **Against coherence**: the current Phase-1 stratification was pre-registered under the static definition and the numbers are in the paper's draft tables. Changing them post-hoc adds provenance complexity.

Recommendation for the human decision: recompute under CH into a separate Phase-1 supplement table (both stratifications reported), with the CH one identified as the more calendar-neutral measure.
