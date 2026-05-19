# Preprocessing Pipeline Design

## Pipeline order (leakage-safe)

```
1. Parse     → (T, N, d) array + coords DataFrame
2. Impute    → fill NaN via IDW (ERA5 has none; needed for SAWS)
3. Split     → chronological 60/20/20 ← MUST come before steps 4 and 5
4. Label     → 95th-percentile thresholds (computed on train only)
5. Normalise → z-score (mean/std computed on train only)
6. Graph     → adjacency matrix A of shape (N, N)
7. DataLoaders → sliding-window Dataset + DataLoader
```

**Why Split is step 3, not later:** if you label or normalise before splitting, the threshold and z-score stats see validation and test data — data leakage. Stats must be computed from training partition only.

## Output shapes

| Array | Shape | Notes |
|---|---|---|
| X (raw) | (T, N, d) | float32 |
| X_train/val/test | (T_split, N, d) | float32, normalised |
| y_train/val/test | (T_split, N, d) | float32 binary labels (int8 internally) |
| adj | (N, N) | float32, row-normalised |
| mean, std | (1, N, d) | broadcast-ready |
| thresholds | (N, d) | one per node per variable |

**ERA5 at stride=2:** T≈122,640 (14 years hourly), N=999, d=6.  
**DataLoader batches:** x=(B, input_len, N, d), y=(B, horizon, N, d) for regression; y=(B, horizon, N) for classification.

## Adjacency matrix

Built in `preprocessing/graph.py` using **Gaussian kernel on haversine distance**:

```
w_ij = exp(-d_ij² / 2σ²)   if d_ij < threshold
     = 0                     otherwise
```

Defaults: threshold=300 km, σ=100 km. Rows normalised to sum 1. No self-loops.

This static A is passed to every model. Models may additionally learn dynamic adjacency internally (AGCRN does this) — that is separate from this static graph.

## Cache

`preprocessing/cache/<dataset>/` holds `.npy` files for all arrays. `preprocessing/cache/preprocessing_hash.json` holds SHA-256 fingerprints of raw files + config. Cache is valid as long as the hash matches — pipeline skips all steps on a hit.

Force rerun: `python cli.py ... --no-cache`

## Node strategy (ERA5 vs SAWS)

- **ERA5 standalone:** use grid points directly as nodes (stride configurable)
- **ERA5 + SAWS joined:** interpolate ERA5 to SAWS station coordinates via bilinear interpolation so both datasets share the same N nodes

SAWS mode not yet active — waiting for data. Configured via `preprocessing.lat_stride` / `preprocessing.lon_stride` in YAML.

## Implementation status (2026-05-19)

All modules implemented and smoke-tested against 16 months of ERA5 data:

| Module | Status |
|---|---|
| `era5_parser.py` | ✓ |
| `graph.py` | ✓ |
| `split.py` | ✓ |
| `normalise.py` | ✓ |
| `labels.py` | ✓ |
| `impute.py` | ✓ |
| `dataset.py` | ✓ |
| `pipeline.py` | ✓ |
| `saws_parser.py` | stub — waiting for data |
