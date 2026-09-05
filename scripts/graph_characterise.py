"""
Adjacency-matrix characterisation artifact (Adjacency_Audit_Spec.md §5.3).

Produces the numbers + figure needed by any methods-section paragraph about
the static A: per-node degree distribution, weight-vs-distance curve,
threshold placement, sparsity summary. Runs locally in seconds.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display available
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir("/home/praise/Computer_Science_Resources/Weatherflow")
sys.path.insert(0, "/home/praise/Computer_Science_Resources/Weatherflow")
from preprocessing.graph import build_distance_matrix, build_adjacency_matrix

OUT = Path("outputs/adjacency_probe")
OUT.mkdir(parents=True, exist_ok=True)


# Load the SAWS station coordinates
coords = pd.read_csv("data/saws/station_coordinates.csv")
coords = coords[["station", "latitude", "longitude"]].rename(
    columns={"latitude": "lat", "longitude": "lon"}
)
coords["node_id"] = range(len(coords))
N = len(coords)

# SAWS defaults
DIST_THRESH = 500.0
SIGMA = 200.0

D = build_distance_matrix(coords)
A = build_adjacency_matrix(coords, distance_threshold_km=DIST_THRESH, sigma_km=SIGMA)

# ---- degree distribution ----
degrees = (A > 0).sum(axis=1)
degree_hist = np.bincount(degrees)

# ---- weight-vs-distance curve (compute raw kernel for plotting) ----
d_plot = np.linspace(0, 800, 200)
w_plot = np.exp(-(d_plot ** 2) / (2 * SIGMA ** 2))
w_plot[d_plot >= DIST_THRESH] = 0.0

# ---- summary stats ----
n_edges = int((A > 0).sum() // 2)
sparsity_pct = float(100 * (A > 0).mean())
row_sum_min = float(A.sum(axis=1).min())
row_sum_max = float(A.sum(axis=1).max())

pairwise = D[np.triu_indices_from(D, k=1)]
dist_summary = {
    "min":    float(pairwise.min()),
    "p25":    float(np.percentile(pairwise, 25)),
    "median": float(np.percentile(pairwise, 50)),
    "p75":    float(np.percentile(pairwise, 75)),
    "max":    float(pairwise.max()),
}

nn_dist_per_station = D.copy()
np.fill_diagonal(nn_dist_per_station, np.inf)
nearest_nb = nn_dist_per_station.min(axis=1)

result = {
    "network": "SAWS 25-station",
    "params":  {"distance_threshold_km": DIST_THRESH, "sigma_km": SIGMA,
                "row_normalise": True, "self_loops": False},
    "n_nodes": N,
    "n_undirected_edges": n_edges,
    "sparsity_pct":       sparsity_pct,
    "row_sum_range":      [row_sum_min, row_sum_max],
    "degree_distribution": {
        "min":     int(degrees.min()),
        "max":     int(degrees.max()),
        "mean":    float(degrees.mean()),
        "std":     float(degrees.std()),
        "per_node_degree": [
            {"station": coords["station"].iloc[i], "degree": int(degrees[i]),
             "nearest_neighbour_km": float(nearest_nb[i])}
            for i in range(N)
        ],
    },
    "pairwise_distance_summary_km": dist_summary,
}

(OUT / "graph_characterisation.json").write_text(json.dumps(result, indent=2))
print(f"Wrote {OUT}/graph_characterisation.json")

# ---- figure: degree hist + weight curve side-by-side ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: degree histogram
ax1.bar(range(len(degree_hist)), degree_hist, color="steelblue", edgecolor="black")
ax1.set_xlabel("Node degree (# non-zero edges)")
ax1.set_ylabel("Number of stations")
ax1.set_title(f"SAWS 25-station degree distribution\n"
              f"(threshold {DIST_THRESH:.0f} km, σ={SIGMA:.0f} km)")
ax1.grid(alpha=0.3)
ax1.set_xticks(range(0, len(degree_hist)))

# Right: weight vs distance
ax2.plot(d_plot, w_plot, color="crimson", lw=2, label="Gaussian kernel")
ax2.axvline(DIST_THRESH, color="black", linestyle="--", alpha=0.6, label=f"cutoff = {DIST_THRESH:.0f} km")
ax2.axvline(SIGMA, color="grey", linestyle=":", alpha=0.6, label=f"σ = {SIGMA:.0f} km")
ax2.axhline(np.exp(-0.5), color="grey", linestyle=":", alpha=0.4)
ax2.set_xlabel("Great-circle distance (km)")
ax2.set_ylabel("Adjacency weight (pre row-norm)")
ax2.set_title("Weight-vs-distance curve")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(-0.02, 1.05)

plt.tight_layout()
plt.savefig(OUT / "graph_characterisation.png", dpi=120, bbox_inches="tight")
print(f"Wrote {OUT}/graph_characterisation.png")

# ---- console summary ----
print("\nSAWS graph summary:")
print(f"  {N} nodes, {n_edges} undirected edges, {sparsity_pct:.1f}% non-zero entries")
print(f"  degree: min={degrees.min()}, mean={degrees.mean():.1f}, max={degrees.max()}")
print(f"  pairwise dist median={dist_summary['median']:.0f} km, max={dist_summary['max']:.0f} km")
print(f"  row-normalised (row sums in [{row_sum_min:.3f}, {row_sum_max:.3f}])")
