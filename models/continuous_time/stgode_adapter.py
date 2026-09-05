"""
Adapter (method) for STGODE — Fang et al. (KDD 2021).

STGODE requires TWO adjacencies (paper §4.1):
  - Spatial A_sp  (Eq. 4):  A_ij = exp(−d_ij²/σ²)      if that value ≥ ε; else 0
                            d_ij = raw geographic distance (km)
  - Semantic A_se (Eq. 6):  A_ij = 1  if DTW(X_i, X_j) < ε; else 0
                            DTW between per-station diurnal-profile signals

Both are then symmetrically renormalised (Eq. 1):
                A_hat = ½·α·(I + D^{−½}·A·D^{−½}),  α = 0.8

Two construction modes exposed under `model.graph_construction`:

  - "paper"  (DEFAULT, and paper-faithful): implements Eq. 4 + Eq. 6 verbatim.
             Uses `model.sigma_km` (raw-distance Gaussian bandwidth for spatial),
             `model.cutoff_km` (hard cutoff on raw distance), and `model.epsilon_dtw`
             (upper bound on raw DTW distance for the semantic edge). This is the
             correct implementation of what the paper defines and is more
             geographically meaningful than upstream's own code.

  - "upstream" (opt-in ablation): reproduces `STGODE/utils.py::read_data`'s
                z-scored kernel — z-score the distance matrix, then apply a
                Gaussian at the σ/threshold values `model.sigma_sp/thres_sp`
                (spatial) and `model.sigma_se/thres_se` (semantic, binarised
                above threshold). This is what the upstream repo actually
                shipped; on their PEMS road-network data an infinity mask on
                non-adjacent edges did the sparsification and the Gaussian
                itself was nearly a no-op. On our full-connectivity haversine
                matrix the upstream recipe produces a near-uniform "graph"
                (verified 2026-08-11 — see specs/STGODE_INTEGRATION_AUDIT.md
                §C1). Retained here only for a paper-vs-upstream sensitivity
                ablation, NOT as the production path.

DTW distance matrix (both modes read the same file) is precomputed once via
`scripts/build_stgode_semantic_adj.py` and cached at
`preprocessing/cache/saws/dtw_matrix_{variable}.npy`. The path is variable-
suffixed because the semantic similarity depends on which variable's diurnal
profile is DTW'd; per-target files stay cheap because with the corrected
mean-across-days axis the DTW input is only 24 steps long.

forward() and predict() call the same model path; the adapter no longer forces
`.train()` or `.eval()` inside forward — the trainer owns that mode (see
`training/trainer.py`), consistent with our other adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from models.base_adapter import BaseAdapter
from models.continuous_time.stgode_model import STGODEModel


class STGODEAdapter(BaseAdapter):

    def __init__(self):
        self.model: STGODEModel | None = None
        self._A_sp_hat: torch.Tensor | None = None
        self._A_se_hat: torch.Tensor | None = None
        self._adjacencies_on_device: torch.device | None = None

    # ------------------------------------------------------------------
    # Adjacency construction — paper Eq. 4 / Eq. 6 (default)
    # ------------------------------------------------------------------

    @staticmethod
    def _paper_spatial_adj(dist_km: np.ndarray, sigma_km: float,
                          cutoff_km: float) -> np.ndarray:
        """Paper Eq. 4: A_ij = exp(−d_ij²/σ²), zero where d_ij ≥ cutoff.

        Distances in km, σ in km. Uses a HARD-DISTANCE cutoff (rather than a
        weight-space threshold) so the sparsity has a physical meaning.
        """
        A = np.exp(-(dist_km ** 2) / (sigma_km ** 2))
        A[dist_km >= cutoff_km] = 0.0
        np.fill_diagonal(A, 0.0)
        return A.astype(np.float32)

    @staticmethod
    def _paper_semantic_adj(dtw_dist: np.ndarray, epsilon_dtw: float) -> np.ndarray:
        """Paper Eq. 6: A_ij = 1 if DTW(X_i, X_j) < ε; 0 otherwise.

        Binary adjacency directly from RAW DTW distance. No Gaussian, no
        z-score — those are upstream deviations.
        """
        A = (dtw_dist < epsilon_dtw).astype(np.float32)
        np.fill_diagonal(A, 0.0)
        return A

    # ------------------------------------------------------------------
    # Adjacency construction — upstream z-scored variant (opt-in ablation)
    # ------------------------------------------------------------------

    @staticmethod
    def _upstream_zscored_kernel(dist_matrix: np.ndarray, sigma: float,
                                threshold: float, binarise: bool) -> np.ndarray:
        """Reproduces upstream `STGODE/utils.py::read_data`'s recipe.

        On PEMS data upstream sparsifies via inf-mask on non-adjacent edges
        BEFORE this z-score, so the resulting kernel behaves reasonably. On
        a full-connectivity distance matrix (like ours) it collapses to a
        near-uniform matrix that peaks at the MEAN distance. Retained here
        only for the paper-vs-upstream sensitivity ablation.
        """
        finite = np.isfinite(dist_matrix)
        mean = float(dist_matrix[finite].mean()) if finite.any() else 0.0
        std = float(dist_matrix[finite].std()) if finite.any() else 1.0
        if std <= 0:
            std = 1.0
        z = (dist_matrix - mean) / std
        A = np.exp(-(z ** 2) / (sigma ** 2))
        if binarise:
            out = np.zeros_like(A)
            out[A > threshold] = 1.0
            return out.astype(np.float32)
        A[A < threshold] = 0.0
        return A.astype(np.float32)

    # ------------------------------------------------------------------
    # Symmetric renormalisation — paper Eq. 1 (both modes)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_adj(A: np.ndarray, alpha: float = 0.8) -> np.ndarray:
        """Paper Eq. 1 (identical to upstream `utils.py::get_normalized_adj`):

            A_hat = ½·α·(I + D^{−½}·A·D^{−½})

        α = 0.8 is the paper's default (§5.3).
        """
        D = np.asarray(A.sum(axis=1)).reshape(-1)
        D[D <= 1e-4] = 1e-4
        Dinv_sqrt = 1.0 / np.sqrt(D)
        A_norm = (Dinv_sqrt.reshape(-1, 1) * A) * Dinv_sqrt.reshape(1, -1)
        return ((alpha / 2.0) * (np.eye(A.shape[0], dtype=A_norm.dtype) + A_norm)).astype(np.float32)

    # ------------------------------------------------------------------
    # Main entry: build both adjacencies from cache
    # ------------------------------------------------------------------

    def _build_adjacencies(self, config: Dict[str, Any], graph: torch.Tensor) -> None:
        model_cfg = config.get("model", {})
        mode = str(model_cfg.get("graph_construction", "paper")).lower()

        cache_dir = Path(config.get("preprocessing", {}).get("cache_dir",
                                                             "preprocessing/cache/saws"))

        # --- Spatial: haversine from cache coords ---
        coords_path = cache_dir / "coords.npy"
        if not coords_path.exists():
            raise FileNotFoundError(
                f"STGODE needs {coords_path} (station lat/lon). "
                "Rebuild the cache with the standard pipeline."
            )
        coords = np.load(coords_path)                       # (N, 2) = (lat, lon)
        dist_sp = _haversine_matrix_km(coords[:, 0], coords[:, 1])

        # --- Semantic: DTW from cache ---
        # Variable-suffixed filename — see scripts/build_stgode_semantic_adj.py.
        # Fall back to legacy `dtw_matrix.npy` for the old (uncorrected) build,
        # so a paper-vs-old comparison is possible without recomputing.
        dtw_var = str(config.get("data", {}).get("target_variable", "t2m"))
        dtw_path = cache_dir / f"dtw_matrix_{dtw_var}.npy"
        if not dtw_path.exists():
            legacy = cache_dir / "dtw_matrix.npy"
            if legacy.exists():
                dtw_path = legacy
            else:
                raise FileNotFoundError(
                    f"STGODE needs {dtw_path} — semantic DTW adjacency for {dtw_var}. "
                    f"Run: python scripts/build_stgode_semantic_adj.py --variable {dtw_var}"
                )
        dtw_dist = np.load(dtw_path)

        # --- Construct raw adjacencies per selected mode ---
        if mode == "paper":
            sigma_km    = float(model_cfg.get("sigma_km", 200.0))
            cutoff_km   = float(model_cfg.get("cutoff_km", 500.0))
            epsilon_dtw = float(model_cfg.get("epsilon_dtw", 0.5))
            A_sp = self._paper_spatial_adj(dist_sp, sigma_km, cutoff_km)
            A_se = self._paper_semantic_adj(dtw_dist, epsilon_dtw)
        elif mode == "upstream":
            sigma_sp = float(model_cfg.get("sigma_sp", 10.0))
            thres_sp = float(model_cfg.get("thres_sp", 0.5))
            sigma_se = float(model_cfg.get("sigma_se", 0.1))
            thres_se = float(model_cfg.get("thres_se", 0.6))
            A_sp = self._upstream_zscored_kernel(dist_sp, sigma_sp, thres_sp, binarise=False)
            A_se = self._upstream_zscored_kernel(dtw_dist, sigma_se, thres_se, binarise=True)
        else:
            raise ValueError(f"Unknown graph_construction {mode!r}; must be 'paper' or 'upstream'")

        # --- Symmetric renormalisation (paper Eq. 1) ---
        A_sp_hat = self._normalise_adj(A_sp)
        A_se_hat = self._normalise_adj(A_se)

        self._A_sp_hat = torch.from_numpy(A_sp_hat)
        self._A_se_hat = torch.from_numpy(A_se_hat)
        self._adjacencies_on_device = None  # invalidate device-cache

    # ------------------------------------------------------------------
    # BaseAdapter contract
    # ------------------------------------------------------------------

    def build_model(self, config: Dict[str, Any], graph: torch.Tensor) -> STGODEModel:
        """
        Args:
            config: full experiment config dict.
            graph:  our cache's spatial adj (N, N). STGODE ignores this — it
                    reconstructs both adjacencies from the cache directory
                    (`coords.npy` + `dtw_matrix_{var}.npy`) so it can apply
                    the paper's own kernel.
        """
        N = graph.shape[0]
        self._build_adjacencies(config, graph)

        merged = {
            **config.get("model", {}),
            "input_dim":  config["meta"]["n_vars"],
            "horizon":    config["data"]["horizon"],
            "input_len":  config["data"]["input_len"],
            "ode":        config.get("ode", {}),
        }
        self.model = STGODEModel(
            merged, num_nodes=N,
            A_sp_hat=self._A_sp_hat, A_se_hat=self._A_se_hat,
        )
        return self.model

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, N, D). Returns (B, H, N, 1).

        Trainer owns train/eval MODE; this adapter does NOT set mode (M6 fix).
        Grad context is inherited from the caller.
        """
        self._move_adjacencies_to(x.device)
        return self.model(x)

    def predict(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, N, D). Returns (B, H, N, 1).

        Trainer owns train/eval MODE (M6 fix); but we DO wrap in `no_grad`
        because the trainer's `_evaluate_test` loop calls `pred.cpu().numpy()`
        directly on the result — that fails on grad-tracking tensors. The
        `no_grad` context is a separate concern from mode (BatchNorm running
        statistics update, dropout activation), so keeping it here is
        consistent with the M6 spirit (mode ownership) while preserving the
        eval-path invariant the trainer relies on. Verified: predict()
        does not toggle model.training.
        """
        self._move_adjacencies_to(x.device)
        with torch.no_grad():
            return self.model(x)

    def _move_adjacencies_to(self, device: torch.device) -> None:
        """Move stored adjacencies + upstream's plain-attribute adj into `device`.

        Upstream `odegcn.ODEG` stores `adj` as a plain attribute, not
        `register_buffer`, so `nn.Module.to(device)` doesn't move it. We
        walk the submodules and manually transfer.

        Also moves `self._A_sp_hat` / `self._A_se_hat` so the guard latches
        (previously the guard checked a tensor that never moved, so this
        method ran on every forward pass — small perf leak).

        Device comparison uses the CANONICAL device (with index for CUDA) —
        `torch.device("cuda") != torch.device("cuda:0")`, so we normalise
        via a moved tensor's actual `.device` attribute.
        """
        # Normalise: for CUDA without an explicit index, torch resolves to the
        # current CUDA device (e.g. cuda:0). Get the canonical form by moving
        # a scratch tensor and reading its .device back.
        canonical = torch.zeros((), device=device).device
        if self._adjacencies_on_device == canonical:
            return
        if self._A_sp_hat is not None:
            self._A_sp_hat = self._A_sp_hat.to(canonical)
        if self._A_se_hat is not None:
            self._A_se_hat = self._A_se_hat.to(canonical)
        for m in self.model.net.modules():
            if hasattr(m, "adj") and isinstance(m.adj, torch.Tensor):
                m.adj = m.adj.to(canonical)
            if hasattr(m, "A_hat") and isinstance(m.A_hat, torch.Tensor):
                m.A_hat = m.A_hat.to(canonical)
        self._adjacencies_on_device = canonical

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Haversine utility (matches preprocessing/graph.py::_haversine_matrix)
# ---------------------------------------------------------------------------

def _haversine_matrix_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6371.0
    lat = np.radians(lats)
    lon = np.radians(lons)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))
