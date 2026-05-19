"""
Model wrapper for CLCRN (Conditional Local Convolution Recurrent Network).

Cloned repo: models/baselines/CLCRN/
Key entry point: CLCRN/model/clcnn/recurrent/seq2seq_model.py — CLCRNModel

CLCRN is a spherical-geometry GNN. It requires four geometric tensors derived
from node lat/lon coordinates at instantiation:
  - loc_info    (N*k, 4)  : local coordinate frame inputs for the CLConv kernel
  - sparse_idx  (2, N*k)  : k-NN graph edge indices
  - geodesic    (N*k,)    : great-circle distances along each edge
  - angle_ratio (N*k,)    : angular density weights for each edge

These are computed by CLCRN's own KernelGenerator using the node coordinates
passed in from preprocessing/coords. The static adj from preprocessing is not
used — CLCRN uses its own spherical k-NN graph.

Input convention: CLCRN expects (seq_len, batch, N, d); this wrapper handles
the permutation from our pipeline's (batch, seq_len, N, d) convention.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

REPO_PATH = Path(__file__).parent / "CLCRN"


def _patch_numpy_compat():
    """
    CLCRN's sphere.py imports numpy.core.fromnumeric.product which was removed
    in NumPy 2.x. Inject the alias before importing any CLCRN code.
    """
    import numpy.core.fromnumeric as _fnumeric  # noqa: PLC0415
    if not hasattr(_fnumeric, "product"):
        _fnumeric.product = np.prod


def _build_kernel_inputs(coords_lonlat: np.ndarray, k_neighbors: int):
    """
    Use CLCRN's KernelGenerator to derive the four geometric tensors needed
    by CLCRNModel from a (N, 2) array of [lon, lat] node coordinates.
    """
    if str(REPO_PATH) not in sys.path:
        sys.path.insert(0, str(REPO_PATH))

    _patch_numpy_compat()
    from experiments.dataloader import KernelGenerator  # noqa: PLC0415

    kg = KernelGenerator(coords_lonlat, k_neighbors=k_neighbors, local_map="fast")
    sparse_idx  = torch.from_numpy(kg.sparse_idx).long()            # (2, N*k)
    loc_info    = torch.from_numpy(kg.MLP_inputs).float()           # (N*k, 4)
    geodesic    = torch.from_numpy(kg.geodesic.flatten()).float()    # (N*k,)
    angle_ratio = torch.from_numpy(kg.ratio_lists.flatten()).float() # (N*k,)
    return loc_info, sparse_idx, geodesic, angle_ratio


class CLCRNModel(nn.Module):

    def __init__(
        self,
        config: Dict[str, Any],
        num_nodes: int,
        coords_lonlat: np.ndarray,
    ):
        """
        Args:
            config:        model sub-dict from YAML.
            num_nodes:     N — number of graph nodes.
            coords_lonlat: (N, 2) numpy array of [lon, lat] in degrees.
        """
        super().__init__()
        if str(REPO_PATH) not in sys.path:
            sys.path.insert(0, str(REPO_PATH))

        from model.clcnn.recurrent.seq2seq_model import CLCRNModel as _CLCRN  # noqa: PLC0415

        k = config.get("k_neighbors", 25)
        loc_info, sparse_idx, geodesic, angle_ratio = _build_kernel_inputs(
            coords_lonlat, k
        )

        model_kwargs = {
            "node_num":                num_nodes,
            "layer_num":               config["num_layers"],
            "rnn_units":               config["hidden_dim"],
            "input_dim":               config["input_dim"],
            "output_dim":              config["output_dim"],
            "seq_len":                 config["input_len"],
            "horizon":                 config["horizon"],
            "embed_dim":               config["embed_dim"],
            "location_dim":            2,
            "lckstructure":            config.get("lck_structure", [4, 8]),
            "max_view":                config.get("max_view", 2),
            "use_curriculum_learning": False,
            "cl_decay_steps":          1000,
        }

        self.net = _CLCRN(
            loc_info=loc_info,
            sparse_idx=sparse_idx,
            geodesic=geodesic,
            angle_ratio=angle_ratio,
            logger=None,
            **model_kwargs,
        )
        self.horizon = config["horizon"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, d)  — pipeline convention
        Returns:
            (B, H, N, d)
        """
        x_t = x.permute(1, 0, 2, 3)          # (T, B, N, d)  — CLCRN convention
        out = self.net(x_t, labels=None)      # (H, B, N, d)
        return out.permute(1, 0, 2, 3)        # (B, H, N, d)  — back to pipeline
