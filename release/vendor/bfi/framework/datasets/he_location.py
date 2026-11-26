"""He CR1 BFI location dataset loader for the BFI framework.

Loads the He et al. IEEE Dataport "Conference Room 1" subset:
    BFI.pth shape: (N, n_subcarriers, n_antenna_pairs, 2)  -- Givens angles
    y.pth   shape: (N,)                                    -- location IDs

Path resolution:
    explicit ``data_root`` arg > BFI_HE_LOCATION_ROOT env var > a built-in
    default that points at the paper-repo path. The default is shared
    with the legacy ``demo_oneshot_he`` and is the most common location
    on the lab machine; override via env var on other machines.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .substrate import Substrate


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HE_LOCATION_ROOT_ENV = "BFI_HE_LOCATION_ROOT"
# Repository default (release/vendor/bfi/framework/datasets -> repository root).
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_HE_LOCATION_ROOT = str(
    REPO_ROOT / "data" / "he_bfi" / "dataset" / "transfer and light weight"
    / "Conference Room 1"
)

DEFAULT_SC_STRIDE = 4
DEFAULT_ANTENNA_PAIR = 0

# Givens angles span [-pi, pi) and are circular.
HE_LOCATION_LO = -float(np.pi)
HE_LOCATION_HI = float(np.pi)


def _resolve_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    env = os.environ.get(HE_LOCATION_ROOT_ENV)
    if env:
        return Path(env)
    return Path(DEFAULT_HE_LOCATION_ROOT)


def load_he_location(
    data_root: str | Path | None = None,
    *,
    sc_stride: int = DEFAULT_SC_STRIDE,
    antenna_pair: int = DEFAULT_ANTENNA_PAIR,
) -> Substrate:
    """Load the He CR1 BFI tensor and reduce to (N, D) features.

    Pipeline:
        1. Load BFI.pth -> (N, n_sc, n_pair, 2) numpy.
        2. Pick antenna_pair, stride subcarriers; flatten the trailing 2.
        3. Returned dim D = (ceil(n_sc / sc_stride)) * 2.
    """
    root = _resolve_root(data_root)
    bfi_path = root / "BFI.pth"
    y_path = root / "y.pth"
    if not bfi_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"He CR1 BFI not found at {root} (need BFI.pth and y.pth); "
            f"set ${HE_LOCATION_ROOT_ENV} or pass data_root=..."
        )

    bfi = torch.load(str(bfi_path), map_location="cpu",
                     weights_only=True).numpy()
    y_raw = torch.load(str(y_path), map_location="cpu",
                       weights_only=True).numpy().astype(np.int64)
    if bfi.ndim != 4 or bfi.shape[3] != 2:
        raise ValueError(
            f"unexpected He BFI shape {bfi.shape}; "
            f"expected (N, n_sc, n_pair, 2)"
        )
    X = bfi[:, ::sc_stride, antenna_pair, :]
    X = X.reshape(X.shape[0], -1).astype(np.float64)
    return Substrate(
        name="he_location",
        X=X,
        y=y_raw,
        lo=HE_LOCATION_LO,
        hi=HE_LOCATION_HI,
        circular=True,
        meta={
            "raw_shape": tuple(bfi.shape),
            "sc_stride": sc_stride,
            "antenna_pair": antenna_pair,
            "n_classes": int(np.unique(y_raw).size),
        },
    )
