"""He BFI presence-detection dataset loader for the BFI framework.

Source: He et al., IEEE Dataport — BFI presence-detection corpus. Each
``.npy`` under ``<root>/<scenario>/<bandwidth>/`` contains a stream of
packets with phi/psi cell indices in [0, 64) (6-bit AP capture).
Filenames encode binary presence: '1p' = human present, '0p' = empty.

This loader:
    1. Aggregates each file to a single feature vector (mean over packets).
    2. Maps integer cell indices back to mid-cell angles in [-pi, pi].
    3. Returns a flat :class:`Substrate` with circular=True.

Path resolution:
    explicit ``data_root`` arg > BFI_HE_PRESENCE_ROOT env var > a built-in
    default that points at the paper-repo path on the lab machine.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Literal

import numpy as np

from .substrate import Substrate


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HE_PRESENCE_ROOT_ENV = "BFI_HE_PRESENCE_ROOT"
# Repository default (release/vendor/bfi/framework/datasets -> repository root).
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_HE_PRESENCE_ROOT = str(REPO_ROOT / "data" / "he_presence" / "dataset")

# AP captures at b=6 (cell indices in [0, 63]).
DEFAULT_B_CAPTURE = 6

DEFAULT_SCENARIO = "Meeting"
DEFAULT_BANDWIDTH: Literal["20M", "40M", "80M"] = "80M"

# Filename markers for the binary presence label.
LABEL_PRESENT = "1p"
LABEL_EMPTY = "0p"

# Output range.
HE_PRESENCE_LO = -float(np.pi)
HE_PRESENCE_HI = float(np.pi)


def _resolve_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    env = os.environ.get(HE_PRESENCE_ROOT_ENV)
    if env:
        return Path(env)
    return Path(DEFAULT_HE_PRESENCE_ROOT)


def _label_from_filename(name: str) -> int | None:
    if LABEL_PRESENT in name:
        return 1
    if LABEL_EMPTY in name:
        return 0
    return None


def load_he_presence(
    data_root: str | Path | None = None,
    *,
    scenario: str = DEFAULT_SCENARIO,
    bandwidth: str = DEFAULT_BANDWIDTH,
    b_capture: int = DEFAULT_B_CAPTURE,
) -> Substrate:
    """Load all ``(scenario, bandwidth)`` files; aggregate per file.

    Returns one row per ``.npy`` file, with feature vector = mean over
    packets of mid-cell angles in [-pi, pi].
    """
    root = _resolve_root(data_root)
    pattern = str(root / scenario / bandwidth / "*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"no .npy files matched {pattern}; "
            f"set ${HE_PRESENCE_ROOT_ENV} or pass data_root=..."
        )

    cell_count = 1 << b_capture
    feats: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0
    for f in files:
        a = np.load(f, allow_pickle=True)
        a = np.asarray(a.tolist(), dtype=np.int64)  # (n_pkts, n_sc, 2)
        if a.size == 0 or a.ndim != 3:
            skipped += 1
            continue
        m = a.mean(axis=0).reshape(-1).astype(np.float64)
        # Map cell index -> mid-cell angle in [-pi, pi].
        m = (m + 0.5) / cell_count * (2 * np.pi) - np.pi
        label = _label_from_filename(os.path.basename(f))
        if label is None:
            skipped += 1
            continue
        feats.append(m)
        labels.append(label)
    if not feats:
        raise RuntimeError(
            f"no labelled packets found under {pattern} (skipped {skipped})"
        )
    X = np.stack(feats, axis=0).astype(np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return Substrate(
        name=f"he_presence_{scenario.lower()}_{bandwidth.lower()}",
        X=X,
        y=y,
        lo=HE_PRESENCE_LO,
        hi=HE_PRESENCE_HI,
        circular=True,
        meta={
            "scenario": scenario,
            "bandwidth": bandwidth,
            "b_capture": b_capture,
            "n_files_skipped": skipped,
            "n_classes": int(np.unique(y).size),
        },
    )
