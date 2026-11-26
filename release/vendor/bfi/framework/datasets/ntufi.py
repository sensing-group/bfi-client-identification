"""NTU-Fi HumanID loader for the BFI framework.

The full corpus is loaded via :class:`bfi.datasets.NTUFiLoader`; this
adapter aggregates each window to the feature vector used by the legacy
demos (mean over packets along the time axis, optional subcarrier
stride, then flatten).

Defaults reproduce ``demo_oneshot_attack`` exactly:
    pkt_reduce='mean', sc_stride=4, modal='CSIamp'.

Path resolution:
    explicit ``data_root`` arg > BFI_NTUFI_ROOT env var > 'data/ntu_fi/NTU-Fi-HumanID'.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

from bfi.datasets import NTUFiLoader

from .substrate import Substrate


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NTUFI_ROOT_ENV = "BFI_NTUFI_ROOT"
DEFAULT_NTUFI_ROOT = "data/ntu_fi/NTU-Fi-HumanID"

DEFAULT_PKT_REDUCE: Literal["mean", "std", "first"] = "mean"
DEFAULT_SC_STRIDE = 4
DEFAULT_MODAL: Literal["CSIamp", "CSIphase"] = "CSIamp"
DEFAULT_TASK: Literal["HumanID", "HAR"] = "HumanID"


def _aggregate(rec_csi: np.ndarray, pkt_reduce: str, sc_stride: int) -> np.ndarray:
    """Reduce a (3, 114, 500) NTU-Fi tensor to a 1D feature vector.

    pkt_reduce: 'mean' | 'std' | 'first' over the packet axis.
    sc_stride:  subcarrier stride (>=1).
    """
    if pkt_reduce == "mean":
        x = rec_csi.mean(axis=-1)
    elif pkt_reduce == "std":
        x = rec_csi.std(axis=-1)
    elif pkt_reduce == "first":
        x = rec_csi[..., 0]
    else:
        raise ValueError(f"unknown pkt_reduce: {pkt_reduce!r}")
    x = x[:, ::sc_stride]
    return x.reshape(-1)


def _resolve_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    env = os.environ.get(NTUFI_ROOT_ENV)
    if env:
        return Path(env)
    return Path(DEFAULT_NTUFI_ROOT)


def load_ntufi(
    data_root: str | Path | None = None,
    *,
    task: str = DEFAULT_TASK,
    modal: str = DEFAULT_MODAL,
    pkt_reduce: str = DEFAULT_PKT_REDUCE,
    sc_stride: int = DEFAULT_SC_STRIDE,
    splits: tuple[str, ...] = ("train", "test"),
) -> Substrate:
    """Load NTU-Fi (HumanID by default) and return a flat :class:`Substrate`.

    The returned ``X`` concatenates the requested splits in the order
    given by ``splits`` (within each split, NTUFiLoader's natural file
    order is preserved). Per-row split membership is stored in
    ``meta['split_of_row']`` so callers can reconstruct the original
    train/test partition (used by the demos).

    For cohort-disjoint experiments, pass the result through
    :func:`bfi.framework.cohort_split.even_odd_split`.
    """
    root = _resolve_root(data_root)
    loader = NTUFiLoader(root, task=task, modal=modal)

    feats: list[np.ndarray] = []
    labels: list[int] = []
    split_of_row: list[str] = []
    for split in splits:
        for rec in loader.iter_records(split=split):
            feats.append(_aggregate(rec.csi, pkt_reduce, sc_stride))
            labels.append(int(rec.label))
            split_of_row.append(split)
    if not feats:
        raise RuntimeError(f"no NTU-Fi records found under {root}")
    X = np.stack(feats, axis=0).astype(np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return Substrate(
        name=f"ntufi_{task.lower()}",
        X=X,
        y=y,
        lo=float(X.min()),
        hi=float(X.max()),
        circular=False,
        meta={
            "task": task,
            "modal": modal,
            "pkt_reduce": pkt_reduce,
            "sc_stride": sc_stride,
            "splits": list(splits),
            "split_of_row": np.asarray(split_of_row),
            "n_classes": int(np.unique(y).size),
        },
    )
