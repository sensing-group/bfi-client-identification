"""Cohort-split helpers for BFI substrates.

A "cohort split" partitions class labels into a calibration set (which
the encoder is trained on) and a held-out target set (which the encoder
never sees during training). Three modes are supported:

    even_odd_split      even-indexed labels -> calib, odd -> held-out
    lowest_n_split      n_calib lowest labels -> calib, rest -> held-out
    random_split        random partition with given calib_size, seeded
    closed_set_split    same labels in calib and held-out; row-wise
                        train/test partition per class

Each function takes ``(X, y)`` and returns ``(X_calib, y_calib, X_held, y_held)``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _partition_by_labels(
    X: np.ndarray, y: np.ndarray,
    calib_labels: Iterable[int], held_labels: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calib_mask = np.isin(y, list(calib_labels))
    held_mask = np.isin(y, list(held_labels))
    return X[calib_mask], y[calib_mask], X[held_mask], y[held_mask]


def even_odd_split(
    X: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Even-indexed labels go to calibration, odd-indexed to held-out.

    Used by ``demo_oneshot_attack`` on NTU-Fi HumanID (14 subjects):
        calib  = labels {0, 2, 4, 6, 8, 10, 12}
        held   = labels {1, 3, 5, 7, 9, 11, 13}
    """
    unique = sorted(set(int(v) for v in y.tolist()))
    calib = [v for v in unique if v % 2 == 0]
    held = [v for v in unique if v % 2 == 1]
    return _partition_by_labels(X, y, calib, held)


def lowest_n_split(
    X: np.ndarray, y: np.ndarray, n_calib: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """First ``n_calib`` sorted labels are calibration; rest are held-out.

    Used by ``demo_oneshot_he`` on He CR1 (39 locations): 20 lowest IDs
    are calibration, 19 highest are held-out.
    """
    unique = sorted(set(int(v) for v in y.tolist()))
    if not 0 < n_calib < len(unique):
        raise ValueError(
            f"n_calib={n_calib} must be in (0, {len(unique)}); "
            f"got {len(unique)} unique labels"
        )
    calib = unique[:n_calib]
    held = unique[n_calib:]
    return _partition_by_labels(X, y, calib, held)


def random_split(
    X: np.ndarray, y: np.ndarray, n_calib: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random partition of labels into calib and held-out.

    ``seed`` makes the partition reproducible.
    """
    rng = np.random.default_rng(seed)
    unique = sorted(set(int(v) for v in y.tolist()))
    if not 0 < n_calib < len(unique):
        raise ValueError(
            f"n_calib={n_calib} must be in (0, {len(unique)})"
        )
    perm = rng.permutation(len(unique))
    selected = [unique[i] for i in perm]
    calib = selected[:n_calib]
    held = selected[n_calib:]
    return _partition_by_labels(X, y, calib, held)


def closed_set_split(
    X: np.ndarray, y: np.ndarray, train_frac: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-set: every label appears in BOTH calibration and held-out.

    Per class, ``train_frac`` of the rows go into calibration; the
    remainder into held-out. Row order is shuffled per class with the
    given seed.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1); got {train_frac}")
    rng = np.random.default_rng(seed)
    cal_idx_list, held_idx_list = [], []
    for label in np.unique(y):
        idxs = np.flatnonzero(y == label)
        rng.shuffle(idxs)
        n_train = max(1, int(round(train_frac * len(idxs))))
        cal_idx_list.append(idxs[:n_train])
        held_idx_list.append(idxs[n_train:])
    cal_idx = np.concatenate(cal_idx_list) if cal_idx_list else np.empty(0, dtype=int)
    held_idx = np.concatenate(held_idx_list) if held_idx_list else np.empty(0, dtype=int)
    return X[cal_idx], y[cal_idx], X[held_idx], y[held_idx]
