"""Common :class:`Substrate` dataclass returned by all framework loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Substrate:
    """A loaded dataset ready for the BFI framework.

    Attributes:
        name:     human-readable identifier (e.g. 'ntufi_humanid').
        X:        (N, D) float64 feature matrix.
        y:        (N,)  int64 labels.
        lo, hi:   natural support of the feature space (used by defenses).
                  For NTU-Fi amplitude this is (X.min(), X.max());
                  for Givens angles this is (-pi, pi).
        circular: True for angle features (DP-SQ wraps), False otherwise.
        meta:     free-form metadata dict (e.g. raw shape, antenna pair).
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    lo: float
    hi: float
    circular: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {self.X.shape}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X has {self.X.shape[0]} rows but y has {self.y.shape[0]}"
            )

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])
