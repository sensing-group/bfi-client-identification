"""Uniform dataset loaders for the BFI framework.

Each ``load_*`` function returns a :class:`Substrate` object:

    Substrate(name, X, y, lo, hi, circular, meta)

with ``X`` shaped (N, D) float64 and ``y`` shaped (N,) int64. ``lo`` and
``hi`` give the natural support of the feature space (used by quantisation
defenses); ``circular`` is True for angle features in [-pi, pi).

Public functions:
    load_ntufi(...)          -- NTU-Fi HumanID (gait re-identification)
    load_he_location(...)    -- He CR1 BFI (39 locations)
    load_he_presence(...)    -- He BFI presence detection (binary)
"""

from __future__ import annotations

from .substrate import Substrate
from .ntufi import load_ntufi
from .he_location import load_he_location
from .he_presence import load_he_presence

__all__ = [
    "Substrate",
    "load_ntufi",
    "load_he_location",
    "load_he_presence",
]
