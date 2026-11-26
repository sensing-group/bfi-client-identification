"""Stateless defense functions for the BFI framework.

Each defense takes a feature matrix and returns a transformed feature
matrix. All randomness is explicit (``rng`` parameter) so calls are
reproducible.

Public API:
    deterministic(X, b, lo=None, hi=None, circular=False)
    dp_sq(X, b, eps, rng, lo=None, hi=None, circular=False)
    structural(X, rng, granularity='per-window')
"""

from __future__ import annotations

import numpy as np

from .quantize import quantise_rows
from bfi.structural_defense import structural_obfuscate


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default boundary handling. ``circular=False`` matches amplitude CSI;
# ``circular=True`` matches Givens angle features in [-pi, pi).
DEFAULT_CIRCULAR = False

# Default granularity for the structural (LeakyBeam) defense.
DEFAULT_STRUCTURAL_GRANULARITY = "per-window"


def _resolve_range(
    X: np.ndarray, lo: float | None, hi: float | None,
) -> tuple[float, float]:
    """Resolve (lo, hi) from explicit args or by inferring from ``X``."""
    lo_v = float(X.min()) if lo is None else float(lo)
    hi_v = float(X.max()) if hi is None else float(hi)
    return lo_v, hi_v


def deterministic(
    X: np.ndarray,
    b: int,
    *,
    lo: float | None = None,
    hi: float | None = None,
    circular: bool = DEFAULT_CIRCULAR,
) -> np.ndarray:
    """Deterministic midpoint quantisation defense.

    Args:
        X:        (N, D) feature matrix.
        b:        bit depth (>= 1).
        lo, hi:   range of the feature space; inferred from X if None.
        circular: ignored for deterministic midpoint (kept for API symmetry).

    Returns:
        Quantised (N, D) array.
    """
    del circular  # midpoint quantisation is identical for circular/non-circular
    lo_v, hi_v = _resolve_range(X, lo, hi)
    return quantise_rows(X, b=b, lo=lo_v, hi=hi_v, eps=None, rng=None)


def dp_sq(
    X: np.ndarray,
    b: int,
    eps: float,
    rng: np.random.Generator,
    *,
    lo: float | None = None,
    hi: float | None = None,
    circular: bool = DEFAULT_CIRCULAR,
) -> np.ndarray:
    """DP-SQ stochastic quantisation defense (Seif et al., DySPAN 2026).

    Args:
        X:        (N, D) feature matrix.
        b:        bit depth (>= 1).
        eps:      privacy budget (>= 0). eps -> infinity recovers
                  deterministic; eps = 0 yields uniform mixing between
                  the two adjacent reconstruction levels.
        rng:      numpy Generator used for stochastic draws.
        lo, hi:   range of the feature space; inferred from X if None.
        circular: True for angle features in [-pi, pi); False for amplitude.

    Returns:
        Quantised (N, D) array.
    """
    if rng is None:
        raise ValueError("dp_sq requires an explicit numpy Generator (rng=...)")
    lo_v, hi_v = _resolve_range(X, lo, hi)
    return quantise_rows(
        X, b=b, lo=lo_v, hi=hi_v, eps=eps, rng=rng, circular=circular,
    )


def structural(
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    granularity: str = DEFAULT_STRUCTURAL_GRANULARITY,
) -> np.ndarray:
    """LeakyBeam-style random orthogonal obfuscation.

    Wraps :func:`bfi.structural_defense.structural_obfuscate`. Each row
    (per-window) or the full matrix (per-session / shared) is multiplied
    by a freshly drawn d x d orthogonal matrix.

    Args:
        X:           (N, D) feature matrix.
        rng:         numpy Generator.
        granularity: "per-window" | "per-session" | "shared".

    Returns:
        Obfuscated (N, D) array.
    """
    if rng is None:
        raise ValueError("structural requires an explicit numpy Generator (rng=...)")
    return structural_obfuscate(X, rng=rng, granularity=granularity)
