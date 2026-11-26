"""Low-level quantization primitives for the BFI framework.

This module owns the deterministic and DP-SQ stochastic quantization
implementations used by the framework. It re-exports the canonical
``_dp_sq_quantise_1d`` from ``bfi.quantisation`` (the tested original)
and adds a thin row-by-row vectorisation wrapper used by the framework's
defenses.

Constants are module-level (no magic numbers buried in function bodies).

Public API:
    midpoint_quantise_1d(x, lo, hi, b)
    dp_sq_quantise_1d(x, lo, hi, b, eps, rng, circular=False)
    quantise_rows(X, b, lo, hi, eps=None, rng=None, circular=False)
"""

from __future__ import annotations

import numpy as np

# Re-export the canonical implementation so behaviour matches the legacy
# scripts byte-for-byte (same RNG draw order).
from bfi.quantisation import _dp_sq_quantise_1d as dp_sq_quantise_1d  # noqa: F401


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# When b <= 0 the quantiser collapses to the midpoint of [lo, hi]; we
# expose this threshold as a named constant for clarity.
DEGENERATE_BIT_DEPTH = 0


def midpoint_quantise_1d(x: np.ndarray, lo: float, hi: float, b: int) -> np.ndarray:
    """Deterministic midpoint quantisation: bin into 2^b cells, return cell centre.

    Matches IEEE 802.11-2020 §19.3.12.3.6. For b <= 0 returns a constant
    array filled with (lo + hi) / 2 (degenerate; useful at the privacy
    extreme).
    """
    if b <= DEGENERATE_BIT_DEPTH:
        return np.full_like(x, (lo + hi) / 2.0, dtype=float)
    n_bins = 1 << b
    cell = (hi - lo) / n_bins
    idx = np.clip(np.floor((x - lo) / cell).astype(int), 0, n_bins - 1)
    return lo + (idx + 0.5) * cell


def quantise_rows(
    X: np.ndarray,
    b: int,
    lo: float,
    hi: float,
    *,
    eps: float | None = None,
    rng: np.random.Generator | None = None,
    circular: bool = False,
) -> np.ndarray:
    """Apply quantisation row-by-row.

    If ``rng is None`` (or ``eps is None``), apply deterministic midpoint
    quantisation. Otherwise apply DP-SQ stochastic quantisation with the
    given (b, eps) and circular boundary flag.

    The row-by-row order is preserved (same RNG draw order as the legacy
    ``quantise_arr``) so seeded outputs are bit-identical to the demo
    scripts that pre-date this framework.
    """
    out = np.empty_like(X, dtype=float)
    if rng is None or eps is None:
        for i in range(X.shape[0]):
            out[i] = midpoint_quantise_1d(X[i], lo, hi, b)
    else:
        for i in range(X.shape[0]):
            out[i] = dp_sq_quantise_1d(
                X[i], lo=lo, hi=hi, b=b, eps=eps, rng=rng, circular=circular,
            )
    return out
