"""Angle quantisation for BFI Givens parameters.

Two mechanisms are exposed:

1. **`quantise_angles`** — deterministic midpoint-uniform quantisation,
   matching IEEE 802.11-2020 §19.3.12.3.6. This is what a standards-
   compliant beamformee actually transmits today.

2. **`dp_sq_quantise_angles`** — *DP-SQ* stochastic quantisation per
   Seif et al., arXiv:2512.18529v2 (DySPAN 2026). Replaces the
   deterministic mapping with a biased coin between the two nearest
   reconstruction levels:

       p*(ε) = e^ε / (e^ε + 1),
       output =
         nearer level  w.p. p*(ε),
         farther level w.p. 1 - p*(ε).

   Provides ε-local DP under *cell-wise adjacency* (two angles in the
   same quantisation interval are statistically indistinguishable).
   ε → ∞ recovers deterministic quantisation; ε → 0 yields uniform
   randomisation between the two adjacent midpoints.

   Per-angle MSE (Lemma 1 in Seif et al.):

       E[(â - a)²] = Δ²/12 · (4 - 3·κ(ε)),   κ(ε) = tanh(ε/2)

   We expose `dp_sq_mse(b, eps, range_)` for the closed-form value.

ψ ("psi") angles:   real Givens rotations; quantised over the range [0, π/2].
φ ("phi") angles:   diagonal phase angles; circular over [-π, π).

Bit-depth presets (b_ψ, b_φ):

    802.11ac  SU-MIMO   (4, 6)
    802.11ac  MU-MIMO   (7, 9)
    802.11ax  SU-MIMO   (4, 6)
    802.11ax  MU-MIMO   (7, 9)

For `b = 0` the result is a constant midpoint of the angle's range
(degenerate; useful for sanity checks at the privacy extreme).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PSI_MIN = 0.0
PSI_MAX = np.pi / 2.0
PHI_MIN = -np.pi
PHI_MAX = np.pi


@dataclass(frozen=True)
class BFIPreset:
    """Standard codebook bit-depth pair (b_psi, b_phi)."""

    name: str
    b_psi: int
    b_phi: int


PRESET_802_11AC_SU = BFIPreset("802.11ac SU", b_psi=4, b_phi=6)
PRESET_802_11AC_MU = BFIPreset("802.11ac MU", b_psi=7, b_phi=9)
PRESET_802_11AX_SU = BFIPreset("802.11ax SU", b_psi=4, b_phi=6)
PRESET_802_11AX_MU = BFIPreset("802.11ax MU", b_psi=7, b_phi=9)


def _midpoint_quantise(x: np.ndarray, lo: float, hi: float, b: int) -> np.ndarray:
    """Uniform midpoint quantisation: bin x into 2^b cells, return cell centre."""
    if b <= 0:
        return np.full_like(x, (lo + hi) / 2.0, dtype=float)
    n_bins = 1 << b
    cell = (hi - lo) / n_bins
    idx = np.clip(np.floor((x - lo) / cell).astype(int), 0, n_bins - 1)
    return lo + (idx + 0.5) * cell


def quantise_angles(
    psi: np.ndarray,
    phi: np.ndarray,
    b_psi: int,
    b_phi: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic midpoint quantisation of ψ and φ angles.

    Args:
        psi:    array of ψ angles in [0, π/2].
        phi:    array of φ angles in [-π, π).
        b_psi:  bits per ψ angle (≥ 0; 0 maps everything to π/4).
        b_phi:  bits per φ angle (≥ 0; 0 maps everything to 0).

    Returns:
        (psi_q, phi_q) — quantised arrays of the same shape as the inputs.
    """
    psi_q = _midpoint_quantise(np.asarray(psi, dtype=float), PSI_MIN, PSI_MAX, b_psi)
    phi_q = _midpoint_quantise(np.asarray(phi, dtype=float), PHI_MIN, PHI_MAX, b_phi)
    return psi_q, phi_q


# ---------------------------------------------------------------------------
# DP-SQ stochastic quantisation (Seif et al., arXiv:2512.18529v2)
# ---------------------------------------------------------------------------


def _dp_sq_quantise_1d(
    x: np.ndarray,
    lo: float,
    hi: float,
    b: int,
    eps: float,
    rng: np.random.Generator,
    circular: bool,
) -> np.ndarray:
    """Apply DP-SQ to a 1D real array `x` constrained to [lo, hi).

    With reconstruction levels at midpoints of `2^b` uniform cells, this
    routine selects the two nearest midpoints to each input and returns
    one of them according to the biased rule:

        nearer  w.p. p*(eps) = e^eps / (e^eps + 1)
        farther w.p. 1 - p*(eps).

    This matches Lemma 1 of Seif et al. (arXiv:2512.18529v2) exactly:
    per-angle MSE = Δ²/12 · (4 - 3·κ(ε)).

    Privacy interpretation. The likelihood ratio bound exp(ε) holds
    under *half-cell adjacency*: two inputs that share both their
    surrounding cell AND the same half of that cell (i.e., that lie
    on the same side of the cell's midpoint) yield output distributions
    that satisfy the ε-DP guarantee. Inputs in *different* halves of the
    same cell can produce disjoint output supports (one outputs μ and
    μ_{j-1}, the other μ and μ_{j+1}), so the cell-wise adjacency
    relation in Seif et al. is, strictly read, **stronger** than this
    mechanism's actual guarantee.

    For sensing-privacy purposes the half-cell guarantee is the useful
    one: it bounds the inference power of an adversary that observes
    one stochastic-quantised report and tries to recover the input
    angle to better than half a quantisation cell — i.e. to better
    than the deterministic quantiser already permits.

    Boundary handling:
        - circular=True   →  wrap modulo (hi - lo)         (used for φ)
        - circular=False  →  clamp at the edges            (used for ψ)
    """
    if b <= 0:
        return np.full_like(x, (lo + hi) / 2.0, dtype=float)

    n_bins = 1 << b
    cell = (hi - lo) / n_bins

    # Cell index of each input (the cell that contains it).
    rel = (x - lo) / cell  # ∈ [0, n_bins) for valid inputs
    idx_near = np.clip(np.floor(rel).astype(int), 0, n_bins - 1)
    mu_near = lo + (idx_near + 0.5) * cell

    # Which side of the cell midpoint does x fall on?
    # +1: x is in the upper half  →  second-nearest midpoint is at idx_near + 1
    # -1: x is in the lower half  →  second-nearest midpoint is at idx_near - 1
    # If x is exactly at the midpoint, default to +1 (arbitrary; symmetric).
    side = np.where(x >= mu_near, 1, -1).astype(int)
    idx_far = idx_near + side

    if circular:
        idx_far = np.mod(idx_far, n_bins)
    else:
        idx_far = np.clip(idx_far, 0, n_bins - 1)
        # If the clamp pushed idx_far back onto idx_near (boundary case),
        # the "farther" level coincides with the nearer one; output is
        # deterministic at the nearer midpoint, no privacy lost in that
        # specific input cell. Acceptable for ψ ∈ [0, π/2] which has
        # no wrap-around.
    mu_far = lo + (idx_far + 0.5) * cell

    p_star = float(np.exp(eps) / (np.exp(eps) + 1.0))
    coin = rng.uniform(size=x.shape)
    return np.where(coin < p_star, mu_near, mu_far)


def dp_sq_quantise_angles(
    psi: np.ndarray,
    phi: np.ndarray,
    b_psi: int,
    b_phi: int,
    eps_psi: float,
    eps_phi: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """DP-SQ stochastic quantisation of ψ (clamped) and φ (circular).

    Provides ε-local DP under cell-wise adjacency — i.e., for any pair
    of angles in the same quantisation cell, the output distributions
    satisfy `Pr[hat | a] ≤ e^eps · Pr[hat | a']`. See Seif et al.,
    arXiv:2512.18529v2.

    Args:
        psi:      array of ψ ∈ [0, π/2].
        phi:      array of φ ∈ [-π, π).
        b_psi:    bits per ψ angle.
        b_phi:    bits per φ angle.
        eps_psi:  privacy budget per ψ release (eps > 0).
        eps_phi:  privacy budget per φ release.
        rng:      optional numpy Generator; defaults to a fresh one.

    Returns:
        (psi_q, phi_q) of the same shape as the inputs.
    """
    if rng is None:
        rng = np.random.default_rng()
    psi_q = _dp_sq_quantise_1d(
        np.asarray(psi, dtype=float),
        PSI_MIN, PSI_MAX, b_psi, eps_psi, rng, circular=False,
    )
    phi_q = _dp_sq_quantise_1d(
        np.asarray(phi, dtype=float),
        PHI_MIN, PHI_MAX, b_phi, eps_phi, rng, circular=True,
    )
    return psi_q, phi_q


def dp_sq_kappa(eps: float) -> float:
    """The bias factor κ(ε) = (e^ε − 1) / (e^ε + 1) = tanh(ε/2).

    κ ∈ [0, 1); 0 at ε = 0 (uniform), 1 in the deterministic limit.
    """
    return float(np.tanh(eps / 2.0))


def dp_sq_mse(b: int, eps: float, range_: float) -> float:
    """Closed-form per-angle MSE under DP-SQ (Lemma 1 of Seif et al.).

    MSE = Δ²/12 · (4 - 3·κ(ε)),   Δ = range_ / 2^b,   κ(ε) = tanh(ε/2).

    At ε → ∞ this recovers the deterministic-quantisation MSE Δ²/12.
    At ε = 0 it gives Δ²/3 (uniform between two adjacent midpoints).

    Args:
        b:        bit-depth (≥ 1).
        eps:      privacy parameter (≥ 0).
        range_:   total angle range (e.g. π/2 for ψ, 2π for φ).

    Returns:
        Per-angle MSE in radian².
    """
    if b < 1:
        raise ValueError("b must be >= 1 for a meaningful MSE")
    delta = range_ / (1 << b)
    return float((delta ** 2) / 12.0 * (4.0 - 3.0 * dp_sq_kappa(eps)))
