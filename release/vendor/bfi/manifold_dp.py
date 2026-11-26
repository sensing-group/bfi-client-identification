"""Geometry-aware differential privacy for the BFI Givens-angle representation.

The DP-SQ mechanism of Seif et al. perturbs each quantization cell
independently with a per-cell Bernoulli flip, giving pure eps-DP at
any bit-depth b. The bit-depth dead zone (Section V) is a structural
limit of this construction: when the cells are wide (small b), the
mechanism's noise is comparable to the true variability of the
features and the per-coordinate flips destroy identity information.
When the cells are narrow (large b), the mechanism's noise is fixed
in cell units (probability p* = e^eps / (e^eps + 1)) but the
underlying angular detail in each cell vanishes, leaving the
attacker's information-theoretic floor untouched.

A geometry-aware alternative is to perturb in the continuous angle
domain *before* quantization. On the Givens torus T^d the natural
choice is the wrapped Gaussian:

    y_i = (x_i + z_i) mod 2 pi,    z_i ~ N(0, sigma^2)

with sigma calibrated so that the released (post-quantization)
output is (eps, delta)-DP under the Gaussian-mechanism analysis.
On a flat amplitude domain in R^d (the NTU-Fi proxy used in this
paper) the wrap is unnecessary and the construction reduces to the
ordinary Gaussian mechanism followed by deterministic quantization.
Quantization is post-processing of the noisy continuous output and
preserves DP at no additional privacy cost (Dwork & Roth, Theorem
2.1).

Privacy guarantee. The released quantized vector is (eps, delta)-DP
when sigma >= Delta * sqrt(2 ln(1.25 / delta)) / eps, where Delta is
the L2 sensitivity of the per-coordinate feature with respect to
adjacent inputs. This is the standard Gaussian mechanism (Dwork &
Roth 2014, A.1).

Trade-off vs DP-SQ. DP-SQ gives pure eps-DP; this mechanism gives
(eps, delta)-DP. The two privacy regimes are not comparable in
general, but for the per-window threat model studied in this paper
the (eps, delta) guarantee at delta = 1e-5 is operationally similar
and avoids the dead zone empirically.
"""

from __future__ import annotations

import numpy as np


def gaussian_sigma(*, eps: float, delta: float, sensitivity: float) -> float:
    """Standard Gaussian mechanism noise scale.

    For an input x with L2 sensitivity Delta, releasing x + N(0, sigma^2 I)
    is (eps, delta)-DP when sigma >= sqrt(2 ln(1.25 / delta)) * Delta / eps.

    Parameters
    ----------
    eps : float
        Privacy budget. Smaller eps = more noise.
    delta : float
        Privacy failure probability. Standard values: 1e-5 to 1e-9.
    sensitivity : float
        L2 sensitivity of the feature (the maximum L2 distance between
        any two adjacent inputs).
    """
    if eps <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("require eps > 0 and 0 < delta < 1")
    return float(sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / eps)


def gaussian_pre_quant_noise(
    feats: np.ndarray,
    *,
    eps: float,
    delta: float,
    lo: float,
    hi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add calibrated Gaussian noise BEFORE deterministic quantization.

    The mechanism is the standard Gaussian mechanism with per-coordinate
    sensitivity equal to the dynamic range (hi - lo). The downstream
    deterministic quantization is post-processing and preserves DP
    (Dwork & Roth 2014, Proposition 2.1).

    Parameters
    ----------
    feats : (N, d) feature matrix.
    eps, delta : (eps, delta)-DP parameters.
    lo, hi : per-coordinate dynamic range used for sensitivity calibration.
    rng : NumPy generator for reproducibility.

    Returns
    -------
    (N, d) noisy feature matrix in the same range as `feats`.
    """
    sensitivity = float(hi - lo)
    sigma = gaussian_sigma(eps=eps, delta=delta, sensitivity=sensitivity)
    return feats + rng.standard_normal(feats.shape) * sigma


def manifold_quantise_arr(
    feats: np.ndarray,
    b: int,
    lo: float,
    hi: float,
    *,
    eps: float,
    delta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply Gaussian noise then deterministic b-bit quantization.

    The released output is (eps, delta)-DP at the per-window level. For
    fair comparison with DP-SQ this is applied independently per row of
    `feats` with the same sigma; cross-window composition is not
    accounted for here (the attacker sees windows as independent
    samples).

    Parameters
    ----------
    feats : (N, d) feature matrix.
    b : bit-depth of the deterministic quantizer (cells = 2^b).
    lo, hi : per-coordinate range for both noise calibration and quantization.
    eps, delta : (eps, delta)-DP parameters.
    rng : NumPy generator for reproducibility.

    Returns
    -------
    (N, d) quantized feature matrix.
    """
    if b <= 0:
        # b = 0 case: collapse everything to the midpoint, no privacy needed
        return np.full_like(feats, (lo + hi) / 2.0, dtype=float)
    noisy = gaussian_pre_quant_noise(
        feats, eps=eps, delta=delta, lo=lo, hi=hi, rng=rng,
    )
    n_bins = 1 << b
    cell = (hi - lo) / n_bins
    out = np.empty_like(noisy, dtype=float)
    for i in range(noisy.shape[0]):
        idx = np.clip(
            np.floor((noisy[i] - lo) / cell).astype(int),
            0, n_bins - 1,
        )
        out[i] = lo + (idx + 0.5) * cell
    return out
