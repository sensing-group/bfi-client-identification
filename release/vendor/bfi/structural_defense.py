"""LeakyBeam-style structural obfuscation for the BFI side channel.

Background. Xiao et al. (NDSS 2025) propose an AP-side defense in which
the AP applies a per-packet random unitary transform Q^obf to the LTF
before sounding. The beamformee returns V^obf = Q^obf^T @ V (the
steering matrix obfuscated by Q^obf), and the AP recovers the true V
via Q^obf @ V^obf. The eavesdropper, who does not know Q^obf, sees
only the obfuscated matrix.

This module simulates the *information-theoretic effect* of the same
mechanism on the amplitude-CSI features used by our DP-SQ pipeline.
For each window we draw a fresh random orthogonal matrix Q ∈ R^{d×d}
and emit Y = X @ Q^T (equivalently, Q @ X^T as a column vector). The
AP can recover X via Q^T @ Y, but the attacker, building a 1-NN
classifier on Y from training-side captures, must classify on a random
orthogonal projection of the original features.

Note (proxy and scope). This is not a faithful protocol-level
simulation — LeakyBeam's Q^obf operates on the steering matrix V, not
on amplitude CSI. As with our DP-SQ evaluation, we use NTU-Fi as a
proxy because no public BFI capture exists at the scale required.
The mechanism's information-theoretic effect (random per-window
orthogonal mixing of the feature vector) is preserved across
substrates.
"""

from __future__ import annotations

import numpy as np


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random d×d orthogonal matrix uniformly via QR of a
    standard Gaussian. The sign of each Q column is normalised so
    that the diagonal of R is positive, giving a uniform sample on
    the orthogonal group O(d) (Mezzadri 2007)."""
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    q *= np.sign(d)[None, :]
    return q


def structural_obfuscate(
    feats: np.ndarray,
    rng: np.random.Generator,
    *,
    granularity: str = "per-window",
) -> np.ndarray:
    """Apply LeakyBeam-style random orthogonal obfuscation.

    Parameters
    ----------
    feats : (N, d) feature matrix (rows = windows, cols = features).
    rng   : numpy generator for reproducibility.
    granularity : where the random Q is regenerated.
        - "per-window"  : independent Q per window (LeakyBeam-style).
        - "per-session" : single Q per call (illustrates the
                          degenerate case where the AP and attacker
                          share Q across the whole capture).
        - "shared"      : same as "per-session" but kept for clarity.

    Returns
    -------
    Obfuscated (N, d) matrix Y where each row Y_i = Q_i @ X_i for
    "per-window", or Y = X @ Q^T for the shared cases.
    """
    n, d = feats.shape
    if granularity == "per-window":
        out = np.empty_like(feats)
        for i in range(n):
            q = random_orthogonal(d, rng)
            out[i] = q @ feats[i]
        return out
    elif granularity in ("per-session", "shared"):
        q = random_orthogonal(d, rng)
        return feats @ q.T
    else:
        raise ValueError(f"unknown granularity: {granularity}")
