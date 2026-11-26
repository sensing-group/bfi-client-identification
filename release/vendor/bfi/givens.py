"""IEEE 802.11-style Givens decomposition of the steering matrix V.

Forward (`givens_decompose`):  V → GivensAngles  (φ row-phases + ψ rotations).
Inverse (`givens_reconstruct`): GivensAngles → V.

The decomposition follows the standard QR-with-phase peeling used in IEEE
802.11ac/ax/be Compressed Beamforming Reports (cf. IEEE 802.11-2020
§19.3.12.3.6). For each column index i = 0, …, min(N_c, N_r − 1) − 1
(0-indexed):

    1. Extract a diagonal phase φ_{i,l} = arg(V[l, i]) for l = i, …, N_r − 1
       and apply exp(-j φ_{i,l}) to row l.  After this pass, V[l, i] is real.

    2. For l = N_r − 1, …, i + 1 (descending), apply a real Givens rotation
       G_{l,i}^T(ψ) on rows (i, l) with ψ = atan2(V[l, i], V[i, i]) so that
       V[l, i] is zeroed.

After all columns are processed, V is reduced to the (N_r × N_c) identity
prefix Ĩ_{N_r, N_c}. The recovered (φ, ψ) values are exactly those a
standards-compliant beamformee would report (modulo bit-quantisation).

Round-trip correctness: `givens_reconstruct(givens_decompose(V))` recovers V
to numerical precision. See `bfi/tests/test_givens.py`.

Bit-quantisation: the φ values lie in (−π, π] and the ψ values in [0, π/2);
quantisation lives in `bfi.quantisation`.

Indexing convention: angles are stored as flat 1D arrays in the order in
which they are extracted by the algorithm above. `factor_kinds` and
`factor_indices` record the type ('phi'|'psi') and (i, l) location of each
angle, enabling reconstruction without recomputing the algorithmic schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class GivensAngles:
    """All (φ, ψ) angles for one Givens decomposition of an N_r × N_c V.

    Attributes:
        phi:            array of all φ row-phase angles (range (-π, π]).
        psi:            array of all ψ Givens-rotation angles (range [0, π/2]).
        n_r:            number of rows of V (receive antennas).
        n_c:            number of columns of V (≤ n_r).
        phi_indices:    int array of shape (len(phi), 2) recording (i, l) for
                        each φ angle (i.e., column i, row l, both 0-indexed).
        psi_indices:    int array of shape (len(psi), 2) recording (i, l) for
                        each ψ angle.
        schedule:       opaque list of (kind, k) pairs in extraction order,
                        where kind ∈ {'phi', 'psi'} and k indexes into the
                        respective angle/index arrays. Used by the inverse.

    For batched input, `phi` and `psi` carry leading batch dimensions; e.g.
    if input V has shape (B, T, N_r, N_c), then phi has shape (B, T, n_phi)
    and psi has shape (B, T, n_psi).
    """

    phi: np.ndarray
    psi: np.ndarray
    n_r: int
    n_c: int
    phi_indices: np.ndarray
    psi_indices: np.ndarray
    schedule: list  # list[tuple[str, int]]


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def n_phi_for_shape(n_r: int, n_c: int) -> int:
    """Number of φ row-phase angles in the decomposition of an N_r × N_c V.

    Extracts phases on rows i .. N_r-1 for each column i = 0 .. min(N_c, N_r)-1.
    For tall V (N_c < N_r) the extra bottom-row phases are redundant but
    harmless for round-trip; for square V they capture the residual diagonal
    phase that the standard's compact form leaves implicit.
    """
    n = 0
    for i in range(min(n_c, n_r)):
        n += n_r - i  # rows i .. n_r - 1
    return n


def n_psi_for_shape(n_r: int, n_c: int) -> int:
    """Number of ψ Givens-rotation angles in the decomposition."""
    n = 0
    for i in range(min(n_c, n_r)):
        n += max(n_r - i - 1, 0)  # rows i + 1 .. n_r - 1
    return n


# ---------------------------------------------------------------------------
# Forward decomposition
# ---------------------------------------------------------------------------


def givens_decompose(V: np.ndarray) -> GivensAngles:
    """Decompose V into Givens angles.

    Args:
        V:  complex array of shape (..., N_r, N_c) with N_c ≤ N_r and
            orthonormal columns (V^H V = I_{N_c}).

    Returns:
        GivensAngles populated with φ ∈ (-π, π] and ψ ∈ [0, π/2].
    """
    V = np.asarray(V, dtype=np.complex128)
    *batch, n_r, n_c = V.shape
    if n_c > n_r:
        raise ValueError(f"N_c ({n_c}) must be ≤ N_r ({n_r}).")

    n_phi = n_phi_for_shape(n_r, n_c)
    n_psi = n_psi_for_shape(n_r, n_c)
    batch_shape = tuple(batch)

    phi = np.zeros(batch_shape + (n_phi,), dtype=float)
    psi = np.zeros(batch_shape + (n_psi,), dtype=float)
    phi_indices = np.zeros((n_phi, 2), dtype=int)
    psi_indices = np.zeros((n_psi, 2), dtype=int)
    schedule: list = []

    Vc = V.copy()
    p_i = 0  # next free φ slot
    s_i = 0  # next free ψ slot

    upper_i = min(n_c, n_r)
    for i in range(upper_i):
        # ---------- step 1: extract φ for rows l = i .. n_r - 1 ----------
        for l in range(i, n_r):
            ang = np.angle(Vc[..., l, i])  # array, shape = batch
            # Wrap to (-π, π]; np.angle already returns in this range.
            phi[..., p_i] = ang
            phi_indices[p_i] = (i, l)
            schedule.append(("phi", p_i))
            # Apply exp(-j ang) to row l of Vc, with broadcasting over batch.
            factor = np.exp(-1j * ang)
            Vc[..., l, :] = factor[..., None] * Vc[..., l, :]
            p_i += 1

        # ---------- step 2: Givens rotations, l = n_r - 1 down to i + 1 ----------
        for l in range(n_r - 1, i, -1):
            x_i = np.real(Vc[..., i, i])
            x_l = np.real(Vc[..., l, i])
            psi_val = np.arctan2(x_l, x_i)  # range (-π, π]; near zero in practice
            psi[..., s_i] = psi_val
            psi_indices[s_i] = (i, l)
            schedule.append(("psi", s_i))

            c = np.cos(psi_val)
            s = np.sin(psi_val)
            row_i = c[..., None] * Vc[..., i, :] + s[..., None] * Vc[..., l, :]
            row_l = -s[..., None] * Vc[..., i, :] + c[..., None] * Vc[..., l, :]
            Vc[..., i, :] = row_i
            Vc[..., l, :] = row_l
            s_i += 1

    return GivensAngles(
        phi=phi,
        psi=psi,
        n_r=n_r,
        n_c=n_c,
        phi_indices=phi_indices,
        psi_indices=psi_indices,
        schedule=schedule,
    )


# ---------------------------------------------------------------------------
# Inverse reconstruction
# ---------------------------------------------------------------------------


def givens_reconstruct(angles: GivensAngles) -> np.ndarray:
    """Reconstruct V from Givens angles.

    Args:
        angles: GivensAngles from `givens_decompose` (possibly after
                quantisation via `bfi.quantisation.quantise_angles`).

    Returns:
        V of shape (..., n_r, n_c).
    """
    n_r = angles.n_r
    n_c = angles.n_c
    phi = angles.phi
    psi = angles.psi
    phi_idx = angles.phi_indices
    psi_idx = angles.psi_indices

    batch_shape = phi.shape[:-1]
    # Initialise V = Ĩ_{n_r, n_c}
    V = np.zeros(batch_shape + (n_r, n_c), dtype=np.complex128)
    for c in range(n_c):
        V[..., c, c] = 1.0

    # Walk schedule in reverse, undoing each factor.
    for kind, k in reversed(angles.schedule):
        if kind == "phi":
            i, l = int(phi_idx[k, 0]), int(phi_idx[k, 1])
            ang = phi[..., k]
            # Forward applied exp(-j ang); inverse applies exp(+j ang).
            factor = np.exp(1j * ang)
            V[..., l, :] = factor[..., None] * V[..., l, :]
        else:  # 'psi'
            i, l = int(psi_idx[k, 0]), int(psi_idx[k, 1])
            ang = psi[..., k]
            c = np.cos(ang)
            s = np.sin(ang)
            # Forward applied G^T (rows i, l ← [+c, +s; -s, +c] · rows).
            # Inverse applies G        (rows i, l ← [+c, -s; +s, +c] · rows).
            row_i = c[..., None] * V[..., i, :] - s[..., None] * V[..., l, :]
            row_l = s[..., None] * V[..., i, :] + c[..., None] * V[..., l, :]
            V[..., i, :] = row_i
            V[..., l, :] = row_l

    return V


# ---------------------------------------------------------------------------
# Convenience: end-to-end round-trip with quantisation
# ---------------------------------------------------------------------------


def quantised_round_trip(
    V: np.ndarray, b_psi: int, b_phi: int
) -> Tuple[np.ndarray, GivensAngles]:
    """V → (φ, ψ) → quantise → (φ_q, ψ_q) → V_hat round trip.

    Args:
        V:      complex steering matrix, shape (..., n_r, n_c).
        b_psi:  ψ codebook bit-depth (≥ 0).
        b_phi:  φ codebook bit-depth (≥ 0).

    Returns:
        (V_hat, quantised_angles).
    """
    from .quantisation import quantise_angles

    angles = givens_decompose(V)
    psi_q, phi_q = quantise_angles(angles.psi, angles.phi, b_psi, b_phi)
    angles_q = GivensAngles(
        phi=phi_q,
        psi=psi_q,
        n_r=angles.n_r,
        n_c=angles.n_c,
        phi_indices=angles.phi_indices,
        psi_indices=angles.psi_indices,
        schedule=angles.schedule,
    )
    V_hat = givens_reconstruct(angles_q)
    return V_hat, angles_q
