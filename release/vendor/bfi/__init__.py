"""BFI (Beamforming Feedback Information) research toolkit.

Modules:
    givens          IEEE 802.11 Givens decomposition (forward / inverse / quantise).
    quantisation    Angle quantisation with standard codebook presets.
    mi_estimators   Mutual-information estimators (Kraskov / InfoNCE / MINE).
    synthetic       Synthetic CSI generators for round-trip testing.

This package is for the BFI privacy-utility frontier project (Thread A
in `docs/4-research/in-progress/bfi-privacy-research/MASTER_PLAN.md`).
"""

from .givens import givens_decompose, givens_reconstruct, GivensAngles
from .quantisation import (
    BFIPreset,
    PRESET_802_11AC_SU,
    PRESET_802_11AC_MU,
    PRESET_802_11AX_SU,
    PRESET_802_11AX_MU,
    quantise_angles,
    dp_sq_quantise_angles,
    dp_sq_mse,
    dp_sq_kappa,
)

__all__ = [
    "givens_decompose",
    "givens_reconstruct",
    "GivensAngles",
    "BFIPreset",
    "PRESET_802_11AC_SU",
    "PRESET_802_11AC_MU",
    "PRESET_802_11AX_SU",
    "PRESET_802_11AX_MU",
    "quantise_angles",
    "dp_sq_quantise_angles",
    "dp_sq_mse",
    "dp_sq_kappa",
]
