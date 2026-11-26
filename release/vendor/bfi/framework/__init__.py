"""bfi.framework — reusable one-shot identification framework for BFI.

Public API:

    from bfi.framework import (
        OneShotAttacker, ClosedSetAttacker,
        dp_sq, deterministic, structural,
        kshot_eval, compute_prototypes, predict_by_cosine,
    )
    from bfi.framework.datasets import (
        load_ntufi, load_he_location, load_he_presence, Substrate,
    )
    from bfi.framework.cohort_split import (
        even_odd_split, lowest_n_split, random_split, closed_set_split,
    )

Design overview: see ``bfi/framework/README.md``.
"""

from .attacker import OneShotAttacker
from .baselines import ClosedSetAttacker
from .defenses import deterministic, dp_sq, structural
from .encoder import (
    Encoder,
    IdentityEncoder,
    MLPEncoder,
    MLPv2Encoder,
    build_encoder,
)
from .inference import (
    compute_prototypes,
    kshot_eval,
    predict_by_cosine,
    select_support_indices,
)
from .losses import supcon_loss, supcon_simple_loss, triplet_loss
from .quantize import (
    dp_sq_quantise_1d,
    midpoint_quantise_1d,
    quantise_rows,
)

__all__ = [
    # High-level
    "OneShotAttacker",
    "ClosedSetAttacker",
    # Defenses
    "deterministic",
    "dp_sq",
    "structural",
    # Encoders
    "Encoder",
    "MLPEncoder",
    "MLPv2Encoder",
    "IdentityEncoder",
    "build_encoder",
    # Losses
    "supcon_loss",
    "supcon_simple_loss",
    "triplet_loss",
    # Inference
    "kshot_eval",
    "select_support_indices",
    "compute_prototypes",
    "predict_by_cosine",
    # Quantization primitives
    "midpoint_quantise_1d",
    "dp_sq_quantise_1d",
    "quantise_rows",
]
