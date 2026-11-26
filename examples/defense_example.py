"""A user-written defense for the test bench (scripts/defense_bench.py).

`Dither` is client-side noise: before quantising, the client adds to every
Givens angle a uniform offset `cells` quantisation cells wide (half a cell by
default), wraps phi and clips psi back into range, and reports the result at
the codebook of the run. It keeps no secret, so the access point rebuilds V
from the reported angles (the bench's default `recover`).

The file uses only the bench's public interface (`Defense`, `cell_widths`) and
the angle helpers of joint_defense_common.

Run it from the repository root:
    python3 scripts/defense_bench.py --defense examples/defense_example.py:Dither
    python3 scripts/defense_bench.py --defense examples/defense_example.py:Dither --param cells=2 --bits 6/4
"""
from __future__ import annotations

from typing import Any

import numpy as np

import joint_defense_common as C
from defense_bench import Angles, Defense, cell_widths

DEFAULT_CELLS = 0.5                       # width of the dither, in quantisation cells
CENTRED_UNIFORM = (-0.5, 0.5)             # unit-width uniform draw, scaled by the dither width


class Dither(Defense):
    """Uniform dither of `cells` quantisation cells on every angle, then the codebook."""

    def __init__(self, cells: float = DEFAULT_CELLS):
        self.cells = float(cells)
        self.name = f"dither_c{self.cells:g}"

    def report(self, angles: Angles, targets: np.ndarray,
               rng: np.random.Generator) -> tuple[Angles, Any]:
        phi, psi = angles
        w_phi, w_psi = cell_widths()
        phi = phi + self.cells * w_phi * rng.uniform(*CENTRED_UNIFORM, phi.shape)
        psi = psi + self.cells * w_psi * rng.uniform(*CENTRED_UNIFORM, psi.shape)
        phi = np.mod(phi - C.PHI_LO, C.PHI_HI - C.PHI_LO) + C.PHI_LO
        return C.he_quantise(phi, np.clip(psi, C.PSI_LO, C.PSI_HI)), None
