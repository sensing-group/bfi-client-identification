# `bfi/` — BFI privacy-utility research toolkit

Implements the IEEE 802.11 Givens decomposition, angle quantisation, and
mutual-information estimators needed for the BFI privacy-utility frontier
project (**Thread A** of the BFI research commitment, 2026-05-06).

See `docs/4-research/in-progress/bfi-privacy-research/MASTER_PLAN.md` for
the research framing.

---

## Modules

| File | Purpose |
|---|---|
| `givens.py`         | Forward / inverse Givens decomposition of the steering matrix V; round-trip exact (~1e-16) for tall and square V. |
| `quantisation.py`   | Deterministic midpoint quantiser + **DP-SQ stochastic quantiser** (Seif et al. 2025); 802.11ac/ax codebook presets; Lemma-1 closed-form MSE. |
| `mi_estimators.py`  | Joint Kraskov (KSG-Ross 2014), InfoNCE (numpy), MINE (optional torch). |
| `synthetic.py`      | Random unitary V + complex CSI generators for testing. |
| `datasets/beamsense.py`   | BeamSense (Northeastern + Padova) `.mat` corpus loader. |
| `datasets/internal_csi.py`| HSG ICV `dataset-release/` JSONL.gz loader. |
| `tests/`            | Round-trip + estimator + loader + DP-SQ tests (5 suites, all green). |
| `demo_pareto_synthetic.py`     | Synthetic data → privacy-utility curve. |
| `demo_pareto_beamsense.py`     | Privacy-utility on real or mock BeamSense. |
| `demo_pareto_internal.py`      | Privacy-utility on the HSG ICV CSI corpus. |
| `demo_pareto_dpsq_vs_det.py`   | **DP-SQ vs deterministic head-to-head**, paper figure precursor. |

---

## Quickstart

Round-trip and quantisation:

```python
import numpy as np
from bfi import givens_decompose, givens_reconstruct, quantise_angles
from bfi.synthetic import random_unitary_v

V = random_unitary_v(n_r=4, n_c=4)            # complex unitary, 4×4
angles = givens_decompose(V)                   # exact (φ, ψ)
psi_q, phi_q = quantise_angles(angles.psi, angles.phi, b_psi=4, b_phi=6)

from bfi.givens import GivensAngles
angles_q = GivensAngles(
    phi=phi_q, psi=psi_q,
    n_r=angles.n_r, n_c=angles.n_c,
    phi_indices=angles.phi_indices,
    psi_indices=angles.psi_indices,
    schedule=angles.schedule,
)
V_hat = givens_reconstruct(angles_q)
print("RMSE:", np.sqrt(np.mean(np.abs(V - V_hat) ** 2)))
```

End-to-end privacy-utility curve:

```bash
python3 bfi/demo_pareto_synthetic.py --n-samples 2000 --classes 4 --plot
# → /tmp/bfi_pareto_synth.png
```

---

## Tests

```bash
python3 bfi/tests/test_givens.py
python3 bfi/tests/test_mi_estimators.py
```

Both should print `All tests passed.`.

---

## Design notes

### Givens convention

The decomposition follows IEEE 802.11-2020 §19.3.12.3.6's QR-with-phase
peeling, with one harmless deviation: we extract one "redundant" diagonal
phase per column (on row N_r − 1) that the standard's compact form leaves
implicit. This phase is mathematically removable in the standard because
of the AP-side phase-recovery convention; for our research purposes we
keep it because it makes round-trip exact for square V (which the
standard's compact form does not handle losslessly).

The code is parameterised so that, for a future paper version, the exact
IEEE bit-allocation can be plugged in without restructuring the API.

### MI estimator triangulation

Two complementary estimators are exposed by default (`kraskov_mi` and
`infonce_mi`). MINE (`mine_mi`) is gated on `torch`. Per the master plan:
**reject results where the two primary estimators disagree by more than
2×**. Discrepancy is most likely in the high-MI regime; if your KSG
estimate sits near `log₂ K`, also report the InfoNCE estimate, which
saturates correctly at `log₂ K`.

### Reproducibility

All randomness is seeded; all estimators take a `seed` / `random_state`
argument with a documented default. The synthetic-data demo is fully
deterministic given `--seed`.

---

## Status (2026-05-06)

| Item | Status |
|---|---|
| Givens forward / inverse | ✅ exact round-trip for tall and square V |
| Angle quantisation       | ✅ midpoint-uniform + 802.11ac/ax SU/MU presets |
| **DP-SQ** (Seif+2025)    | ✅ implemented; empirical MSE matches Lemma 1 within ~5%; ε-DP under half-cell adjacency verified; head-to-head demo against deterministic. |
| Joint Kraskov KSG        | ✅ implemented, tested |
| InfoNCE                  | ✅ Gaussian-critic version implemented |
| MINE                     | ⏸ optional, gated on torch (not in `.venv`) |
| BeamSense loader         | ✅ walks `<root>/<env>/processed_dataset/<sta>/beamf_angles/`; parses .mat batches; activity + person labels; deterministic split (seed 111). Validated on mock fixture; real data TBD (180 GB, see master plan). |
| Internal CSI loader      | ✅ walks `dataset-release/sessions/`; loads complex CSI per (link, subcarrier) with presence + person_count labels; per-session and global link universes; complex-domain quantiser. Tested on real on-disk data (15 sessions, ~187 links, 52 subcarriers). |
| Privacy-utility curve    | ✅ end-to-end on synthetic, BeamSense mock, real internal CSI, and synthetic DP-SQ-vs-deterministic. **First quantitative finding**: at low b (≤ 3), DP-SQ at ε ≤ 1 reduces identifying-info leakage by ~50% vs deterministic; at b ≥ 4, the gap closes. |
