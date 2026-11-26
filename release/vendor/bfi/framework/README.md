# `bfi.framework` — One-shot identification framework for BFI

A small, opinionated package for running contrastive one-shot / K-shot
identification attacks against BFI (Beamforming Feedback Information)
substrates with quantization-based privacy defenses. Replaces the
encoder + loss + inference + dataset code that was duplicated across
`bfi/demo_oneshot_attack.py`, `bfi/demo_kshot_attack.py`,
`bfi/demo_oneshot_he*.py`, and the He learned-attacker demos.

## Design overview

```
bfi/framework/
├── __init__.py           # public API re-exports
├── attacker.py           # OneShotAttacker (high-level orchestrator)
├── baselines.py          # ClosedSetAttacker (1-NN, LR, MLP for closed-set demos)
├── encoder.py            # MLPEncoder, MLPv2Encoder, IdentityEncoder
├── losses.py             # supcon_loss, supcon_simple_loss, triplet_loss
├── inference.py          # kshot_eval, compute_prototypes, predict_by_cosine
├── defenses.py           # deterministic, dp_sq, structural (stateless funcs)
├── quantize.py           # midpoint_quantise_1d, dp_sq_quantise_1d, quantise_rows
├── cohort_split.py       # even_odd_split, lowest_n_split, random_split, closed_set_split
├── datasets/             # uniform Substrate loaders for the three corpora
│   ├── substrate.py      #   Substrate dataclass
│   ├── ntufi.py          #   load_ntufi
│   ├── he_location.py    #   load_he_location
│   └── he_presence.py    #   load_he_presence
├── tests/                # pytest suite
└── verify_paper_numbers.py
```

Three guiding principles:

1. **Stateless defenses.** `dp_sq(X, b, eps, rng)`, `deterministic(X, b)`,
   `structural(X, rng)` are functions, not objects. RNG is always
   explicit. Easy to compose (`dp_sq(structural(X, rng_s), b, eps, rng_q)`).
2. **One Encoder protocol.** All encoders subclass `Encoder` and emit
   L2-normalised vectors so cosine similarity reduces to inner product.
3. **Reproducible.** Every random source is seeded (`OneShotAttacker.fit(seed=...)`,
   `np.random.default_rng(seed)`). `numpy.random.default_rng` is preferred
   over the legacy `RandomState`.

## Quickstart

```python
from bfi.framework import OneShotAttacker, deterministic, dp_sq
from bfi.framework.cohort_split import even_odd_split
from bfi.framework.datasets import load_ntufi
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load and split a substrate.
sub = load_ntufi()                        # uses BFI_NTUFI_ROOT env var
X_calib, y_calib, X_held, y_held = even_odd_split(sub.X, sub.y)

# 2. Standardise (defender pipeline).
scaler = StandardScaler()
X_calib_s = scaler.fit_transform(X_calib)
X_held_s = scaler.transform(X_held)

# 3. Apply the test-time defense.
rng = np.random.default_rng(42)
X_held_def = dp_sq(X_held_s, b=4, eps=0.5, rng=rng,
                   lo=float(X_calib_s.min()), hi=float(X_calib_s.max()),
                   circular=sub.circular)

# 4. Build, train, embed, evaluate.
attacker = OneShotAttacker(encoder="mlp", embed_dim=128, hidden_dim=256)
attacker.fit(X_calib_s, y_calib, epochs=200, lr=1e-3, seed=42)
acc, n_q = attacker.kshot_eval(X_held_def, y_held, K=1, support_seed=1)
print(f"K=1 attacker accuracy: {acc:.3f} (n_query={n_q})")
```

### Defense-aware training

```python
def dpsq_aug(xb, rng_aug):
    return dp_sq(xb, b=4, eps=0.5, rng=rng_aug, lo=lo, hi=hi)

attacker.fit(X_calib_s, y_calib, epochs=200, seed=42, augment=dpsq_aug)
```

### Custom encoder

```python
import torch.nn as nn

class MyEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Linear(in_dim, embed_dim)
    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), dim=-1)

attacker = OneShotAttacker(encoder=MyEncoder(in_dim=87, embed_dim=64))
```

## Migrating from the demo scripts

| Legacy code in demo                        | Framework call                                     |
|--------------------------------------------|----------------------------------------------------|
| `Encoder` (2-layer MLP)                    | `MLPEncoder` or `OneShotAttacker(encoder='mlp')`   |
| `ImprovedEncoder` (3-layer + BN + dropout) | `MLPv2Encoder` or `OneShotAttacker(encoder='mlpv2')` |
| `supcon_loss(z, y, tau)` (stable variant)  | `bfi.framework.supcon_loss`                        |
| `supcon_loss(z, y, tau)` (closed-set He variant) | `bfi.framework.supcon_simple_loss` or `OneShotAttacker(contrastive='supcon_simple')` |
| `quantise_arr(X, b, lo, hi, eps=0, rng=None)` | `deterministic(X, b, lo, hi)`                   |
| `quantise_arr(X, b, lo, hi, eps, rng)`     | `dp_sq(X, b, eps, rng, lo, hi, circular=...)`      |
| `quantise_arr_circular(...)`               | `dp_sq(X, ..., circular=True)`                     |
| `kshot_topk_acc(Z, y, K, sup_seed)`        | `kshot_eval(Z, y, K, sup_seed)`                    |
| `train_encoder(...)` + `embed(enc, X)`     | `OneShotAttacker.fit(...)` + `.embed(X)`           |
| `train_encoder` with `torch.randperm` shuffle | `OneShotAttacker.fit(..., batch_strategy='torch_perm')` |
| `KNeighborsClassifier(n_neighbors=1)`      | `ClosedSetAttacker(method='1nn')`                  |
| `LogisticRegression(...)` + StandardScaler | `ClosedSetAttacker(method='lr', seed=...)`         |
| `MLPClassifier(...)` + StandardScaler      | `ClosedSetAttacker(method='mlp', seed=...)`        |
| `collect_split` + `split_calib_heldout`    | `load_ntufi(...)` + `even_odd_split(X, y)`         |
| `cohort_split(unique, n_calib)` (He)       | `lowest_n_split(X, y, n_calib)`                    |
| Hardcoded `HE_DATA_ROOT`                   | `BFI_HE_LOCATION_ROOT` env var                     |

`bfi/demo_oneshot_attack.py`, `bfi/demo_kshot_attack.py`,
`bfi/demo_oneshot_he.py`, `bfi/demo_oneshot_he_closedset.py`,
`bfi/demo_learned_attacker_he.py`, and `bfi/demo_dim_penalty_he.py`
are all framework clients. Their CSV outputs are bit-identical to the
pre-refactor versions (kept at `bfi/legacy/`) on the same seed; we
verified this on tile combinations of the b/K grid.

## Path resolution

Each loader resolves its data root in this order:

1. Explicit `data_root=...` argument.
2. Environment variable (`BFI_NTUFI_ROOT`, `BFI_HE_LOCATION_ROOT`,
   `BFI_HE_PRESENCE_ROOT`).
3. A built-in default.

The defaults are convenient for the lab machine; on other hosts set the
env vars or pass `data_root=` explicitly.

## Tests

```
.venv/bin/python3 -m pytest bfi/framework/tests/
```

Suite covers:
- Defenses: deterministic round-trip, DP-SQ at high eps ≈ deterministic,
  DP-SQ circular wrap, structural orthogonality.
- Encoder: output shapes, L2 normalisation, factory dispatch.
- Inference: K-shot accuracy is 100% on perfectly-separable toy data.
- Cohort splits: disjoint partitions, seed reproducibility.
- Smoke integration: OneShotAttacker on synthetic clusters reaches
  > 2x chance.

## Verification

```
.venv/bin/python3 bfi/framework/verify_paper_numbers.py
```

Trains short (3 enc seeds x 3 support seeds, 200 epochs) attackers on
NTU-Fi HumanID at the headline cell (K=1 and K=3, b=4, eps=0.5, DP-SQ)
and asserts the mean accuracy lies in a seed-stable expected range.
Use `--epochs 50` for a quick smoke pass.
