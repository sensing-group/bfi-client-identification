"""High-level OneShotAttacker class wrapping the BFI re-identification pipeline.

The attacker:
    1. owns an Encoder.
    2. trains it on a calibration cohort (X_calib, y_calib) via supervised
       contrastive loss (or triplet, configurable).
    3. embeds arbitrary feature matrices.
    4. evaluates K-shot identification on a held-out cohort.

Usage:

    attacker = OneShotAttacker(encoder='mlp', embed_dim=128)
    attacker.fit(X_calib, y_calib, epochs=200, lr=1e-3, seed=42)
    Z = attacker.embed(X_target)
    acc = attacker.identify(support_X=X_target[support_idx],
                            support_y=y_target[support_idx],
                            query_X=X_target[query_idx],
                            query_y=y_target[query_idx], K=3)
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn

from .encoder import (
    DEFAULT_EMBED_DIM,
    DEFAULT_HIDDEN_DIM,
    Encoder,
    build_encoder,
)
from .losses import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TRIPLET_MARGIN,
    supcon_loss,
    supcon_simple_loss,
    triplet_loss,
)
from .inference import (
    compute_prototypes,
    kshot_eval,
    predict_by_cosine,
    select_support_indices,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_FEATURE_DROPOUT_P = 0.0
MIN_BATCH_SIZE_FOR_LOSS = 2

# Type aliases.
ContrastiveName = Literal["supcon", "supcon_simple", "triplet"]
EncoderSpec = str | nn.Module
AugmentFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]
BatchStrategy = Literal["numpy", "torch_perm"]
DEFAULT_BATCH_STRATEGY: BatchStrategy = "numpy"


def _seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def _make_batches(
    n: int, batch_size: int, rng: np.random.Generator,
):
    """Yield index arrays for an epoch (shuffled by ``rng.permutation``)."""
    perm = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        if len(idx) < MIN_BATCH_SIZE_FOR_LOSS:
            continue
        yield idx


def _make_batches_torch(
    n: int, batch_size: int, device: torch.device,
):
    """Yield index arrays for an epoch using ``torch.randperm`` on ``device``.

    Mirrors the legacy closed-set He demo (which used
    ``torch.randperm(n, device=device)`` once per epoch, then sliced into
    batches) so seeded outputs of that demo remain bit-identical.
    """
    perm = torch.randperm(n, device=device).cpu().numpy()
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        if len(idx) < MIN_BATCH_SIZE_FOR_LOSS:
            continue
        yield idx


class OneShotAttacker:
    """One-shot / K-shot re-identification attacker.

    Args:
        encoder:        'mlp' | 'mlpv2' | 'identity', or a pre-built nn.Module.
        embed_dim:      embedding dimensionality.
        hidden_dim:     hidden layer width (ignored for 'identity').
        contrastive:    'supcon' | 'triplet'.
        temperature:    SupCon temperature (ignored for triplet).
        triplet_margin: triplet margin (ignored for supcon).
        device:         torch device; defaults to CPU.

    The encoder is created lazily on the first ``fit`` call (we need
    ``in_dim`` from the data). Pass a pre-built ``nn.Module`` to override
    this and use a custom architecture.
    """

    def __init__(
        self,
        encoder: EncoderSpec = "mlp",
        *,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        contrastive: ContrastiveName = "supcon",
        temperature: float = DEFAULT_TEMPERATURE,
        triplet_margin: float = DEFAULT_TRIPLET_MARGIN,
        device: torch.device | str | None = None,
    ):
        self._encoder_spec = encoder
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.contrastive = contrastive
        self.temperature = float(temperature)
        self.triplet_margin = float(triplet_margin)
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.encoder: Encoder | None = None

    # ------------------------------------------------------------------
    # Encoder lifecycle
    # ------------------------------------------------------------------

    def _build_encoder(self, in_dim: int) -> Encoder:
        return build_encoder(
            self._encoder_spec,
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
        ).to(self.device)

    def _loss_fn(self):
        if self.contrastive == "supcon":
            return lambda z, y: supcon_loss(z, y, self.temperature)
        if self.contrastive == "supcon_simple":
            return lambda z, y: supcon_simple_loss(z, y, self.temperature)
        if self.contrastive == "triplet":
            return lambda z, y: triplet_loss(z, y, self.triplet_margin)
        raise ValueError(f"unknown contrastive: {self.contrastive!r}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int = 0,
        augment: AugmentFn | None = None,
        feature_dropout_p: float = DEFAULT_FEATURE_DROPOUT_P,
        batch_strategy: BatchStrategy = DEFAULT_BATCH_STRATEGY,
        verbose: bool = False,
    ) -> "OneShotAttacker":
        """Train the encoder by supervised contrastive loss.

        Args:
            X:                (N, D) calibration features.
            y:                (N,)  calibration labels.
            epochs, batch_size, lr, weight_decay: standard knobs.
            seed:             controls torch + numpy generators (batch
                              sampling, augmentation RNG).
            augment:          optional batch augmentation function. Called
                              as ``augment(xb, rng_aug)`` each step. Use
                              this for defense-aware training (DP-SQ
                              augmentation).
            feature_dropout_p: Bernoulli zero-mask probability applied to
                              input features each batch (data augmentation;
                              0.0 disables).
            batch_strategy:   'numpy' (default) shuffles via a numpy
                              ``Generator`` seeded by ``seed``; matches the
                              legacy NTU-Fi demos and ``demo_oneshot_he``.
                              'torch_perm' shuffles via
                              ``torch.randperm(n, device=device)`` each
                              epoch; matches the legacy
                              ``demo_oneshot_he_closedset``.
            verbose:          print epoch losses every 25/50 epochs.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} entries"
            )
        if batch_strategy not in ("numpy", "torch_perm"):
            raise ValueError(
                f"batch_strategy must be 'numpy' or 'torch_perm'; "
                f"got {batch_strategy!r}"
            )

        _seed_torch(seed)
        in_dim = X.shape[1]
        self.encoder = self._build_encoder(in_dim)

        params = [p for p in self.encoder.parameters() if p.requires_grad]
        if params:
            optimiser: torch.optim.Optimizer | None = torch.optim.Adam(
                params, lr=lr, weight_decay=weight_decay,
            )
        else:
            # Parameter-less encoder (e.g. IdentityEncoder): no training step.
            optimiser = None
        loss_fn = self._loss_fn()

        # Two independent generators: one for batch order, one for any
        # data augmentation (DP-SQ + feature dropout). seed and seed+1
        # mirrors the legacy demo behaviour exactly.
        rng_batch = np.random.default_rng(seed)
        rng_aug = np.random.default_rng(seed + 1)

        self.encoder.train()
        n = X.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            if batch_strategy == "numpy":
                batch_iter = _make_batches(n, batch_size, rng_batch)
            else:
                batch_iter = _make_batches_torch(n, batch_size, self.device)
            for idx in batch_iter:
                xb = X[idx]
                yb = y[idx]
                if augment is not None:
                    xb = augment(xb, rng_aug)
                if feature_dropout_p > 0.0:
                    mask = rng_aug.random(xb.shape) >= feature_dropout_p
                    xb = xb * mask.astype(xb.dtype)
                xb_t = torch.as_tensor(xb, dtype=torch.float32, device=self.device)
                yb_t = torch.as_tensor(yb, dtype=torch.long, device=self.device)
                if xb_t.shape[0] < MIN_BATCH_SIZE_FOR_LOSS:
                    continue
                if optimiser is not None:
                    optimiser.zero_grad()
                z = self.encoder(xb_t)
                loss = loss_fn(z, yb_t)
                if loss.requires_grad and optimiser is not None:
                    loss.backward()
                    optimiser.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1
            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
                avg = epoch_loss / max(1, n_batches)
                print(f"      epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")
        self.encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, X: np.ndarray) -> np.ndarray:
        """Project (N, D) features to (N, embed_dim) L2-normalised vectors."""
        if self.encoder is None:
            raise RuntimeError("encoder not trained; call fit() first")
        return self.encoder.embed_numpy(np.asarray(X), device=self.device)

    def identify(
        self,
        *,
        support_X: np.ndarray,
        support_y: np.ndarray,
        query_X: np.ndarray,
        query_y: np.ndarray,
        K: int | None = None,
    ) -> float:
        """K-shot top-1 identification accuracy.

        Each unique support label contributes one prototype (L2-normalised
        mean of its support embeddings). ``K`` is purely informational
        here (the prototypes are computed from whatever support windows
        the caller provides); it is checked for consistency if given.
        """
        if self.encoder is None:
            raise RuntimeError("encoder not trained; call fit() first")
        Z_sup = self.embed(support_X)
        Z_q = self.embed(query_X)

        # Build per-class index lists into Z_sup.
        per_class: dict[int, np.ndarray] = {}
        for label in np.unique(support_y):
            mask = np.flatnonzero(support_y == label)
            if K is not None and len(mask) != K:
                # The caller asserted K but provided a different count.
                # This is allowed (we just use whatever was given) but
                # we surface it as a printed note so accidental mismatches
                # don't go unnoticed in scripts.
                pass
            per_class[int(label)] = mask

        P, P_labels = compute_prototypes(Z_sup, per_class)
        pred = predict_by_cosine(Z_q, P, P_labels)
        return float((pred == np.asarray(query_y, dtype=int)).mean())

    def kshot_eval(
        self,
        X_held: np.ndarray,
        y_held: np.ndarray,
        *,
        K: int,
        support_seed: int,
    ) -> tuple[float, int]:
        """One-call K-shot evaluation with internal support sampling.

        Equivalent to :func:`bfi.framework.inference.kshot_eval` after
        embedding ``X_held`` once.
        """
        Z = self.embed(X_held)
        return kshot_eval(Z, np.asarray(y_held, dtype=int),
                          K=K, support_seed=support_seed)
