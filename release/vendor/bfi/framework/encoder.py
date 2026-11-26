"""Neural encoders for the BFI one-shot identification framework.

All encoders inherit from :class:`Encoder` and produce L2-normalised
embeddings of shape (B, embed_dim). The output normalisation is applied
in the base class via the ``forward`` template method, so subclasses
implement ``_features``.

Public API:
    Encoder            abstract base class
    MLPEncoder         2-layer MLP (matches the legacy demo_oneshot_attack)
    MLPv2Encoder       3-layer MLP with BatchNorm + Dropout (legacy "improved")
    IdentityEncoder    no-op (normalises raw features to unit length)
    build_encoder      factory: encoder='mlp' | 'mlpv2' | 'identity'
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HIDDEN_DIM = 256
DEFAULT_EMBED_DIM = 128
DEFAULT_DROPOUT_P = 0.2


class Encoder(nn.Module):
    """Abstract base: subclasses implement ``_features``; ``forward``
    L2-normalises the result so cosine similarity reduces to inner product.

    Public attributes:
        in_dim:    input feature dimension.
        embed_dim: output embedding dimension.
    """

    def __init__(self, in_dim: int, embed_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._features(x), dim=-1)

    @torch.no_grad()
    def embed_numpy(self, X: np.ndarray, device: torch.device | None = None) -> np.ndarray:
        """Convenience: embed a numpy matrix and return numpy."""
        if device is None:
            device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            t = torch.as_tensor(X, dtype=torch.float32, device=device)
            z = self.forward(t)
            return z.cpu().numpy()
        finally:
            if was_training:
                self.train()


class MLPEncoder(Encoder):
    """Two-layer MLP encoder: in_dim -> hidden -> embed (L2-normalised).

    Matches ``demo_oneshot_attack.Encoder`` (paper headline architecture).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        embed_dim: int = DEFAULT_EMBED_DIM,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPv2Encoder(Encoder):
    """Three-layer MLP with BatchNorm + Dropout regularisation.

    Matches ``demo_kshot_attack.ImprovedEncoder``. Requires batch_size >= 2
    during training (BatchNorm1d).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        embed_dim: int = DEFAULT_EMBED_DIM,
        dropout_p: float = DEFAULT_DROPOUT_P,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout_p)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, embed_dim),
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IdentityEncoder(Encoder):
    """No-op encoder that just L2-normalises raw features.

    Useful as a baseline (cosine 1-NN on raw features).
    """

    def __init__(self, in_dim: int):
        super().__init__(in_dim=in_dim, embed_dim=in_dim)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_encoder(
    spec: str | nn.Module,
    in_dim: int,
    *,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    embed_dim: int = DEFAULT_EMBED_DIM,
    dropout_p: float = DEFAULT_DROPOUT_P,
) -> Encoder:
    """Construct an encoder from a string spec or pass-through Module.

    Args:
        spec:     'mlp' | 'mlpv2' | 'identity', or a pre-built nn.Module.
        in_dim:   input feature dimension (ignored if ``spec`` is a Module).
        hidden_dim, embed_dim, dropout_p: architecture knobs.
    """
    if isinstance(spec, nn.Module):
        return spec  # type: ignore[return-value]
    if spec == "mlp":
        return MLPEncoder(in_dim, hidden_dim=hidden_dim, embed_dim=embed_dim)
    if spec == "mlpv2":
        return MLPv2Encoder(
            in_dim, hidden_dim=hidden_dim, embed_dim=embed_dim,
            dropout_p=dropout_p,
        )
    if spec == "identity":
        return IdentityEncoder(in_dim)
    raise ValueError(f"unknown encoder spec: {spec!r}")
