"""Contrastive loss functions for the BFI framework.

Implementations match those in the legacy demo scripts (so seeded outputs
remain bit-identical):

    supcon_loss          Supervised contrastive (Khosla et al., NeurIPS 2020)
                         — matches demo_oneshot_attack/demo_kshot_attack/
                         demo_oneshot_he (the "stable-subtracted" SupCon-out
                         variant).
    supcon_simple_loss   Lighter variant used by the closed-set He demo:
                         masks the self-similarity with a large negative
                         constant in the logits and uses
                         ``F.log_softmax`` directly. Mathematically
                         equivalent in the limit but numerically distinct;
                         exposed as a separate name so demos that
                         pre-date the framework can be migrated without
                         drifting the CSV outputs.
    triplet_loss         Margin-based triplet (random positive + hardest negative).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 0.1
DEFAULT_TRIPLET_MARGIN = 0.2

# Numerical-stability epsilon used in log denominators.
LOG_EPS = 1e-12

# Large negative value used to mask self-similarity in supcon_simple_loss
# (matches the legacy closed-set He demo).
SUPCON_SIMPLE_SELF_MASK_VALUE = -1e9


def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Supervised contrastive loss (SupCon-out, Khosla et al., 2020).

    Args:
        features:    (B, D) L2-normalised embeddings.
        labels:      (B,)  integer labels.
        temperature: scalar temperature tau (default 0.1).

    For each anchor i, positives are j != i with y_j = y_i; loss is the
    mean log-prob over positives, then mean over anchors with at least
    one positive.
    """
    device = features.device
    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

    logits = torch.matmul(features, features.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    self_mask = torch.eye(batch_size, device=device)
    mask = (mask - self_mask).clamp(min=0.0)

    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + LOG_EPS)

    n_pos = mask.sum(dim=1)
    has_pos = n_pos > 0
    if has_pos.sum() == 0:
        return torch.zeros((), device=device, requires_grad=True)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[has_pos] / n_pos[has_pos]
    return -mean_log_prob_pos.mean()


def supcon_simple_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Lighter SupCon variant matching the legacy closed-set He demo.

    Identical mathematical objective to :func:`supcon_loss` in the limit,
    but constructed slightly differently:

        1. Masks the self-similarity by adding ``SUPCON_SIMPLE_SELF_MASK_VALUE``
           (= -1e9) to the diagonal of the (B, B) similarity matrix
           BEFORE the row-wise log-softmax.
        2. Applies ``F.log_softmax`` directly (no max-subtraction in
           Python; ``log_softmax`` does this internally).
        3. Counts positives via ``pos_mask.float().sum(dim=1).clamp(min=1)``
           rather than masking out anchors with no positives.
        4. Averages over ALL anchors (including those with no positives,
           which contribute 0 thanks to the ``pos_count.clamp(min=1)``
           denominator) instead of filtering ``[has_pos]``.

    Numerically distinct from :func:`supcon_loss`; producing byte-identical
    output for the legacy ``demo_oneshot_he_closedset`` requires this
    variant.
    """
    bs = features.shape[0]
    sim = features @ features.T / temperature
    mask_self = torch.eye(bs, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask_self, SUPCON_SIMPLE_SELF_MASK_VALUE)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & ~mask_self
    log_prob = F.log_softmax(sim, dim=1)
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
    return loss.mean()


def triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = DEFAULT_TRIPLET_MARGIN,
) -> torch.Tensor:
    """Margin-based triplet loss with hardest-negative mining.

    Args:
        features:    (B, D) L2-normalised embeddings.
        labels:      (B,)  integer labels.
        margin:      additive margin between positive and negative cosine
                     distances (default 0.2).

    For each anchor i, picks the hardest negative (smallest cosine
    distance among j with y_j != y_i) and any positive (largest cosine
    distance among j with y_j = y_i, j != i). Loss is the hinge over
    (d_pos - d_neg + margin).
    """
    device = features.device
    sim = torch.matmul(features, features.T)              # cosine sim
    dist = 1.0 - sim                                      # cosine distance
    bs = features.shape[0]
    eye = torch.eye(bs, device=device, dtype=torch.bool)

    labels = labels.contiguous().view(-1, 1)
    same = labels == labels.T
    pos_mask = same & ~eye
    neg_mask = ~same

    inf = torch.full_like(dist, float("inf"))
    pos_dist = torch.where(pos_mask, dist, -inf)          # max for hardest pos
    neg_dist = torch.where(neg_mask, dist, inf)           # min for hardest neg

    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid = has_pos & has_neg
    if not valid.any():
        return torch.zeros((), device=device, requires_grad=True)

    d_pos = pos_dist[valid].max(dim=1).values
    d_neg = neg_dist[valid].min(dim=1).values
    return F.relu(d_pos - d_neg + margin).mean()
