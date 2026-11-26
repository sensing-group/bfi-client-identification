"""K-shot evaluation utilities for the BFI framework.

Public API:
    select_support_indices(y, K, rng)
    compute_prototypes(Z, y, support_idx_per_class)
    predict_by_cosine(Z_query, prototypes, prototype_labels)
    kshot_eval(Z_held, y_held, K, support_seed) -> (acc, n_query)
"""

from __future__ import annotations

import numpy as np

# Numerical-stability epsilon used when normalising prototypes.
PROTO_NORM_EPS = 1e-12


def select_support_indices(
    y: np.ndarray, K: int, rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """For each unique class, pick K support window indices.

    Sampling is without replacement when possible; falls back to with-
    replacement if a class has fewer than K windows. Returns a dict
    {class_label: array of K indices into y}.
    """
    out: dict[int, np.ndarray] = {}
    for label in np.unique(y):
        idxs = np.flatnonzero(y == label)
        if len(idxs) < K:
            chosen = rng.choice(idxs, size=K, replace=True)
        else:
            chosen = rng.choice(idxs, size=K, replace=False)
        out[int(label)] = chosen
    return out


def compute_prototypes(
    Z: np.ndarray,
    support_idx_per_class: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one L2-normalised prototype per class.

    Args:
        Z: (N, D) L2-normalised embeddings.
        support_idx_per_class: mapping label -> indices into Z.

    Returns:
        (P, labels) where P is (C, D) L2-normalised prototypes and
        labels is (C,) int array (same order as P rows).
    """
    proto_list = []
    labels = []
    for label, chosen in sorted(support_idx_per_class.items()):
        proto = Z[chosen].mean(axis=0)
        norm = np.linalg.norm(proto) + PROTO_NORM_EPS
        proto_list.append(proto / norm)
        labels.append(int(label))
    return np.stack(proto_list, axis=0), np.asarray(labels, dtype=int)


def predict_by_cosine(
    Z_query: np.ndarray,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
) -> np.ndarray:
    """Predict labels for queries by argmax cosine similarity to prototypes.

    Both inputs must be L2-normalised so that cosine similarity reduces
    to inner product. Returns int array of predicted labels.
    """
    sims = Z_query @ prototypes.T  # (M, C)
    return prototype_labels[np.argmax(sims, axis=1)]


def kshot_eval(
    Z_held: np.ndarray,
    y_held: np.ndarray,
    K: int,
    support_seed: int,
) -> tuple[float, int]:
    """K-shot prototype classification.

    For each held-out class, draw K random windows as the support set;
    the prototype is the L2-normalised mean of their embeddings. Predict
    the label of each remaining (query) window by argmax cosine similarity.

    Returns (top1_accuracy, n_query_windows).
    """
    rng = np.random.default_rng(support_seed)
    support_idx_per_class = select_support_indices(y_held, K, rng)

    # Aggregate all support indices into a single set for the query mask.
    support_set: set[int] = set()
    for chosen in support_idx_per_class.values():
        support_set.update(int(i) for i in chosen)

    query_idx = np.array(
        [i for i in range(len(y_held)) if i not in support_set],
        dtype=int,
    )
    if len(query_idx) == 0:
        return float("nan"), 0

    P, P_labels = compute_prototypes(Z_held, support_idx_per_class)
    pred = predict_by_cosine(Z_held[query_idx], P, P_labels)
    acc = float((pred == y_held[query_idx]).mean())
    return acc, int(len(query_idx))
