"""Closed-set baseline attackers for the BFI framework.

Provides a single :class:`ClosedSetAttacker` with three methods:

    - '1nn': 1-NN with euclidean metric on raw features (no scaling).
    - 'lr':  Logistic Regression on standardised features.
    - 'mlp': sklearn MLPClassifier (one hidden layer) on standardised features.

This wraps the per-attacker scikit-learn knobs used by
``demo_learned_attacker_he.py`` and ``demo_dim_penalty_he.py``. All three
attackers share the same fit / predict / accuracy interface so the demos
can iterate uniformly.

Public API:
    ClosedSetAttacker(method='1nn'|'lr'|'mlp', seed=...)
    .fit(X, y) -> self
    .predict(X) -> array of int labels
    .accuracy(X, y) -> float

Convention: 1-NN does NOT standardise (matches both legacy scripts);
LR/MLP fit a StandardScaler on training features and apply it at predict.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants (no magic numbers buried in function bodies).
# ---------------------------------------------------------------------------

# 1-NN
KNN_N_NEIGHBORS = 1
KNN_METRIC = "euclidean"

# Logistic regression
LR_C = 1.0
LR_MAX_ITER = 1000
LR_SOLVER = "lbfgs"
LR_N_JOBS = 1

# MLP (sklearn classifier; 64 hidden units, one layer)
MLP_HIDDEN_LAYER_SIZES = (64,)
MLP_MAX_ITER = 500


ClosedSetMethod = Literal["1nn", "lr", "mlp"]
SUPPORTED_METHODS: tuple[ClosedSetMethod, ...] = ("1nn", "lr", "mlp")


class ClosedSetAttacker:
    """Closed-set classification attacker (1-NN, LR, or MLP).

    Args:
        method: '1nn' | 'lr' | 'mlp'.
        seed:   random seed used by LR / MLP (1-NN is deterministic).

    The attacker holds an internal :class:`StandardScaler` for LR and
    MLP. 1-NN operates on raw features.
    """

    def __init__(self, method: ClosedSetMethod = "1nn", *, seed: int = 0):
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {SUPPORTED_METHODS}; got {method!r}"
            )
        self.method = method
        self.seed = int(seed)
        self._clf = None
        self._scaler: StandardScaler | None = None

    # ------------------------------------------------------------------
    # Internal: build the underlying sklearn estimator.
    # ------------------------------------------------------------------

    def _make_estimator(self):
        if self.method == "1nn":
            return KNeighborsClassifier(
                n_neighbors=KNN_N_NEIGHBORS, metric=KNN_METRIC,
            )
        if self.method == "lr":
            return LogisticRegression(
                C=LR_C, max_iter=LR_MAX_ITER,
                solver=LR_SOLVER, random_state=self.seed, n_jobs=LR_N_JOBS,
            )
        if self.method == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                max_iter=MLP_MAX_ITER, random_state=self.seed,
            )
        raise AssertionError(f"unhandled method {self.method!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClosedSetAttacker":
        X = np.asarray(X)
        y = np.asarray(y)
        self._clf = self._make_estimator()
        if self.method == "1nn":
            self._scaler = None
            self._clf.fit(X, y)
        else:
            self._scaler = StandardScaler()
            X_s = self._scaler.fit_transform(X)
            self._clf.fit(X_s, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("ClosedSetAttacker not fit; call fit() first")
        X = np.asarray(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._clf.predict(X)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())
