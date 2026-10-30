"""
LEARNING-TO-RANK reranker  --  a pairwise linear model, written from scratch.

WHY PAIRWISE
------------
Ranking is not classification.  We do not care whether the model predicts 0.9
or 0.4 for a relevant document; we only care that it scores the relevant
document ABOVE the irrelevant ones.  So instead of fitting labels, we fit
*comparisons* -- this is RankNet's objective:

  for every (relevant d+, irrelevant d-) pair of the same query:
        maximise  sigma( w.x(d+) - w.x(d-) )

  loss  = -log sigma( w.(x+ - x-) )
  grad  = -(1 - sigma(w.(x+ - x-))) * (x+ - x-)

which is exactly logistic regression on FEATURE DIFFERENCES, with no intercept
(an intercept would cancel in the difference anyway).  Trained with plain
gradient descent and L2 regularisation -- about 15 lines of numpy.

WHY IT IS IN THE PROJECT
------------------------
It is the honest middle rung of the cascade: a model that genuinely learns from
the patent data, costs microseconds, and whose learned weights we can show as a
bar chart.  It also sets the bar the cross-encoder must clear to justify being
~1000x slower.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES


class PairwiseLTR:
    name = "ltr"

    def __init__(self, l2: float = 1e-3, lr: float = 0.5, epochs: int = 400) -> None:
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.weights: np.ndarray | None = None
        self.history: list[float] = []
        self.feature_names = list(FEATURE_NAMES)

    # ------------------------------------------------------------------ train #
    def fit(self, pos: np.ndarray, neg: np.ndarray, verbose: bool = False) -> "PairwiseLTR":
        """`pos` and `neg` are aligned (n_pairs, n_features) matrices: row i of
        each is a relevant / irrelevant document for the same query."""
        diff = (pos - neg).astype(np.float64)
        n, d = diff.shape
        w = np.zeros(d, dtype=np.float64)
        self.history = []

        for _ in range(self.epochs):
            z = diff @ w
            # sigma(-z) is the gradient weight; computed stably.
            sig = 1.0 / (1.0 + np.exp(np.clip(z, -30, 30)))
            loss = float(np.mean(np.logaddexp(0.0, -np.clip(z, -30, 30)))) + self.l2 * float(w @ w)
            grad = -(diff * sig[:, None]).mean(axis=0) + 2.0 * self.l2 * w
            w -= self.lr * grad
            self.history.append(loss)

        self.weights = w.astype(np.float32)
        if verbose:
            print(f"  pairwise LTR: {n:,} pairs, loss {self.history[0]:.4f} -> {self.history[-1]:.4f}")
        return self

    def training_accuracy(self, pos: np.ndarray, neg: np.ndarray) -> float:
        """Fraction of pairs ordered correctly -- the pairwise accuracy."""
        z = (pos - neg) @ self.weights
        return float((z > 0).mean())

    # ----------------------------------------------------------------- predict #
    def score(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("PairwiseLTR is not trained. Run scripts/train_reranker.py")
        return features @ self.weights

    def rerank(self, candidates: list[str], features: np.ndarray) -> list[tuple[str, float]]:
        scores = self.score(features)
        order = np.argsort(-scores, kind="stable")
        return [(candidates[i], float(scores[i])) for i in order]

    def weight_table(self) -> list[dict]:
        """For the UI: which signals did the model decide to trust?"""
        if self.weights is None:
            return []
        return sorted(
            (
                {"feature": n, "weight": float(w), "abs": abs(float(w))}
                for n, w in zip(self.feature_names, self.weights)
            ),
            key=lambda r: -r["abs"],
        )

    # ------------------------------------------------------------------- i/o  #
    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "weights": self.weights.tolist(),
                    "feature_names": self.feature_names,
                    "history": self.history[-200:],
                    "hyper": {"l2": self.l2, "lr": self.lr, "epochs": self.epochs},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "PairwiseLTR":
        state = json.loads(path.read_text(encoding="utf-8"))
        obj = cls(**state.get("hyper", {}))
        obj.weights = np.asarray(state["weights"], dtype=np.float32)
        obj.feature_names = state["feature_names"]
        obj.history = state.get("history", [])
        return obj
