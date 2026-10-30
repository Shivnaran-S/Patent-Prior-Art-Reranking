"""
HYBRID RETRIEVAL  --  combining a lexical and a dense ranker.

BM25 and the dense retriever fail in *different* ways:

    BM25   nails exact technical terminology, misses paraphrases
    Dense  catches paraphrases, drifts on rare part numbers and chemistry

So their errors are only weakly correlated, and fusing them raises recall above
either one alone.  Recall is what matters here, because stage 2 can only rerank
what stage 1 found -- a relevant document missed at stage 1 is lost forever.

TWO WAYS TO FUSE (both implemented, switchable in the UI)
---------------------------------------------------------
1. RECIPROCAL RANK FUSION (default).  Uses ranks only:

                             1
       RRF(d) = SUM  -------------------
                 i    k + rank_i(d)

   Because it never looks at raw scores, it does not care that BM25 returns
   e.g. 18.4 while cosine returns 0.71 -- no normalisation, nothing to tune.
   k=60 (from the original RRF paper) softens the dominance of rank 1.

2. WEIGHTED SCORE FUSION.  Min-max normalise each score list to [0,1], then

       score(d) = alpha * bm25_norm(d) + (1 - alpha) * dense_norm(d)

   More control, but it is sensitive to outliers in the raw scores, which is
   exactly what RRF was invented to avoid.
"""

from __future__ import annotations

import numpy as np

from . import config


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked lists into one using ranks only."""
    weights = weights or [1.0] * len(ranked_lists)
    fused: dict[str, float] = {}
    for results, w in zip(ranked_lists, weights):
        for rank, (doc_id, _score) in enumerate(results, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def weighted_score_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float],
) -> list[tuple[str, float]]:
    """Min-max normalise each list's scores, then take a weighted sum."""
    normalised: list[dict[str, float]] = []
    for results in ranked_lists:
        if not results:
            normalised.append({})
            continue
        ids = [d for d, _ in results]
        scores = _minmax(np.asarray([s for _, s in results], dtype=np.float64))
        normalised.append(dict(zip(ids, scores)))

    all_ids = {d for part in normalised for d in part}
    fused = {
        d: sum(w * part.get(d, 0.0) for part, w in zip(normalised, weights))
        for d in all_ids
    }
    return sorted(fused.items(), key=lambda kv: -kv[1])


class HybridRetriever:
    """Runs both first-stage retrievers and fuses their rankings."""

    name = "hybrid"

    def __init__(self, lexical, dense, method: str = "rrf") -> None:
        self.lexical = lexical        # BM25Retriever
        self.dense = dense            # DenseRetriever
        self.method = method          # "rrf" | "weighted"
        self.cfg = config.SETTINGS.fusion

    def search(
        self,
        query: str,
        k: int = 100,
        method: str | None = None,
        alpha: float | None = None,
        rrf_k: int | None = None,
        pool: int | None = None,
    ) -> list[tuple[str, float]]:
        method = method or self.method
        alpha = self.cfg.alpha if alpha is None else alpha
        rrf_k = self.cfg.rrf_k if rrf_k is None else rrf_k
        # Retrieve deeper than k from each arm, so fusion has room to promote a
        # document that one arm ranked 80th and the other ranked 3rd.
        pool = pool or max(k * 2, 200)

        lex = self.lexical.search(query, pool)
        den = self.dense.search(query, pool)

        if method == "weighted":
            fused = weighted_score_fusion([lex, den], [alpha, 1.0 - alpha])
        else:
            fused = reciprocal_rank_fusion([lex, den], k=rrf_k, weights=[alpha, 1.0 - alpha])
        return fused[:k]

    def search_components(self, query: str, k: int = 100) -> dict[str, list[tuple[str, float]]]:
        """Return each arm's ranking plus the fused one -- the UI shows all
        three side by side so the effect of fusion is visible."""
        pool = max(k * 2, 200)
        lex = self.lexical.search(query, pool)
        den = self.dense.search(query, pool)
        return {
            "bm25": lex[:k],
            "dense": den[:k],
            "hybrid": reciprocal_rank_fusion(
                [lex, den], k=self.cfg.rrf_k, weights=[self.cfg.alpha, 1 - self.cfg.alpha]
            )[:k],
        }
