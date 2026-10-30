"""
EVALUATION METRICS  --  implemented from scratch, with the definitions written
out, because these are what the project is finally judged on.

Notation: a run is a ranked list of doc_ids; `relevant` is the set of doc_ids
an examiner judged relevant for that query.

RECALL@k        |retrieved@k n relevant| / |relevant|
                "did we find it at all?"  THE metric for stage 1, because the
                reranker can only reorder what stage 1 returned.

PRECISION@k     |retrieved@k n relevant| / k
                "how much of what we showed is useful?"

MRR@k           1 / rank of the FIRST relevant document  (0 if none in top k)
                Matches how an examiner works: they read from the top and stop
                when they find blocking prior art, so only the first hit counts.

MAP             mean over queries of Average Precision, where
                AP = (1/|relevant|) * SUM_k  P@k * rel(k)
                Rewards getting ALL relevant documents high, not just one.

nDCG@k          DCG@k / IDCG@k ,   DCG@k = SUM_i  rel_i / log2(i + 1)
                The discount log2(i+1) means rank 1 is worth much more than
                rank 10; dividing by the ideal DCG puts it on a 0..1 scale so
                queries with different numbers of relevant documents can be
                averaged fairly.  This is the standard headline metric.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


# --------------------------------------------------------------------------- #
# per-query metrics
# --------------------------------------------------------------------------- #
def recall_at_k(run: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(run[:k]) & relevant) / len(relevant)


def precision_at_k(run: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(run[:k]) & relevant) / k


def reciprocal_rank(run: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    cut = run[:k] if k else run
    for i, doc_id in enumerate(cut, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def average_precision(run: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    if not relevant:
        return 0.0
    cut = run[:k] if k else run
    hits = 0
    total = 0.0
    for i, doc_id in enumerate(cut, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / i                  # == precision@i at a relevant position
    return total / len(relevant)


def dcg(run: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(
        1.0 / math.log2(i + 1)
        for i, doc_id in enumerate(run[:k], start=1)
        if doc_id in relevant
    )


def ndcg_at_k(run: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg(run, relevant, k) / ideal if ideal > 0 else 0.0


def first_relevant_rank(run: Sequence[str], relevant: set[str]) -> int | None:
    for i, doc_id in enumerate(run, start=1):
        if doc_id in relevant:
            return i
    return None


# --------------------------------------------------------------------------- #
# aggregation over a whole run
# --------------------------------------------------------------------------- #
def evaluate_run(
    run: Sequence[str],
    relevant: set[str],
    cutoffs: Iterable[int] = (1, 5, 10, 20, 50, 100),
    primary: int = 10,
) -> dict[str, float]:
    """All metrics for ONE query."""
    out: dict[str, float] = {}
    for k in cutoffs:
        out[f"recall@{k}"] = recall_at_k(run, relevant, k)
        out[f"precision@{k}"] = precision_at_k(run, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(run, relevant, k)
    out[f"mrr@{primary}"] = reciprocal_rank(run, relevant, primary)
    out["mrr"] = reciprocal_rank(run, relevant)
    out["map"] = average_precision(run, relevant)
    rank = first_relevant_rank(run, relevant)
    out["first_relevant_rank"] = float(rank) if rank else 0.0
    out["found"] = 1.0 if rank else 0.0
    return out


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average over queries (every query counts equally)."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q.get(k, 0.0) for q in per_query) / len(per_query) for k in keys}


# --------------------------------------------------------------------------- #
# significance test
# --------------------------------------------------------------------------- #
def paired_randomisation_test(
    a: list[float], b: list[float], n_permutations: int = 10_000, seed: int = 42
) -> float:
    """Two-sided paired randomisation (permutation) test.

    Standard practice in IR: "method B beats method A" is only meaningful if the
    gain survives a significance test.  We repeatedly flip the sign of each
    query's difference at random and ask how often we see a mean difference at
    least as large as the one we observed.  The result is a p-value, with no
    assumption that the differences are normally distributed -- which, for
    per-query IR metrics, they are not.
    """
    import random as _random

    if len(a) != len(b) or not a:
        return 1.0
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(sum(diffs) / len(diffs))
    rng = _random.Random(seed)
    hits = 0
    for _ in range(n_permutations):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / len(diffs)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_permutations + 1)
