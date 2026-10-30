"""
FEATURE EXTRACTION for the Learning-to-Rank reranker.

Classical Learning-to-Rank does not read the text with a neural network.  It
turns each (query, document) pair into a short vector of RANKING SIGNALS and
learns how to weigh them.  That makes it:

  * fast        -- microseconds per pair, no transformer
  * explainable -- the learned weight of every feature can be printed
  * a fair comparison point for the cross-encoder: if a 9-feature linear model
    gets most of the gain, the transformer has to justify its cost.

The nine features, and why each is here:

  1  bm25_norm        lexical evidence, min-max normalised within the candidate list
  2  dense_norm       semantic evidence, same normalisation
  3  rrf              their rank-fusion combination
  4  bm25_rank_recip  1/rank from BM25 -- rank is often a better signal than score
  5  dense_rank_recip 1/rank from the dense retriever
  6  term_overlap     |q n d| / |q|          plain coverage of the query
  7  idf_overlap      IDF-weighted coverage  -- matching a RARE term counts more
  8  doc_len_norm     log document length, so the model can learn its own
                      length bias instead of inheriting BM25's
  9  query_coverage   fraction of the query's HIGH-IDF terms that appear

Normalising features *within each query's candidate list* is deliberate: the
model then learns to rank candidates against each other, not to predict an
absolute score, which is what a ranking task actually needs.
"""

from __future__ import annotations

import numpy as np

from .text_utils import analyze

FEATURE_NAMES = [
    "bm25_norm",
    "dense_norm",
    "rrf",
    "bm25_rank_recip",
    "dense_rank_recip",
    "term_overlap",
    "idf_overlap",
    "doc_len_norm",
    "query_coverage",
]


def _minmax_map(pairs: list[tuple[str, float]]) -> dict[str, float]:
    if not pairs:
        return {}
    vals = np.asarray([s for _, s in pairs], dtype=np.float64)
    lo, hi = vals.min(), vals.max()
    span = hi - lo
    if span < 1e-12:
        return {d: 0.0 for d, _ in pairs}
    return {d: float((s - lo) / span) for d, s in pairs}


def _rank_map(pairs: list[tuple[str, float]]) -> dict[str, int]:
    return {d: i + 1 for i, (d, _) in enumerate(pairs)}


def build_features(
    query: str,
    candidates: list[str],
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    doc_terms: dict[str, set[str]],
    doc_lengths: dict[str, int],
    idf_lookup,
    rrf_k: int = 60,
) -> np.ndarray:
    """Return an (n_candidates, 9) feature matrix."""
    bm25_norm = _minmax_map(bm25_results)
    dense_norm = _minmax_map(dense_results)
    bm25_rank = _rank_map(bm25_results)
    dense_rank = _rank_map(dense_results)

    q_terms = set(analyze(query))
    q_idf = {t: max(idf_lookup(t), 0.0) for t in q_terms}
    total_idf = sum(q_idf.values()) or 1.0
    # "High-IDF" = the half of the query's terms that are rarest, i.e. the ones
    # that actually identify the invention rather than the technical field.
    if q_idf:
        threshold = float(np.median(list(q_idf.values())))
        high_idf = {t for t, v in q_idf.items() if v >= threshold} or q_terms
    else:
        high_idf = set()

    rows = np.zeros((len(candidates), len(FEATURE_NAMES)), dtype=np.float32)
    for i, doc_id in enumerate(candidates):
        d_terms = doc_terms.get(doc_id, set())
        overlap = q_terms & d_terms
        r_bm = bm25_rank.get(doc_id, 10_000)
        r_dn = dense_rank.get(doc_id, 10_000)
        rows[i] = (
            bm25_norm.get(doc_id, 0.0),
            dense_norm.get(doc_id, 0.0),
            1.0 / (rrf_k + r_bm) + 1.0 / (rrf_k + r_dn),
            1.0 / r_bm,
            1.0 / r_dn,
            len(overlap) / max(len(q_terms), 1),
            sum(q_idf[t] for t in overlap) / total_idf,
            np.log1p(doc_lengths.get(doc_id, 0)) / 10.0,
            len(overlap & high_idf) / max(len(high_idf), 1),
        )
    return rows
