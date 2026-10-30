"""
Correctness tests for the parts we implemented ourselves.

Everything here is checked against a value computed BY HAND from the formula,
on a tiny collection small enough to verify with a calculator.  That is the
only way to know an IR implementation is right: a retriever that is subtly
wrong still returns a plausible-looking ranked list.

    python tests/test_core.py

No pytest needed -- it prints a PASS/FAIL line per test and exits non-zero on
any failure.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.bm25_retriever import BM25Retriever              # noqa: E402
from src.features import FEATURE_NAMES, build_features    # noqa: E402
from src.hybrid_retriever import (reciprocal_rank_fusion, # noqa: E402
                                  weighted_score_fusion)
from src.ltr_reranker import PairwiseLTR                  # noqa: E402
from src.metrics import (average_precision, ndcg_at_k,    # noqa: E402
                         paired_randomisation_test, precision_at_k,
                         recall_at_k, reciprocal_rank)
from src.text_utils import analyze, analyze_steps         # noqa: E402
from src.tfidf_retriever import TfidfRetriever            # noqa: E402

TOL = 1e-9
_results: list[tuple[str, bool, str]] = []


def check(name: str):
    def decorator(fn):
        try:
            fn()
            _results.append((name, True, ""))
        except Exception as exc:                       # noqa: BLE001
            _results.append((name, False, f"{exc}\n{traceback.format_exc(limit=2)}"))
        return fn
    return decorator


def close(a: float, b: float, tol: float = TOL) -> None:
    assert abs(a - b) < tol, f"expected {b!r}, got {a!r}"


# --------------------------------------------------------------------------- #
# a 4-document toy collection, chosen so every number can be checked by hand
# --------------------------------------------------------------------------- #
DOC_IDS = ["d1", "d2", "d3", "d4"]
DOCS = [
    "optical filter optical amplifier",       # 'optical' twice
    "optical filter",
    "rubber tire composition",
    "optical filter filter filter filter filter filter filter filter filter",
]


# =========================================================================== #
# metrics
# =========================================================================== #
@check("metrics: precision / recall at k")
def _():
    run = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    rel = {"c", "g"}
    close(precision_at_k(run, rel, 10), 0.2)
    close(precision_at_k(run, rel, 5), 0.2)       # only 'c' in the top 5
    close(recall_at_k(run, rel, 10), 1.0)
    close(recall_at_k(run, rel, 5), 0.5)
    close(recall_at_k(run, rel, 2), 0.0)


@check("metrics: MRR is 1 / rank of the first hit")
def _():
    close(reciprocal_rank(["a", "b", "c"], {"c"}), 1 / 3)
    close(reciprocal_rank(["c", "b", "a"], {"c"}), 1.0)
    close(reciprocal_rank(["a", "b"], {"z"}), 0.0)
    close(reciprocal_rank(["a", "b", "c"], {"c"}, k=2), 0.0)   # cut off before it


@check("metrics: MAP equals the hand-computed average precision")
def _():
    run = ["a", "b", "c", "d", "e", "f", "g"]
    rel = {"c", "g"}
    # hits at rank 3 (precision 1/3) and rank 7 (precision 2/7)
    close(average_precision(run, rel), (1 / 3 + 2 / 7) / 2)
    close(average_precision(["x"], {"x"}), 1.0)
    close(average_precision(["a"], set()), 0.0)


@check("metrics: nDCG equals DCG / ideal DCG")
def _():
    run = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    rel = {"c", "g"}
    dcg = 1 / math.log2(4) + 1 / math.log2(8)       # ranks 3 and 7
    idcg = 1 / math.log2(2) + 1 / math.log2(3)      # ideal: ranks 1 and 2
    close(ndcg_at_k(run, rel, 10), dcg / idcg)
    # a perfect ranking must score exactly 1
    close(ndcg_at_k(["c", "g", "a"], rel, 10), 1.0)
    # nDCG must never exceed 1
    assert ndcg_at_k(run, rel, 10) <= 1.0 + TOL


@check("metrics: nDCG rewards putting the hit higher")
def _():
    rel = {"x"}
    high = ndcg_at_k(["x", "a", "b", "c"], rel, 10)
    low = ndcg_at_k(["a", "b", "c", "x"], rel, 10)
    assert high > low, "ranking the relevant document first must score higher"


@check("significance: permutation test behaves at both extremes")
def _():
    identical = paired_randomisation_test([0.5] * 40, [0.5] * 40, 2000)
    close(identical, 1.0, tol=0.01)                 # no difference => p = 1
    huge = paired_randomisation_test([0.9] * 40, [0.1] * 40, 2000)
    assert huge < 0.01, f"a large consistent gain should be significant, got {huge}"


# =========================================================================== #
# preprocessing
# =========================================================================== #
@check("text: query and document are analysed identically")
def _():
    text = "The optical AMPLIFIERS, wherein comprising a filter."
    assert analyze(text) == analyze(text.lower())
    terms = analyze(text)
    assert "wherein" not in terms and "compris" not in terms, \
        "patent boilerplate must be removed"
    assert "the" not in terms


@check("text: stemming makes related word forms match")
def _():
    assert analyze("amplifier")[0] == analyze("amplifiers")[0]
    assert analyze("filtering")[0] == analyze("filters")[0]


@check("text: the step-by-step view only ever shrinks the token list")
def _():
    steps = analyze_steps("The optical filter 10 comprises a multilayer filter.")
    assert len(steps["tokenized"]) >= len(steps["stopwords_removed"])
    assert len(steps["stopwords_removed"]) >= len(steps["numerals_removed"])
    assert len(steps["numerals_removed"]) == len(steps["stemmed"])
    assert "10" not in steps["numerals_removed"], "bare numerals must be dropped"


# =========================================================================== #
# BM25
# =========================================================================== #
@check("bm25: IDF matches the Robertson/Sparck-Jones formula")
def _():
    r = BM25Retriever().fit(DOC_IDS, DOCS)
    n = len(DOC_IDS)
    for word in ("optical", "rubber"):
        term = analyze(word)[0]
        col = r.vocab[term]
        df = r.df[col]
        close(float(r.idf[col]), math.log1p((n - df + 0.5) / (df + 0.5)), tol=1e-5)


@check("bm25: a rarer term carries more weight than a common one")
def _():
    r = BM25Retriever().fit(DOC_IDS, DOCS)
    rubber = r.idf[r.vocab[analyze("rubber")[0]]]      # appears in 1 of 4
    optical = r.idf[r.vocab[analyze("optical")[0]]]    # appears in 3 of 4
    assert rubber > optical, "IDF must be higher for the rarer term"


@check("bm25: score equals the formula computed by hand")
def _():
    r = BM25Retriever(k1=1.2, b=0.75).fit(DOC_IDS, DOCS)
    term = analyze("rubber")[0]
    col = r.vocab[term]
    d = DOC_IDS.index("d3")
    f = 1.0                                            # 'rubber' once in d3
    norm = 1.2 * (1 - 0.75 + 0.75 * r.doc_len[d] / r.avgdl)
    expected = r.idf[col] * (f * (1.2 + 1)) / (f + norm)
    got = dict(r.search("rubber", 10))["d3"]
    close(got, float(expected), tol=1e-4)


@check("bm25: term frequency saturates (10x the term is not 10x the score)")
def _():
    r = BM25Retriever().fit(DOC_IDS, DOCS)
    scores = dict(r.search("filter", 10))
    # d4 has 'filter' 9 times, d2 has it once.
    assert scores["d4"] > scores["d2"], "more occurrences should still score higher"
    assert scores["d4"] < 9 * scores["d2"], "but nowhere near 9x -- that is saturation"


@check("bm25: b controls length normalisation")
def _():
    no_norm = BM25Retriever(b=0.0).fit(DOC_IDS, DOCS)
    full_norm = BM25Retriever(b=1.0).fit(DOC_IDS, DOCS)
    # d4 is the longest document; penalising length must lower its relative score.
    ratio_none = dict(no_norm.search("filter", 10))["d4"] / dict(no_norm.search("filter", 10))["d2"]
    ratio_full = dict(full_norm.search("filter", 10))["d4"] / dict(full_norm.search("filter", 10))["d2"]
    assert ratio_full < ratio_none, "b=1 must penalise the long document more than b=0"


@check("bm25: explain() contributions sum to the document's score")
def _():
    r = BM25Retriever().fit(DOC_IDS, DOCS)
    query = "optical filter"
    total = dict(r.search(query, 10))["d1"]
    parts = sum(c.contribution for c in r.explain(query, "d1"))
    close(parts, total, tol=1e-4)


@check("bm25: an out-of-vocabulary query returns nothing, without crashing")
def _():
    r = BM25Retriever().fit(DOC_IDS, DOCS)
    assert r.search("zzzznonexistent", 10) == []
    assert r.search("", 10) == []


@check("bm25: a saved-and-reloaded index behaves identically")
def _():
    # The app always loads from artifacts/, never from fit() -- so the load path
    # is the one that actually matters.
    import tempfile

    r = BM25Retriever().fit(DOC_IDS, DOCS)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bm25.pkl"
        r.save(path)
        loaded = BM25Retriever.load(path)
    query = "optical filter"
    assert loaded.search(query, 10) == r.search(query, 10)
    before = [(c.term, round(c.contribution, 6)) for c in r.explain(query, "d1")]
    after = [(c.term, round(c.contribution, 6)) for c in loaded.explain(query, "d1")]
    assert before == after, "explain() must survive a save/load round-trip"
    close(sum(c.contribution for c in loaded.explain(query, "d1")),
          dict(loaded.search(query, 10))["d1"], tol=1e-4)


@check("bm25: results come back sorted by descending score")
def _():
    r = BM25Retriever().fit(DOC_IDS, DOCS)
    scores = [s for _, s in r.search("optical filter", 10)]
    assert scores == sorted(scores, reverse=True)


# =========================================================================== #
# TF-IDF
# =========================================================================== #
@check("tfidf: cosine similarity never leaves [0, 1]")
def _():
    r = TfidfRetriever().fit(DOC_IDS, DOCS)
    for _, score in r.search("optical filter amplifier", 10):
        assert -TOL <= score <= 1 + TOL, f"cosine out of range: {score}"


@check("tfidf: a document identical to the query scores ~1")
def _():
    r = TfidfRetriever().fit(DOC_IDS, DOCS)
    close(dict(r.search("optical filter", 10))["d2"], 1.0, tol=1e-5)


@check("tfidf: document vectors are L2-normalised")
def _():
    r = TfidfRetriever().fit(DOC_IDS, DOCS)
    norms = np.sqrt(r.matrix.multiply(r.matrix).sum(axis=1)).A.ravel()
    for n in norms:
        close(float(n), 1.0, tol=1e-5)


# =========================================================================== #
# fusion
# =========================================================================== #
@check("rrf: score equals sum of 1 / (k + rank)")
def _():
    a = [("x", 9.0), ("y", 8.0)]
    b = [("y", 0.9), ("x", 0.5)]
    fused = dict(reciprocal_rank_fusion([a, b], k=60))
    close(fused["x"], 1 / 61 + 1 / 62)
    close(fused["y"], 1 / 62 + 1 / 61)


@check("rrf: consistently-good beats excellent-in-one-list")
def _():
    # E is 3rd and 3rd; F is 1st and 40th.
    lex = [(f"pad{i}", 1.0) for i in range(2)] + [("E", 1.0)]
    lex = [("F", 2.0)] + [("pad0", 1.5), ("E", 1.0)]
    dense = [("pad1", 1.0), ("pad2", 1.0), ("E", 1.0)] + \
            [(f"z{i}", 0.1) for i in range(36)] + [("F", 0.05)]
    fused = dict(reciprocal_rank_fusion([lex, dense], k=60))
    assert fused["E"] > fused["F"], "RRF should prefer the consistently-ranked document"


@check("rrf: ignores raw score scale entirely")
def _():
    a = [("x", 1000.0), ("y", 1.0)]
    a_scaled = [("x", 0.002), ("y", 0.001)]
    b = [("y", 0.9), ("x", 0.5)]
    assert reciprocal_rank_fusion([a, b]) == reciprocal_rank_fusion([a_scaled, b])


@check("weighted fusion: min-max normalises each list before combining")
def _():
    a = [("x", 10.0), ("y", 0.0)]      # -> x:1, y:0
    b = [("y", 1.0), ("x", 0.0)]       # -> y:1, x:0
    fused = dict(weighted_score_fusion([a, b], [0.5, 0.5]))
    close(fused["x"], 0.5)
    close(fused["y"], 0.5)


@check("weighted fusion: alpha=1 reproduces the first list's order")
def _():
    a = [("x", 10.0), ("y", 5.0), ("z", 0.0)]
    b = [("z", 1.0), ("y", 0.5), ("x", 0.0)]
    order = [d for d, _ in weighted_score_fusion([a, b], [1.0, 0.0])]
    assert order[0] == "x", f"alpha=1 should follow list A, got {order}"


# =========================================================================== #
# Learning-to-Rank
# =========================================================================== #
@check("ltr: learns to separate a linearly separable pair set")
def _():
    rng = np.random.default_rng(0)
    n, d = 400, len(FEATURE_NAMES)
    pos = rng.normal(1.0, 0.3, size=(n, d))
    neg = rng.normal(-1.0, 0.3, size=(n, d))
    model = PairwiseLTR(epochs=300).fit(pos, neg)
    acc = model.training_accuracy(pos, neg)
    assert acc > 0.95, f"should separate an easy set, got {acc:.3f}"
    assert model.history[-1] < model.history[0], "loss must decrease"


@check("ltr: scoring is order-preserving and shape-correct")
def _():
    rng = np.random.default_rng(1)
    model = PairwiseLTR(epochs=50).fit(
        rng.normal(1, 0.2, (50, len(FEATURE_NAMES))),
        rng.normal(-1, 0.2, (50, len(FEATURE_NAMES))),
    )
    feats = rng.normal(0, 1, (7, len(FEATURE_NAMES)))
    ranked = model.rerank([f"d{i}" for i in range(7)], feats)
    assert len(ranked) == 7
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


@check("ltr: weight_table reports one weight per feature")
def _():
    rng = np.random.default_rng(2)
    model = PairwiseLTR(epochs=20).fit(
        rng.normal(1, 0.2, (20, len(FEATURE_NAMES))),
        rng.normal(-1, 0.2, (20, len(FEATURE_NAMES))),
    )
    table = model.weight_table()
    assert len(table) == len(FEATURE_NAMES)
    assert {r["feature"] for r in table} == set(FEATURE_NAMES)


# =========================================================================== #
# features
# =========================================================================== #
@check("features: matrix has the documented shape and no NaNs")
def _():
    bm25 = [("d1", 5.0), ("d2", 3.0), ("d3", 1.0)]
    dense = [("d2", 0.9), ("d1", 0.7), ("d3", 0.2)]
    doc_terms = {d: set(analyze(t)) for d, t in zip(DOC_IDS, DOCS)}
    lengths = {d: len(t.split()) for d, t in zip(DOC_IDS, DOCS)}
    feats = build_features("optical filter", ["d1", "d2", "d3"], bm25, dense,
                           doc_terms, lengths, lambda t: 1.0)
    assert feats.shape == (3, len(FEATURE_NAMES))
    assert np.isfinite(feats).all(), "features must never contain NaN or inf"


@check("features: term_overlap is the share of query terms present")
def _():
    doc_terms = {"d2": set(analyze("optical filter"))}
    lengths = {"d2": 2}
    feats = build_features("optical filter", ["d2"], [("d2", 1.0)], [("d2", 1.0)],
                           doc_terms, lengths, lambda t: 1.0)
    close(float(feats[0, FEATURE_NAMES.index("term_overlap")]), 1.0, tol=1e-6)


@check("features: a candidate missing from both lists gets zeroed scores")
def _():
    doc_terms = {"d9": set()}
    feats = build_features("optical", ["d9"], [], [], doc_terms, {"d9": 1},
                           lambda t: 1.0)
    close(float(feats[0, FEATURE_NAMES.index("bm25_norm")]), 0.0)
    close(float(feats[0, FEATURE_NAMES.index("dense_norm")]), 0.0)


# =========================================================================== #
def main() -> int:
    print("Core correctness tests")
    print("=" * 68)
    width = max(len(n) for n, _, _ in _results)
    failures = 0
    for name, ok, err in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}")
        if not ok:
            failures += 1
            print("        " + err.replace("\n", "\n        ").rstrip())
    print("=" * 68)
    passed = len(_results) - failures
    print(f"{passed}/{len(_results)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
