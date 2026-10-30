"""
BM25  --  implemented from scratch (no rank_bm25, no sklearn).

THE FORMULA
-----------
                         f(t,d) * (k1 + 1)
  score(q,d) = SUM  IDF(t) * -----------------------------------
               t in q         f(t,d) + k1 * (1 - b + b * |d|/avgdl)

                         N - df(t) + 0.5
  IDF(t) = ln( 1 + ----------------------- )          <- Robertson/Sparck-Jones
                         df(t) + 0.5

Reading it left to right:
  * IDF(t)            rare words matter more than common ones
  * f(t,d)*(k1+1)/... term frequency, but SATURATING -- the 20th occurrence of
                      a word adds almost nothing (controlled by k1)
  * b * |d|/avgdl     long documents are penalised, because they match more
                      words simply by being long (controlled by b)

WHY IT IS STORED THE WAY IT IS
------------------------------
We keep the raw term-frequency matrix in CSC (compressed sparse COLUMN) form
and apply the BM25 weighting at query time, on the query's columns only.
That costs a few extra microseconds but means k1 and b can be changed live --
which is exactly what the Ablation page does.  A pre-weighted matrix would be
marginally faster but would have to be rebuilt for every parameter value.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import config
from .text_utils import analyze


@dataclass
class TermContribution:
    """How much one query term contributed to one document's score -- this is
    what makes BM25 explainable in the UI."""
    term: str
    df: int
    idf: float
    tf_in_doc: int
    contribution: float


class BM25Retriever:
    """Okapi BM25 over an in-memory inverted index."""

    name = "bm25"

    def __init__(self, k1: float | None = None, b: float | None = None) -> None:
        cfg = config.SETTINGS.bm25
        self.k1 = cfg.k1 if k1 is None else k1
        self.b = cfg.b if b is None else b
        self.doc_ids: list[str] = []
        self.vocab: dict[str, int] = {}
        self.tf: sp.csc_matrix | None = None    # docs x terms, raw counts
        # A CSR view of the same matrix. CSC gives fast COLUMN slicing, which is
        # what scoring needs (one column per query term); explain() needs one
        # ROW (a single document), and slicing a CSC by row does not give usable
        # .indices -- so we keep both views.
        self._tf_csr: sp.csr_matrix | None = None
        self._pos: dict[str, int] = {}
        self.doc_len: np.ndarray | None = None
        self.avgdl: float = 0.0
        self.df: np.ndarray | None = None
        self.idf: np.ndarray | None = None

    # ------------------------------------------------------------------ build #
    def fit(self, doc_ids: list[str], texts: list[str]) -> "BM25Retriever":
        self.doc_ids = list(doc_ids)
        n_docs = len(texts)

        # Build the inverted index as a sparse matrix in COO triplet form.
        rows: list[int] = []
        cols: list[int] = []
        vals: list[int] = []
        doc_len = np.zeros(n_docs, dtype=np.float32)

        for i, text in enumerate(texts):
            terms = analyze(text)
            doc_len[i] = len(terms)
            counts: dict[int, int] = {}
            for term in terms:
                idx = self.vocab.get(term)
                if idx is None:
                    idx = len(self.vocab)
                    self.vocab[term] = idx
                counts[idx] = counts.get(idx, 0) + 1
            rows.extend([i] * len(counts))
            cols.extend(counts.keys())
            vals.extend(counts.values())

        n_terms = len(self.vocab)
        self.tf = sp.coo_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(n_docs, n_terms),
        ).tocsc()
        self._tf_csr = self.tf.tocsr()
        self._pos = {d: i for i, d in enumerate(self.doc_ids)}
        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if n_docs else 0.0

        # df(t) = number of documents containing t = nonzeros in column t
        self.df = np.diff(self.tf.indptr).astype(np.float32)
        self.idf = np.log1p((n_docs - self.df + 0.5) / (self.df + 0.5)).astype(np.float32)
        return self

    # ------------------------------------------------------------------ score #
    def _score_all(
        self, query: str, k1: float | None = None, b: float | None = None
    ) -> tuple[np.ndarray, list[tuple[str, int]]]:
        """Return the BM25 score of every document, plus the matched terms."""
        k1 = self.k1 if k1 is None else k1
        b = self.b if b is None else b

        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        matched: list[tuple[str, int]] = []

        # Query term frequency: a term repeated in the query counts more.
        q_terms: dict[str, int] = {}
        for t in analyze(query):
            q_terms[t] = q_terms.get(t, 0) + 1

        # Length normalisation denominator, precomputed per document.
        norm = k1 * (1.0 - b + b * (self.doc_len / max(self.avgdl, 1e-9)))

        for term, qtf in q_terms.items():
            col = self.vocab.get(term)
            if col is None:                      # term not in the corpus at all
                continue
            matched.append((term, col))
            start, end = self.tf.indptr[col], self.tf.indptr[col + 1]
            docs = self.tf.indices[start:end]    # documents containing the term
            freqs = self.tf.data[start:end]      # f(t,d) for those documents
            # The BM25 term weight, vectorised over just those documents.
            weight = self.idf[col] * (freqs * (k1 + 1.0)) / (freqs + norm[docs])
            scores[docs] += weight * qtf

        return scores, matched

    def search(
        self,
        query: str,
        k: int = 100,
        k1: float | None = None,
        b: float | None = None,
    ) -> list[tuple[str, float]]:
        scores, _ = self._score_all(query, k1=k1, b=b)
        return self._top_k(scores, k)

    def _top_k(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        k = min(k, len(scores))
        if k == 0:
            return []
        # argpartition is O(n); a full sort of 12k would be wasteful.
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(self.doc_ids[i], float(scores[i])) for i in idx if scores[i] > 0]

    # ---------------------------------------------------------------- explain #
    def explain(self, query: str, doc_id: str, top_terms: int = 12) -> list[TermContribution]:
        """Break a single document's score into per-term contributions.

        This is the payoff of writing BM25 ourselves: we can show the professor
        exactly WHY a document scored what it did.
        """
        d = self._pos.get(doc_id, -1)
        if d < 0:
            return []
        norm_d = self.k1 * (1.0 - self.b + self.b * (self.doc_len[d] / max(self.avgdl, 1e-9)))

        q_terms: dict[str, int] = {}
        for t in analyze(query):
            q_terms[t] = q_terms.get(t, 0) + 1

        out: list[TermContribution] = []
        # CSR row slice: .indices are the TERM columns present in this document.
        row = self._tf_csr[d]
        row_map = dict(zip(row.indices, row.data))
        for term, qtf in q_terms.items():
            col = self.vocab.get(term)
            if col is None:
                continue
            f = float(row_map.get(col, 0.0))
            if f == 0.0:
                continue
            contrib = float(self.idf[col] * (f * (self.k1 + 1.0)) / (f + norm_d) * qtf)
            out.append(
                TermContribution(
                    term=term,
                    df=int(self.df[col]),
                    idf=float(self.idf[col]),
                    tf_in_doc=int(f),
                    contribution=contrib,
                )
            )
        out.sort(key=lambda c: -c.contribution)
        return out[:top_terms]

    def term_stats(self, term: str) -> dict | None:
        col = self.vocab.get(term)
        if col is None:
            return None
        return {
            "term": term,
            "df": int(self.df[col]),
            "idf": float(self.idf[col]),
            "n_docs": len(self.doc_ids),
        }

    # ------------------------------------------------------------------- i/o  #
    def save(self, path: Path) -> None:
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "vocab": self.vocab,
                    "tf": self.tf,
                    "doc_len": self.doc_len,
                    "avgdl": self.avgdl,
                    "df": self.df,
                    "idf": self.idf,
                    "k1": self.k1,
                    "b": self.b,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        with path.open("rb") as fh:
            state = pickle.load(fh)
        obj = cls(k1=state["k1"], b=state["b"])
        obj.doc_ids = state["doc_ids"]
        obj.vocab = state["vocab"]
        obj.tf = state["tf"]
        obj._tf_csr = obj.tf.tocsr()
        obj._pos = {d: i for i, d in enumerate(obj.doc_ids)}
        obj.doc_len = state["doc_len"]
        obj.avgdl = state["avgdl"]
        obj.df = state["df"]
        obj.idf = state["idf"]
        return obj
