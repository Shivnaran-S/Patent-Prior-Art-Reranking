"""
TF-IDF / VECTOR SPACE MODEL  --  implemented from scratch.

This is the baseline every IR course starts with, and it is here so we can
show what BM25 actually buys us over it.

WEIGHTING  (SMART notation: ltc.ltc -- log tf, idf, cosine)

    w(t,d) = (1 + log f(t,d))  *  log(N / df(t))
    then each document vector is L2-normalised, so

    similarity(q,d) = cosine(q, d) = q_hat . d_hat

DIFFERENCE FROM BM25 IN ONE LINE
--------------------------------
TF-IDF uses `1 + log(tf)`, which keeps growing without limit, and normalises by
vector length.  BM25 uses a saturating tf and normalises by *document length*
relative to the average, with a tunable strength b.  That is why BM25 usually
wins on collections with very uneven document lengths -- like patents.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .text_utils import analyze


class TfidfRetriever:
    name = "tfidf"

    def __init__(self) -> None:
        self.doc_ids: list[str] = []
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.matrix: sp.csr_matrix | None = None   # L2-normalised docs x terms

    def fit(self, doc_ids: list[str], texts: list[str]) -> "TfidfRetriever":
        self.doc_ids = list(doc_ids)
        n_docs = len(texts)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i, text in enumerate(texts):
            counts: dict[int, int] = {}
            for term in analyze(text):
                idx = self.vocab.get(term)
                if idx is None:
                    idx = len(self.vocab)
                    self.vocab[term] = idx
                counts[idx] = counts.get(idx, 0) + 1
            rows.extend([i] * len(counts))
            cols.extend(counts.keys())
            # log-weighted term frequency
            vals.extend(1.0 + np.log(np.asarray(list(counts.values()), dtype=np.float32)))

        tf = sp.coo_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(n_docs, len(self.vocab)),
        ).tocsr()

        df = np.diff(tf.tocsc().indptr).astype(np.float32)
        self.idf = np.log(n_docs / np.maximum(df, 1.0)).astype(np.float32)

        # multiply every column by its idf, then L2-normalise every row
        weighted = tf.multiply(self.idf[np.newaxis, :]).tocsr()
        norms = np.sqrt(weighted.multiply(weighted).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        self.matrix = sp.diags(1.0 / norms) @ weighted
        self.matrix = self.matrix.tocsr().astype(np.float32)
        return self

    def _query_vector(self, query: str) -> sp.csr_matrix:
        counts: dict[int, int] = {}
        for term in analyze(query):
            idx = self.vocab.get(term)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            return sp.csr_matrix((1, len(self.vocab)), dtype=np.float32)

        idxs = np.fromiter(counts.keys(), dtype=np.int32)
        tfs = np.fromiter(counts.values(), dtype=np.float32)
        weights = (1.0 + np.log(tfs)) * self.idf[idxs]
        norm = np.linalg.norm(weights) or 1.0
        return sp.csr_matrix(
            (weights / norm, (np.zeros_like(idxs), idxs)),
            shape=(1, len(self.vocab)),
            dtype=np.float32,
        )

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        q = self._query_vector(query)
        if q.nnz == 0:
            return []
        scores = (self.matrix @ q.T).toarray().ravel()   # cosine similarity
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(self.doc_ids[i], float(scores[i])) for i in idx if scores[i] > 0]

    def save(self, path: Path) -> None:
        with path.open("wb") as fh:
            pickle.dump(
                {"doc_ids": self.doc_ids, "vocab": self.vocab,
                 "idf": self.idf, "matrix": self.matrix},
                fh, protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        with path.open("rb") as fh:
            state = pickle.load(fh)
        obj = cls()
        obj.__dict__.update(state)
        return obj
