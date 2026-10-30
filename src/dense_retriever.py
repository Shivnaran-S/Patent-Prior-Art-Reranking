"""
DENSE RETRIEVAL (bi-encoder + FAISS)  --  the fix for vocabulary mismatch.

THE PROBLEM IT SOLVES
---------------------
BM25 can only match words that are literally present.  A claim about a
"fastener" will never match prior art about a "screw", even though an examiner
would call them the same thing.  This is *vocabulary mismatch*, and in patents
it is the dominant failure mode, because applicants deliberately invent their
own terminology.

HOW IT WORKS
------------
A transformer encodes text into a 384-dimensional vector such that texts with
similar MEANING land close together:

    document -> [0.02, -0.11, ...]   (computed once, offline, and stored)
    query    -> [0.03, -0.09, ...]   (computed at search time)
    score    = cosine(query, document)

"Bi-encoder" means query and document are encoded SEPARATELY and never see each
other.  That is the whole point: because documents do not depend on the query,
all 11,849 vectors are precomputed once, and searching is a single matrix
multiply.  (Contrast with the cross-encoder in stage 2, which must re-run the
transformer for every query-document pair.)

POOLING
-------
The transformer returns one vector per token.  We mean-pool them, masking out
padding, then L2-normalise -- after which inner product IS cosine similarity,
so FAISS's IndexFlatIP gives exact cosine search.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


class DenseRetriever:
    name = "dense"

    def __init__(self, model_name: str | None = None) -> None:
        cfg = config.SETTINGS.dense
        self.model_name = model_name or cfg.model_name
        self.max_length = cfg.max_length
        self.batch_size = cfg.batch_size
        self.doc_ids: list[str] = []
        self.embeddings: np.ndarray | None = None       # (n_docs, dim) float32
        self.index = None                               # faiss.IndexFlatIP
        self._tokenizer = None
        self._model = None
        # Encoding a query costs a transformer pass (~15 ms). Training and the
        # UI hit the same query many times, so one small cache removes most of
        # that work.
        #
        # BENCHMARKS MUST TURN THIS OFF (`cache_queries = False`). Encoding the
        # query is a real per-query cost in production, and if it is cached then
        # whichever method runs second looks faster than the one that ran first
        # purely because the cache was already warm -- which would corrupt the
        # accuracy-versus-latency comparison.
        self.cache_queries: bool = True
        self._query_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------- the model #
    def _ensure_model(self) -> None:
        """Load the transformer lazily -- the Streamlit app should start fast
        even if the user only wants to look at BM25."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    def encode(self, texts: list[str], progress=None) -> np.ndarray:
        """Mean-pooled, L2-normalised embeddings for a list of texts."""
        import torch

        self._ensure_model()
        out: list[np.ndarray] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start : start + self.batch_size]
            with torch.no_grad():
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                hidden = self._model(**enc).last_hidden_state      # (B, T, dim)
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            out.append(pooled.cpu().numpy().astype(np.float32))
            if progress is not None:
                progress(min(start + self.batch_size, total), total)
        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)

    # ------------------------------------------------------------------ build #
    def fit(self, doc_ids: list[str], texts: list[str], progress=None) -> "DenseRetriever":
        self.doc_ids = list(doc_ids)
        self.embeddings = self.encode(texts, progress=progress)
        self._build_index()
        return self

    def _build_index(self) -> None:
        import faiss

        dim = self.embeddings.shape[1]
        # IndexFlatIP = exhaustive inner-product search. "Flat" means no
        # approximation, so recall is exact -- at 12k documents an approximate
        # index (IVF/HNSW) would only add error, not speed.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.ascontiguousarray(self.embeddings))

    # ----------------------------------------------------------------- search #
    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        import faiss  # noqa: F401  (index already built)

        q = self.embed_query(query).reshape(1, -1)
        k = min(k, len(self.doc_ids))
        scores, idx = self.index.search(np.ascontiguousarray(q), k)
        return [(self.doc_ids[j], float(s))
                for j, s in zip(idx[0], scores[0]) if j >= 0]

    def search_batch(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        q = self.encode(queries)
        k = min(k, len(self.doc_ids))
        scores, idx = self.index.search(np.ascontiguousarray(q), k)
        return [
            [(self.doc_ids[j], float(s)) for j, s in zip(row_idx, row_scores) if j >= 0]
            for row_idx, row_scores in zip(idx, scores)
        ]

    def embed_query(self, query: str) -> np.ndarray:
        if not self.cache_queries:
            return self.encode([query])[0]
        cached = self._query_cache.get(query)
        if cached is None:
            cached = self.encode([query])[0]
            if len(self._query_cache) < 50_000:
                self._query_cache[query] = cached
        return cached

    def similarity(self, query: str, doc_ids: list[str]) -> dict[str, float]:
        """Cosine similarity of one query against specific documents (used by
        the feature extractor for Learning-to-Rank)."""
        qv = self.embed_query(query)
        pos = {d: i for i, d in enumerate(self.doc_ids)}
        return {
            d: float(qv @ self.embeddings[pos[d]]) for d in doc_ids if d in pos
        }

    # ------------------------------------------------------------------- i/o  #
    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            doc_ids=np.array(self.doc_ids, dtype=object),
            embeddings=self.embeddings,
            model_name=np.array([self.model_name], dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "DenseRetriever":
        state = np.load(path, allow_pickle=True)
        obj = cls(model_name=str(state["model_name"][0]))
        obj.doc_ids = list(state["doc_ids"])
        obj.embeddings = state["embeddings"].astype(np.float32)
        obj._build_index()
        return obj
