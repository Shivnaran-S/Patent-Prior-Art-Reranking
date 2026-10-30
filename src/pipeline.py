"""
THE TWO-STAGE PIPELINE  --  the object the whole project is built around.

    query
      |
      v
  +-------------------------------------------------+
  | STAGE 1  candidate generation   (high RECALL)   |
  |   TF-IDF | BM25 | Dense | Hybrid(RRF)           |
  |   11,849 documents  ->  top 100                 |
  +-------------------------------------------------+
      |
      v
  +-------------------------------------------------+
  | STAGE 2  reranking              (high PRECISION)|
  |   none | Learning-to-Rank | Cross-encoder       |
  |   top 100  ->  reordered top 10                 |
  +-------------------------------------------------+
      |
      v
  final ranking  (+ timings for every stage)

`SearchEngine` owns the indexes and builds/caches them once.  `search()`
returns a `SearchResult` that carries not just the ranking but the stage-1
ranking too, so the UI can show how far each document MOVED -- which is the
single most convincing way to demonstrate what a reranker does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import config
from .bm25_retriever import BM25Retriever
from .cross_encoder_reranker import CrossEncoderReranker
from .dataset import Collection, load_collection
from .dense_retriever import DenseRetriever
from .features import build_features
from .hybrid_retriever import HybridRetriever
from .ltr_reranker import PairwiseLTR
from .text_utils import analyze
from .tfidf_retriever import TfidfRetriever

BM25_PATH = config.ARTIFACT_DIR / "bm25.pkl"
TFIDF_PATH = config.ARTIFACT_DIR / "tfidf.pkl"
DENSE_PATH = config.ARTIFACT_DIR / "dense.npz"
LTR_PATH = config.ARTIFACT_DIR / "ltr.json"
DOCTERMS_PATH = config.ARTIFACT_DIR / "doc_terms.pkl"


@dataclass
class SearchResult:
    query: str
    stage1_method: str
    stage2_method: str
    stage1: list[tuple[str, float]] = field(default_factory=list)
    final: list[tuple[str, float]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    components: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    @property
    def stage1_ranks(self) -> dict[str, int]:
        return {d: i + 1 for i, (d, _) in enumerate(self.stage1)}

    def movement(self) -> list[dict]:
        """Rank change caused by stage 2, for the side-by-side demo table."""
        before = self.stage1_ranks
        rows = []
        for new_rank, (doc_id, score) in enumerate(self.final, start=1):
            old = before.get(doc_id)
            rows.append(
                {
                    "doc_id": doc_id,
                    "rank": new_rank,
                    "stage1_rank": old,
                    "delta": (old - new_rank) if old else None,
                    "score": score,
                }
            )
        return rows


class SearchEngine:
    """Builds, caches and runs every retriever and reranker."""

    def __init__(self, collection: Collection | None = None) -> None:
        self.collection = collection or load_collection()
        self.bm25: BM25Retriever | None = None
        self.tfidf: TfidfRetriever | None = None
        self.dense: DenseRetriever | None = None
        self.hybrid: HybridRetriever | None = None
        self.ltr: PairwiseLTR | None = None
        self._cross: dict[str, CrossEncoderReranker] = {}
        self._doc_terms: dict[str, set[str]] | None = None
        self.build_log: list[str] = []

    # ------------------------------------------------------------------ build #
    def build(self, progress=None, include_dense: bool = True) -> "SearchEngine":
        """Load cached indexes, building any that are missing."""
        col = self.collection
        doc_ids, texts = col.doc_ids, col.doc_texts

        def say(msg: str) -> None:
            self.build_log.append(msg)
            if progress is not None:
                progress(msg)

        # --- lexical ------------------------------------------------------- #
        if BM25_PATH.exists():
            self.bm25 = BM25Retriever.load(BM25_PATH)
            say("BM25 index loaded from cache")
        else:
            say("Building BM25 inverted index ...")
            self.bm25 = BM25Retriever().fit(doc_ids, texts)
            self.bm25.save(BM25_PATH)
            say(f"BM25 built: {len(self.bm25.vocab):,} terms")

        if TFIDF_PATH.exists():
            self.tfidf = TfidfRetriever.load(TFIDF_PATH)
            say("TF-IDF index loaded from cache")
        else:
            say("Building TF-IDF matrix ...")
            self.tfidf = TfidfRetriever().fit(doc_ids, texts)
            self.tfidf.save(TFIDF_PATH)
            say("TF-IDF built")

        # --- dense --------------------------------------------------------- #
        if include_dense:
            if DENSE_PATH.exists():
                self.dense = DenseRetriever.load(DENSE_PATH)
                say(f"Dense index loaded from cache ({self.dense.embeddings.shape[1]}-d)")
            else:
                say("Encoding corpus with the bi-encoder (one-off, a few minutes) ...")
                self.dense = DenseRetriever().fit(doc_ids, texts)
                self.dense.save(DENSE_PATH)
                say("Dense index built")
            self.hybrid = HybridRetriever(self.bm25, self.dense)

        # --- trained reranker --------------------------------------------- #
        if LTR_PATH.exists():
            self.ltr = PairwiseLTR.load(LTR_PATH)
            say("Learning-to-Rank weights loaded")

        return self

    # ------------------------------------------------------------- doc terms #
    @property
    def doc_terms(self) -> dict[str, set[str]]:
        """Analysed term sets, cached -- needed by the LTR features."""
        if self._doc_terms is None:
            import pickle

            if DOCTERMS_PATH.exists():
                with DOCTERMS_PATH.open("rb") as fh:
                    self._doc_terms = pickle.load(fh)
            else:
                self._doc_terms = {
                    d: set(analyze(self.collection.documents[d]["text"]))
                    for d in self.collection.doc_ids
                }
                with DOCTERMS_PATH.open("wb") as fh:
                    pickle.dump(self._doc_terms, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return self._doc_terms

    @property
    def doc_lengths(self) -> dict[str, int]:
        return {d: m["n_words"] for d, m in self.collection.documents.items()}

    def idf(self, term: str) -> float:
        col = self.bm25.vocab.get(term)
        return float(self.bm25.idf[col]) if col is not None else 0.0

    def cross_encoder(self, variant: str = "base", long_doc_strategy: str = "truncate"):
        """Lazily cache cross-encoder instances (loading one costs seconds)."""
        key = f"{variant}:{long_doc_strategy}"
        if key not in self._cross:
            if variant == "finetuned":
                model = CrossEncoderReranker.finetuned(long_doc_strategy=long_doc_strategy)
                if model is None:
                    raise FileNotFoundError(
                        "No fine-tuned checkpoint. Run: python scripts/train_reranker.py"
                    )
            else:
                model = CrossEncoderReranker(long_doc_strategy=long_doc_strategy)
            self._cross[key] = model
        return self._cross[key]

    # ----------------------------------------------------------------- stage 1 #
    def retrieve(self, query: str, method: str, k: int, **kwargs) -> list[tuple[str, float]]:
        if method == "bm25":
            return self.bm25.search(query, k, **kwargs)
        if method == "tfidf":
            return self.tfidf.search(query, k)
        if method in ("dense", "hybrid") and self.dense is None:
            raise FileNotFoundError(
                f"'{method}' needs the dense index, which has not been built. "
                "Run: python scripts/build_indexes.py"
            )
        if method == "dense":
            return self.dense.search(query, k)
        if method == "hybrid":
            return self.hybrid.search(query, k, **kwargs)
        raise ValueError(f"unknown stage-1 method: {method}")

    # ----------------------------------------------------------------- stage 2 #
    def _ltr_features(self, query: str, candidates: list[str], pool: int = 300) -> np.ndarray:
        bm25_results = self.bm25.search(query, pool)
        dense_results = self.dense.search(query, pool) if self.dense else []
        return build_features(
            query=query,
            candidates=candidates,
            bm25_results=bm25_results,
            dense_results=dense_results,
            doc_terms=self.doc_terms,
            doc_lengths=self.doc_lengths,
            idf_lookup=self.idf,
            rrf_k=config.SETTINGS.fusion.rrf_k,
        )

    # ------------------------------------------------------------------ search #
    def search(
        self,
        query: str,
        stage1: str | None = None,
        stage2: str | None = None,
        first_stage_k: int | None = None,
        final_k: int | None = None,
        long_doc_strategy: str = "truncate",
        progress=None,
        with_components: bool = False,
        **stage1_kwargs,
    ) -> SearchResult:
        cfg = config.SETTINGS.pipeline
        stage1 = stage1 or cfg.stage1
        stage2 = stage2 or cfg.stage2
        first_stage_k = first_stage_k or cfg.first_stage_k
        final_k = final_k or cfg.final_k
        timings: dict[str, float] = {}

        # ---- stage 1 ------------------------------------------------------ #
        t0 = time.perf_counter()
        candidates = self.retrieve(query, stage1, first_stage_k, **stage1_kwargs)
        timings["stage1_ms"] = (time.perf_counter() - t0) * 1000

        components: dict[str, list[tuple[str, float]]] = {}
        if with_components and self.hybrid is not None:
            components = self.hybrid.search_components(query, first_stage_k)

        # ---- stage 2 ------------------------------------------------------ #
        t0 = time.perf_counter()
        if stage2 in ("none", None) or not candidates:
            final = list(candidates)
        elif stage2 == "ltr":
            if self.ltr is None:
                raise FileNotFoundError(
                    "Learning-to-Rank is not trained. Run: python scripts/train_reranker.py"
                )
            ids = [d for d, _ in candidates]
            final = self.ltr.rerank(ids, self._ltr_features(query, ids))
        elif stage2 in ("cross_encoder", "cross_encoder_ft"):
            variant = "finetuned" if stage2 == "cross_encoder_ft" else "base"
            model = self.cross_encoder(variant, long_doc_strategy)
            texts = {d: self.collection.documents[d]["text"] for d, _ in candidates}
            final = model.rerank(query, candidates, texts, progress=progress)
        else:
            raise ValueError(f"unknown stage-2 method: {stage2}")
        timings["stage2_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = timings["stage1_ms"] + timings["stage2_ms"]

        return SearchResult(
            query=query,
            stage1_method=stage1,
            stage2_method=stage2 or "none",
            stage1=list(candidates),
            final=final[:final_k] if final_k else final,
            timings_ms=timings,
            components=components,
        )
