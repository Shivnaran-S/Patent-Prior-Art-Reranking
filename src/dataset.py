"""
THE TEST COLLECTION  --  corpus + queries + qrels, loaded once and cached.

This is the only module that touches data/*.jsonl.  Everything downstream asks
this object for documents and relevance judgements, which keeps the experiment
reproducible: change the data, and nothing else has to change.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import config


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing. Run:  python scripts/prepare_data.py"
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class Collection:
    """An IR test collection in its standard three-part form."""

    documents: dict[str, dict]                  # doc_id -> {text, n_words, ...}
    queries: dict[str, dict]                    # query_id -> {text, split, ...}
    qrels: dict[str, dict[str, int]]            # query_id -> {doc_id: relevance}
    hard_negatives: dict[str, list[str]]        # query_id -> [doc_id, ...]
    stats: dict

    # ---- convenient views ------------------------------------------------- #
    @property
    def doc_ids(self) -> list[str]:
        return self._doc_ids

    @property
    def doc_texts(self) -> list[str]:
        return self._doc_texts

    def __post_init__(self) -> None:
        # A fixed document order: every index (BM25 rows, FAISS vectors) uses
        # the SAME integer position for the same document.
        self._doc_ids = list(self.documents.keys())
        self._doc_texts = [self.documents[d]["text"] for d in self._doc_ids]
        self._doc_pos = {d: i for i, d in enumerate(self._doc_ids)}

    def position(self, doc_id: str) -> int:
        return self._doc_pos[doc_id]

    def doc(self, doc_id: str) -> dict:
        return self.documents[doc_id]

    def query(self, query_id: str) -> dict:
        return self.queries[query_id]

    def relevant(self, query_id: str) -> set[str]:
        return set(self.qrels.get(query_id, {}))

    def split_queries(self, split: str) -> list[str]:
        return [q for q, m in self.queries.items() if m["split"] == split]

    def sample_queries(self, split: str, n: int, seed: int = 42) -> list[str]:
        """Deterministic sample -- the same queries every run, so numbers from
        different runs are directly comparable."""
        ids = sorted(self.split_queries(split))
        if n >= len(ids):
            return ids
        return sorted(random.Random(seed).sample(ids, n))

    def showcase_queries(self, n: int = 40) -> list[str]:
        """Queries that kept their real EPO publication number -- nicest for a
        live demo because we can show the genuine patent and a link to it.

        Ordered by how well they read on a projector: a 40-80 word claim is long
        enough to be a realistic patent query but short enough to show on one
        screen, so those come first.
        """
        with_meta = [q for q, m in self.queries.items() if m.get("patent_id")]
        with_meta.sort(key=lambda q: abs(self.queries[q]["n_words"] - 60))
        return with_meta[:n]


@lru_cache(maxsize=1)
def load_collection() -> Collection:
    """Load (and memoise) the whole collection. ~12k docs, so it fits easily."""
    documents = {d["doc_id"]: d for d in _read_jsonl(config.CORPUS_FILE)}
    queries = {q["query_id"]: q for q in _read_jsonl(config.QUERIES_FILE)}

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in _read_jsonl(config.QRELS_FILE):
        qrels[row["query_id"]][row["doc_id"]] = int(row["relevance"])

    hard: dict[str, list[str]] = defaultdict(list)
    if config.HARD_NEG_FILE.exists():
        for row in _read_jsonl(config.HARD_NEG_FILE):
            hard[row["query_id"]].append(row["doc_id"])

    stats = {}
    if config.STATS_FILE.exists():
        stats = json.loads(config.STATS_FILE.read_text(encoding="utf-8"))

    return Collection(
        documents=documents,
        queries=queries,
        qrels=dict(qrels),
        hard_negatives=dict(hard),
        stats=stats,
    )
