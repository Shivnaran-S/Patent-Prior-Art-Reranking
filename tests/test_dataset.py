"""
Integrity checks on the built test collection.

These are not unit tests of code -- they are assertions about the DATA, run
against whatever `scripts/prepare_data.py` produced.  They exist because the
single most damaging failure in an IR project is a silently broken collection:
a leaky split or a dangling qrel still produces a plausible-looking number, and
nothing downstream will ever complain.

The headline check is the last one: no document may be relevant to queries in
two different splits.  That is the property the connected-component group split
is designed to guarantee, and this test is the proof.

    python tests/test_dataset.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config                        # noqa: E402
from src.dataset import load_collection       # noqa: E402

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


if not config.CORPUS_FILE.exists():
    print("data/corpus.jsonl not found -- run `python scripts/prepare_data.py` first.")
    sys.exit(0)

COL = load_collection()


@check("every qrel points at a real query and a real document")
def _():
    docs, queries = set(COL.documents), set(COL.queries)
    for qid, rels in COL.qrels.items():
        assert qid in queries, f"qrel for unknown query {qid}"
        for doc_id in rels:
            assert doc_id in docs, f"qrel for unknown document {doc_id}"


@check("every hard negative points at a real query and document")
def _():
    docs, queries = set(COL.documents), set(COL.queries)
    for qid, doc_ids in COL.hard_negatives.items():
        assert qid in queries, f"hard negative for unknown query {qid}"
        for doc_id in doc_ids:
            assert doc_id in docs, f"hard negative for unknown document {doc_id}"


@check("no query is a hard negative and a positive for the same document")
def _():
    for qid, doc_ids in COL.hard_negatives.items():
        overlap = set(doc_ids) & COL.relevant(qid)
        assert not overlap, f"{qid}: {overlap} is labelled both relevant and negative"


@check("every query has at least one relevant document")
def _():
    # A query with no relevant document can never be scored, so keeping one
    # would silently depress every metric by a fixed amount.
    empty = [q for q in COL.queries if not COL.relevant(q)]
    assert not empty, f"{len(empty)} queries have no relevance judgement"


@check("no empty or duplicate documents")
def _():
    assert all(t.strip() for t in COL.doc_texts), "a document is empty"
    assert len(set(COL.doc_texts)) == len(COL.doc_texts), "duplicate document text"


@check("every query belongs to exactly one known split")
def _():
    valid = {"train", "dev", "test"}
    for qid, meta in COL.queries.items():
        assert meta["split"] in valid, f"{qid} has split {meta['split']!r}"
    counts = {s: len(COL.split_queries(s)) for s in valid}
    assert sum(counts.values()) == len(COL.queries)
    for split, n in counts.items():
        assert n > 0, f"split {split} is empty"


@check("THE LEAKAGE TEST: no document is relevant across two splits")
def _():
    # This is what the connected-component group split guarantees. If it ever
    # fails, every test number in the project is inflated and meaningless.
    by_split = {}
    for split in ("train", "dev", "test"):
        docs = set()
        for qid in COL.split_queries(split):
            docs |= COL.relevant(qid)
        by_split[split] = docs

    for a, b in (("train", "test"), ("train", "dev"), ("dev", "test")):
        shared = by_split[a] & by_split[b]
        assert not shared, (
            f"{len(shared)} documents are relevant in both {a} and {b} -- "
            f"the split leaks"
        )


@check("document order is stable and positions are consistent")
def _():
    # Every index (BM25 rows, FAISS vectors) assumes the same integer position
    # for the same document, so this must hold across the whole collection.
    for i, doc_id in enumerate(COL.doc_ids):
        assert COL.position(doc_id) == i
    assert len(COL.doc_ids) == len(COL.doc_texts) == len(COL.documents)


@check("the recorded statistics match the data on disk")
def _():
    stats = COL.stats
    if not stats:
        return
    assert stats["n_documents"] == len(COL.documents), "document count drifted"
    assert stats["n_queries"] == len(COL.queries), "query count drifted"
    total_qrels = sum(len(v) for v in COL.qrels.values())
    assert stats["n_qrels"] == total_qrels, "judgement count drifted"


def main() -> int:
    print("Test collection integrity")
    print("=" * 72)
    print(f"  {len(COL.documents):,} documents  |  {len(COL.queries):,} queries  |  "
          f"{sum(len(v) for v in COL.qrels.values()):,} judgements")
    print("-" * 72)
    width = max(len(n) for n, _, _ in _results)
    failures = 0
    for name, ok, err in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}")
        if not ok:
            failures += 1
            print("        " + err.replace("\n", "\n        ").rstrip())
    print("=" * 72)
    print(f"{len(_results) - failures}/{len(_results)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
