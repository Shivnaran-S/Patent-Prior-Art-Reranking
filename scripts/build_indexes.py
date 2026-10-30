"""
STEP 2 OF THE PIPELINE  --  BUILD AND CACHE THE INDEXES.

Three indexes are built once and written to artifacts/, so the Streamlit app
starts in about a second instead of re-encoding 11,849 documents every launch:

    artifacts/bm25.pkl       inverted index + document lengths  (seconds)
    artifacts/tfidf.pkl      L2-normalised tf-idf matrix        (seconds)
    artifacts/dense.npz      11,849 x 384 transformer embeddings (minutes)
    artifacts/doc_terms.pkl  analysed term sets, for LTR features

Run:  python scripts/build_indexes.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_collection      # noqa: E402
from src.pipeline import SearchEngine        # noqa: E402


def main() -> None:
    t_start = time.perf_counter()
    col = load_collection()
    print(f"Collection: {len(col.doc_ids):,} documents, {len(col.queries):,} queries")

    engine = SearchEngine(col)
    engine.build(progress=lambda m: print(f"  {m}", flush=True))

    print("  Caching analysed document term sets ...", flush=True)
    _ = engine.doc_terms

    # Smoke test: the pipeline must actually answer a query.
    qid = col.sample_queries("test", 1)[0]
    query = col.query(qid)["text"]
    for method in ("tfidf", "bm25", "dense", "hybrid"):
        t0 = time.perf_counter()
        hits = engine.retrieve(query, method, 10)
        dt = (time.perf_counter() - t0) * 1000
        gold = col.relevant(qid)
        rank = next((i for i, (d, _) in enumerate(hits, 1) if d in gold), None)
        print(f"  {method:<8} {dt:7.1f} ms   top-10 returned={len(hits):<3} "
              f"gold at rank {rank if rank else '>10'}")

    print(f"\nDone in {time.perf_counter() - t_start:.1f}s. Indexes in artifacts/")


if __name__ == "__main__":
    main()
