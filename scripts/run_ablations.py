"""
STEP 5 OF THE PIPELINE  --  THE EXPENSIVE ABLATIONS -> results/ablations.json

Two questions here need the cross-encoder, which makes them far too slow to
recompute inside the app:

  1. RERANK DEPTH       how many candidates should stage 2 rescore?
                        Quality rises, cost rises linearly. Where is the knee?
  2. LONG DOCUMENTS     truncate the passage, or split it into overlapping
                        windows and keep the best score (MaxP)?

The cheaper ablations (BM25 parameters, fusion weight, preprocessing) are run
live in the app instead, because they take milliseconds.

    python scripts/run_ablations.py --queries 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config                                   # noqa: E402
from src.dataset import load_collection                  # noqa: E402
from src.metrics import aggregate, evaluate_run          # noqa: E402
from src.pipeline import SearchEngine                    # noqa: E402

DEPTHS = [10, 20, 50, 100, 200]
STRATEGIES = ["truncate", "max_passage"]


def evaluate(engine: SearchEngine, qids: list[str], stage2: str, depth: int,
             long_doc: str = "truncate", label: str = "") -> dict:
    col = engine.collection
    per, latencies = [], []
    t0 = time.perf_counter()
    for i, qid in enumerate(qids, 1):
        res = engine.search(col.query(qid)["text"], stage1="hybrid", stage2=stage2,
                            first_stage_k=depth, final_k=max(100, depth),
                            long_doc_strategy=long_doc)
        per.append(evaluate_run([d for d, _ in res.final], col.relevant(qid),
                                config.SETTINGS.eval.cutoffs))
        latencies.append(res.timings_ms["total_ms"])
        if i % 25 == 0:
            print(f"    {label} {i}/{len(qids)}  ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    agg = aggregate(per)
    agg["latency_ms"] = sum(latencies) / len(latencies)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=100)
    args = ap.parse_args()

    engine = SearchEngine(load_collection()).build(progress=lambda m: print(f"  {m}"))
    qids = engine.collection.sample_queries("test", args.queries,
                                            config.SETTINGS.eval.seed)

    # Prefer the fine-tuned checkpoint if it exists -- the ablation should
    # describe the system we actually report.
    stage2 = "cross_encoder"
    try:
        engine.cross_encoder("finetuned")
        stage2 = "cross_encoder_ft"
    except FileNotFoundError:
        pass
    # Warm the models before any timing -- see the note in run_benchmark.py.
    warm = "a warm up claim describing an apparatus with a filter and a housing"
    if engine.dense is not None:
        engine.dense.encode([warm])
        # Each depth must pay the real query-encoding cost; see run_benchmark.py.
        engine.dense.cache_queries = False
        engine.dense._query_cache.clear()
    engine.cross_encoder("finetuned" if stage2.endswith("_ft") else "base").score(
        warm, ["warm"], {"warm": "warm up"})

    print(f"\nAblations on {len(qids)} test queries, stage 2 = {stage2}\n")

    out: dict = {"n_queries": len(qids), "stage2": stage2,
                 "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---- 1. rerank depth --------------------------------------------------- #
    print("[1/2] Rerank depth")
    depth_rows = []
    for depth in DEPTHS:
        print(f"  depth={depth}")
        agg = evaluate(engine, qids, stage2, depth, label=f"depth {depth}")
        depth_rows.append({
            "depth": depth, "ndcg@10": agg["ndcg@10"], "mrr": agg["mrr"],
            "map": agg["map"], "recall@10": agg["recall@10"],
            "recall@100": agg["recall@100"], "latency_ms": agg["latency_ms"],
        })
        print(f"    nDCG@10={agg['ndcg@10']:.4f}  MRR={agg['mrr']:.4f}  "
              f"{agg['latency_ms']:.0f} ms/query")
    out["depth"] = depth_rows

    # ---- 2. long-document handling ---------------------------------------- #
    print("\n[2/2] Long-document handling (depth 50)")
    long_rows = []
    for strategy in STRATEGIES:
        print(f"  {strategy}")
        agg = evaluate(engine, qids, stage2, 50, long_doc=strategy, label=strategy)
        long_rows.append({
            "strategy": strategy, "ndcg@10": agg["ndcg@10"], "mrr": agg["mrr"],
            "recall@10": agg["recall@10"], "latency_ms": agg["latency_ms"],
        })
        print(f"    nDCG@10={agg['ndcg@10']:.4f}  {agg['latency_ms']:.0f} ms/query")
    out["long_doc"] = long_rows

    path = config.RESULTS_DIR / "ablations.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
