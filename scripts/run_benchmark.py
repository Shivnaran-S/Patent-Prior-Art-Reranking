"""
STEP 4 OF THE PIPELINE  --  EVALUATE EVERYTHING AND WRITE results/benchmark.json

Runs all eight configurations on the SAME sample of held-out test queries and
records per-query metrics, latency per stage, and a paired randomisation test
against the BM25 baseline.  The Streamlit Evaluation page just reads the file.

    python scripts/run_benchmark.py                 # 300 queries (default)
    python scripts/run_benchmark.py --queries 150   # faster
    python scripts/run_benchmark.py --fast          # skip cross-encoder runs

The cross-encoder runs dominate the time: 300 queries x 100 candidates is
30,000 transformer passes per configuration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config                                      # noqa: E402
from src.dataset import load_collection                     # noqa: E402
from src.evaluate import DEFAULT_RUNS, run_benchmark, save_report  # noqa: E402
from src.pipeline import SearchEngine                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=config.SETTINGS.eval.n_eval_queries)
    ap.add_argument("--depth", type=int, default=config.SETTINGS.pipeline.first_stage_k)
    ap.add_argument("--fast", action="store_true",
                    help="skip the cross-encoder configurations")
    args = ap.parse_args()

    engine = SearchEngine(load_collection()).build(progress=lambda m: print(f"  {m}"))
    qids = engine.collection.sample_queries("test", args.queries,
                                            config.SETTINGS.eval.seed)

    runs = [r for r in DEFAULT_RUNS if not (args.fast and "ce" in r.key)]
    for r in runs:
        r.first_stage_k = args.depth

    print(f"\nEvaluating {len(runs)} configurations on {len(qids)} test queries "
          f"(depth {args.depth}, corpus {len(engine.collection.doc_ids):,})")

    # Warm up every model BEFORE timing anything. Loading a transformer takes
    # tens of seconds; if that landed inside the first query it would be
    # amortised across the run and the latency column -- and the accuracy/cost
    # plot built from it -- would be wrong.
    print("Warming up models so the latency numbers are steady-state ...")
    # Deliberately NOT one of the evaluation queries: the dense retriever caches
    # query embeddings, so warming on a real query would make that one query
    # unfairly fast and skew its latency.
    warm = "a warm up claim describing an apparatus with a filter and a housing"
    if engine.dense is not None:
        engine.dense.encode([warm])
        # Every method must pay the real query-encoding cost, or whichever runs
        # second would look faster purely from a warm cache.
        engine.dense.cache_queries = False
        engine.dense._query_cache.clear()
    for variant in ("base", "finetuned"):
        try:
            engine.cross_encoder(variant).score(warm, ["warm"], {"warm": "warm up"})
        except FileNotFoundError:
            pass
    print()

    state = {"label": None, "t0": time.perf_counter()}

    def progress(label: str, done: int, total: int) -> None:
        if label != state["label"]:
            state["label"] = label
            state["t0"] = time.perf_counter()
            print(f"  {label}")
        if done % 25 == 0 or done == total:
            elapsed = time.perf_counter() - state["t0"]
            eta = elapsed / done * (total - done) if done else 0
            print(f"    {done}/{total}  {elapsed:.0f}s elapsed, {eta:.0f}s left",
                  flush=True)

    t0 = time.perf_counter()
    report = run_benchmark(engine, runs=runs, query_ids=qids, progress=progress)
    save_report(report)

    print(f"\n{'=' * 78}")
    print(f"{'method':<40} {'nDCG@10':>8} {'MRR':>7} {'MAP':>7} {'R@10':>7} "
          f"{'R@100':>7} {'ms/q':>8}")
    print("-" * 78)
    for key, m in report["methods"].items():
        met = m["metrics"]
        print(f"{m['label']:<40} {met['ndcg@10']:>8.4f} {met['mrr']:>7.4f} "
              f"{met['map']:>7.4f} {met['recall@10']:>7.4f} "
              f"{met['recall@100']:>7.4f} {m['mean_latency_ms']:>8.1f}")
    print("=" * 78)

    if report.get("significance"):
        print("\nPaired randomisation test vs BM25 (nDCG@10):")
        for r in report["significance"]:
            mark = "significant" if r["significant_at_05"] else "not significant"
            print(f"  {r['label']:<40} {r['delta_vs_bm25']:+.4f} "
                  f"({r['relative_pct']:+6.1f}%)  p={r['p_value']:.4f}  {mark}")

    if report.get("skipped"):
        print("\nSkipped:")
        for key, reason in report["skipped"].items():
            print(f"  {key}: {reason}")

    print(f"\nWrote results/benchmark.json in {(time.perf_counter()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
