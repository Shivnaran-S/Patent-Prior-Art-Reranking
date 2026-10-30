"""
THE BENCHMARK RUNNER  --  evaluates every configuration on the same queries.

A comparison is only fair if nothing changes between runs except the method, so
this module:
  * uses ONE deterministic sample of test queries for every method
  * records per-query metrics (needed for the significance test and the error
    analysis page), not just the averages
  * records wall-clock latency per query, so the accuracy/cost trade-off that
    motivates the whole cascade can be plotted rather than asserted

Output goes to results/benchmark.json, which the Streamlit app simply reads --
so the Evaluation page is instant and shows numbers that were actually
measured, not recomputed live with a different random sample.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from . import config
from .dataset import Collection
from .metrics import aggregate, evaluate_run, paired_randomisation_test
from .pipeline import SearchEngine


@dataclass
class RunConfig:
    """One row of the results table."""
    key: str
    label: str
    stage1: str
    stage2: str = "none"
    first_stage_k: int = 100
    long_doc_strategy: str = "truncate"
    notes: str = ""


# The experiment grid.  Reads top to bottom as the story of the project:
# baselines -> better first stage -> fusion -> trained rerankers.
DEFAULT_RUNS: list[RunConfig] = [
    RunConfig("tfidf", "TF-IDF", "tfidf", notes="Vector space baseline"),
    RunConfig("bm25", "BM25", "bm25", notes="Probabilistic lexical baseline"),
    RunConfig("dense", "Dense bi-encoder", "dense", notes="Semantic retrieval, FAISS"),
    RunConfig("hybrid", "Hybrid (RRF)", "hybrid", notes="BM25 + dense, rank fusion"),
    RunConfig("hybrid_ltr", "Hybrid + Learning-to-Rank", "hybrid", "ltr",
              notes="9 ranking features, pairwise RankNet loss. The cheap control: "
                    "if a linear model over retrieval signals matched the "
                    "cross-encoder, the transformer would not be worth its cost."),
    RunConfig("bm25_ce", "BM25 + cross-encoder", "bm25", "cross_encoder",
              notes="Shows reranking on a lexical first stage"),
    RunConfig("hybrid_ce", "Hybrid + cross-encoder", "hybrid", "cross_encoder",
              notes="Zero-shot MS MARCO cross-encoder"),
    RunConfig("hybrid_ce_ft", "Hybrid + cross-encoder (fine-tuned)", "hybrid",
              "cross_encoder_ft", notes="Fine-tuned on PatentMatch hard negatives"),
]


@dataclass
class MethodResult:
    key: str
    label: str
    stage1: str
    stage2: str
    first_stage_k: int
    notes: str
    metrics: dict[str, float] = field(default_factory=dict)
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    stage1_ms: float = 0.0
    stage2_ms: float = 0.0


def run_benchmark(
    engine: SearchEngine,
    runs: list[RunConfig] | None = None,
    query_ids: list[str] | None = None,
    progress=None,
) -> dict:
    """Evaluate each configuration on the same queries and return a report."""
    col: Collection = engine.collection
    cfg = config.SETTINGS.eval
    runs = runs if runs is not None else DEFAULT_RUNS
    query_ids = query_ids or col.sample_queries("test", cfg.n_eval_queries, cfg.seed)

    results: dict[str, MethodResult] = {}
    skipped: dict[str, str] = {}

    for run in runs:
        per_query: dict[str, dict[str, float]] = {}
        latencies: list[float] = []
        s1_times: list[float] = []
        s2_times: list[float] = []
        failed: str | None = None

        for i, qid in enumerate(query_ids):
            query = col.query(qid)["text"]
            try:
                res = engine.search(
                    query,
                    stage1=run.stage1,
                    stage2=run.stage2,
                    first_stage_k=run.first_stage_k,
                    final_k=max(cfg.cutoffs),
                    long_doc_strategy=run.long_doc_strategy,
                )
            except FileNotFoundError as exc:      # model/checkpoint not trained yet
                failed = str(exc)
                break
            run_ids = [d for d, _ in res.final]
            per_query[qid] = evaluate_run(run_ids, col.relevant(qid), cfg.cutoffs, cfg.primary_cutoff)
            latencies.append(res.timings_ms["total_ms"])
            s1_times.append(res.timings_ms["stage1_ms"])
            s2_times.append(res.timings_ms["stage2_ms"])

            if progress is not None:
                progress(run.label, i + 1, len(query_ids))

        if failed:
            skipped[run.key] = failed
            continue

        latencies.sort()
        results[run.key] = MethodResult(
            key=run.key,
            label=run.label,
            stage1=run.stage1,
            stage2=run.stage2,
            first_stage_k=run.first_stage_k,
            notes=run.notes,
            metrics=aggregate(list(per_query.values())),
            per_query=per_query,
            mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            median_latency_ms=latencies[len(latencies) // 2] if latencies else 0.0,
            stage1_ms=sum(s1_times) / len(s1_times) if s1_times else 0.0,
            stage2_ms=sum(s2_times) / len(s2_times) if s2_times else 0.0,
        )

    report = {
        "query_ids": query_ids,
        "n_queries": len(query_ids),
        "cutoffs": list(cfg.cutoffs),
        "primary_cutoff": cfg.primary_cutoff,
        "corpus_size": len(col.doc_ids),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods": {k: asdict(v) for k, v in results.items()},
        "skipped": skipped,
        "significance": significance_table(results, cfg.primary_cutoff),
    }
    return report


def significance_table(
    results: dict[str, MethodResult],
    cutoff: int = 10,
    metrics: tuple[str, ...] | None = None,
) -> list[dict]:
    """Paired randomisation test of every method against the BM25 baseline.

    Run on more than one metric on purpose. nDCG@10 is the headline, but it is
    also the noisiest thing we measure: with about one relevant document per
    query it is zero for most queries and the per-query variance is huge.
    Recall@100 is where a first-stage change shows up most cleanly, so testing
    both says which claims are actually supported and which are merely
    suggestive.
    """
    baseline = results.get("bm25")
    if baseline is None:
        return []
    metrics = metrics or (f"ndcg@{cutoff}", "recall@100", "mrr")
    qids = list(baseline.per_query)

    rows = []
    for metric in metrics:
        base_scores = [baseline.per_query[q].get(metric, 0.0) for q in qids]
        base_mean = sum(base_scores) / max(len(qids), 1)
        for key, res in results.items():
            if key == "bm25" or not res.per_query:
                continue
            scores = [res.per_query.get(q, {}).get(metric, 0.0) for q in qids]
            p = paired_randomisation_test(scores, base_scores)
            delta = (sum(scores) - sum(base_scores)) / max(len(qids), 1)
            rows.append(
                {
                    "method": key,
                    "label": res.label,
                    "metric": metric,
                    "delta_vs_bm25": delta,
                    "relative_pct": (delta / base_mean * 100) if base_mean else 0.0,
                    "p_value": p,
                    "significant_at_05": p < 0.05,
                }
            )
    return sorted(rows, key=lambda r: (r["metric"], -r["delta_vs_bm25"]))


def save_report(report: dict, name: str = "benchmark.json") -> None:
    path = config.RESULTS_DIR / name
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_report(name: str = "benchmark.json") -> dict | None:
    path = config.RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
