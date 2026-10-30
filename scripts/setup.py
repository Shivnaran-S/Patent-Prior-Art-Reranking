"""
ONE COMMAND TO BUILD THE WHOLE PROJECT.

    python scripts/setup.py              # everything (about 60-90 min, mostly CPU training)
    python scripts/setup.py --quick      # skip cross-encoder fine-tuning and ablations

It runs the five stages in order and skips any that are already done, so it is
safe to re-run after an interruption:

    1  prepare_data.py     download PatentMatch, build corpus/queries/qrels
    2  build_indexes.py    BM25, TF-IDF, dense embeddings, FAISS index
    3  train_reranker.py   Learning-to-Rank weights + fine-tuned cross-encoder
    4  run_benchmark.py    evaluate every configuration -> results/benchmark.json
    5  run_ablations.py    depth and long-document ablations -> results/ablations.json

It then runs both test suites, so a broken build is caught here rather than
silently corrupting every number in the project.

Then:   streamlit run app.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


def done(path: Path) -> bool:
    return path.exists()


def step(name: str, script: str, args: list[str], skip_if: Path | None) -> bool:
    print("\n" + "=" * 70)
    if skip_if is not None and done(skip_if):
        print(f"SKIP  {name}  (already built: {skip_if.relative_to(ROOT)})")
        return True
    print(f"RUN   {name}")
    print("=" * 70, flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts" / script), *args],
        cwd=str(ROOT),
    )
    ok = result.returncode == 0
    print(f"\n{'OK' if ok else 'FAILED'}  {name}  "
          f"({(time.perf_counter()-t0)/60:.1f} min)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip cross-encoder fine-tuning and the slow ablations")
    ap.add_argument("--queries", type=int, default=200,
                    help="test queries for the benchmark")
    args = ap.parse_args()

    print("PATENT PRIOR-ART SEARCH  --  full build")
    print(f"Python {sys.version.split()[0]}  |  project at {ROOT}")

    steps = [
        ("1/5  Build the test collection", "prepare_data.py", [], config.CORPUS_FILE),
        ("2/5  Build the indexes", "build_indexes.py", [],
         config.ARTIFACT_DIR / "dense.npz"),
    ]
    if args.quick:
        steps.append(("3/5  Train Learning-to-Rank", "train_reranker.py",
                      ["--ltr", "--no-dev-check"], config.ARTIFACT_DIR / "ltr.json"))
    else:
        steps.append(("3/5  Train both rerankers", "train_reranker.py", [],
                      config.SETTINGS.cross_encoder.finetuned_dir / "config.json"))
    steps.append(("4/5  Run the benchmark", "run_benchmark.py",
                  ["--queries", str(args.queries)],
                  config.RESULTS_DIR / "benchmark.json"))
    if not args.quick:
        steps.append(("5/5  Run the ablations", "run_ablations.py",
                      ["--queries", "80"], config.RESULTS_DIR / "ablations.json"))

    for name, script, extra, marker in steps:
        if not step(name, script, extra, marker):
            print(f"\nStopped at: {name}")
            print("Fix the error above and re-run `python scripts/setup.py` -- "
                  "completed stages are skipped.")
            sys.exit(1)

    # Verify what we just built. Cheap (a few seconds) and it catches a broken
    # collection before it quietly corrupts every number in the project.
    print("\n" + "=" * 70)
    print("VERIFY  Running the test suites")
    print("=" * 70, flush=True)
    for test in ("test_core.py", "test_dataset.py"):
        result = subprocess.run([sys.executable, "-I", str(ROOT / "tests" / test)],
                                cwd=str(ROOT))
        if result.returncode != 0:
            print(f"\n{test} reported failures -- see the output above.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("BUILD COMPLETE.  Start the app with:\n\n    streamlit run app.py\n")
    print("=" * 70)


if __name__ == "__main__":
    main()
