"""
STEP 1 OF THE PIPELINE  --  BUILD THE TEST COLLECTION
=====================================================

An Information Retrieval experiment needs three things, always the same three:

    1. CORPUS   - the documents we search through        -> data/corpus.jsonl
    2. QUERIES  - the information needs                  -> data/queries.jsonl
    3. QRELS    - which document is relevant to which query (the "ground truth")
                                                         -> data/qrels.jsonl

This script downloads the public PatentMatch dataset and turns it into exactly
that shape, so every later stage of the project only ever sees these 3 files.

WHERE THE RELEVANCE LABELS COME FROM
------------------------------------
PatentMatch was built from real European Patent Office search reports.  When a
patent examiner rejects a claim for lack of novelty, they must cite the exact
paragraph of an earlier document that already disclosed the invention.  Those
citations are the labels:

    query  = a claim from a new patent application
    doc    = a paragraph from an earlier patent document
    label  = 1 if an examiner cited that paragraph against that claim

So the ground truth is human expert judgement, not a heuristic.  PatentMatch
also gives us the paragraphs an examiner looked at but did NOT consider
novelty-destroying -- those are gold-standard HARD NEGATIVES, which we reuse
later to train the reranker.

HOW WE SPLIT TRAIN / DEV / TEST  (this is the subtle part)
---------------------------------------------------------
A naive random split leaks information: several claims of the same patent
application are near-duplicates of each other and cite the SAME prior-art
paragraphs.  If one lands in train and its twin in test, the test score is
inflated.

We prevent this with a GROUP SPLIT on the bipartite query-document graph:
build a graph with an edge for every (query, relevant doc) pair, find its
connected components, and send a whole component to one split.  After that, no
test query can share a relevant document with any training query.

Run:  python scripts/prepare_data.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"

# Public HuggingFace mirrors of PatentMatch.
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SOURCES = {
    # ~12.7k examiner-confirmed (claim, prior-art paragraph) POSITIVE pairs.
    "pairs": [
        ("bhlim/patentmatch_for_finetuning", "data/train-00000-of-00001.parquet"),
        ("bhlim/patentmatch_for_finetuning", "data/test-00000-of-00001.parquet"),
    ],
    # Labelled pairs: label=1 relevant, label=0 examiner-inspected HARD NEGATIVE.
    "labelled": [
        ("Reynold97/patentmatch_exp", "data/train-00000-of-00001.parquet"),
        ("Reynold97/patentmatch_exp", "data/validation-00000-of-00001.parquet"),
        ("Reynold97/patentmatch_exp", "data/test-00000-of-00001.parquet"),
    ],
    # Small slice that kept the real EPO publication numbers + filing dates.
    "metadata": [
        ("bhlim/patentmatch_train_ultrabalanced_150", "data/train-00000-of-00001.parquet"),
    ],
}

SPLIT_FRACTIONS = {"train": 0.70, "dev": 0.10, "test": 0.20}
SEED = 42

# A claim/paragraph shorter than this is boilerplate ("See figure 3."), not text
# we can retrieve on.
MIN_CHARS = 60


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def download(repo: str, path: str) -> pd.DataFrame:
    url = HF.format(repo=repo, path=path)
    print(f"  downloading {repo}/{path.split('/')[-1]} ...", flush=True)
    return pd.read_parquet(url)


def clean(text: object) -> str:
    """Collapse whitespace and repair the spacing damage in the raw dump."""
    if text is None or not isinstance(text, str):
        return ""
    t = text.replace(" ", " ")
    # The raw files lost the space after sentence-ending periods: "chamber.The"
    t = re.sub(r"(?<=[a-z0-9\)])\.(?=[A-Z])", ". ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def text_id(prefix: str, text: str) -> str:
    """Stable id derived from the text itself, so reruns give identical ids
    and duplicate texts collapse into one entry."""
    key = re.sub(r"[^a-z0-9]+", "", text.lower())
    return f"{prefix}{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def group_split(qrels: list[dict], queries: set[str]) -> dict[str, str]:
    """Assign every query to train/dev/test, keeping connected components of the
    query-document graph together so no relevant document is shared across
    splits."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(("q", q) for q in queries)
    for r in qrels:
        g.add_edge(("q", r["query_id"]), ("d", r["doc_id"]))

    components = []
    for comp in nx.connected_components(g):
        qs = [n[1] for n in comp if n[0] == "q"]
        if qs:
            components.append(qs)

    # Shuffle components, then fill test, then dev, then train. Deterministic.
    random.Random(SEED).shuffle(components)
    n_total = sum(len(c) for c in components)
    targets = {s: f * n_total for s, f in SPLIT_FRACTIONS.items()}

    assignment: dict[str, str] = {}
    filled = {"test": 0, "dev": 0, "train": 0}
    for comp in components:
        if filled["test"] < targets["test"]:
            split = "test"
        elif filled["dev"] < targets["dev"]:
            split = "dev"
        else:
            split = "train"
        filled[split] += len(comp)
        for q in comp:
            assignment[q] = split
    return assignment


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows):,} rows)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print("PatentMatch -> IR test collection\n" + "=" * 60)

    # ---- 1. download ------------------------------------------------------ #
    print("\n[1/5] Downloading source files from HuggingFace")
    pairs = pd.concat([download(r, p) for r, p in SOURCES["pairs"]], ignore_index=True)
    labelled = pd.concat([download(r, p) for r, p in SOURCES["labelled"]], ignore_index=True)
    meta = pd.concat([download(r, p) for r, p in SOURCES["metadata"]], ignore_index=True)
    print(f"  positives={len(pairs):,}  labelled={len(labelled):,}  metadata={len(meta):,}")

    # ---- 2. real EPO publication numbers, where available ----------------- #
    # The metadata slice shares PatentMatch ids with the positive-pair file, so
    # we can recover genuine patent numbers and filing dates for those claims.
    claim_meta: dict[str, dict] = {}
    for row in meta.itertuples(index=False):
        claim = clean(getattr(row, "claim", ""))
        if not claim:
            continue
        date = getattr(row, "application_date", None)
        claim_meta[text_id("Q", claim)] = {
            "patent_id": getattr(row, "patent_application_id", None),
            "filing_date": str(date)[:10] if date is not None else None,
            "epo_link": getattr(row, "doc_link", None),
        }
    doc_meta: dict[str, dict] = {}
    for row in meta.itertuples(index=False):
        cit = clean(getattr(row, "citation", ""))
        if not cit:
            continue
        doc_meta[text_id("D", cit)] = {
            "patent_id": getattr(row, "cited_document_id", None),
            "epo_link": getattr(row, "cited_doc_link", None),
        }
    print(f"  recovered EPO metadata for {len(claim_meta)} claims / {len(doc_meta)} documents")

    # ---- 3. fold every source into corpus / queries / qrels --------------- #
    print("\n[2/5] Normalising into corpus / queries / qrels")
    corpus: dict[str, str] = {}
    query_text: dict[str, str] = {}
    qrels: dict[tuple[str, str], int] = {}
    hard_negatives: set[tuple[str, str]] = set()

    def add(claim: str, doc: str, label: int) -> None:
        claim, doc = clean(claim), clean(doc)
        if len(claim) < MIN_CHARS or len(doc) < MIN_CHARS:
            return
        qid, did = text_id("Q", claim), text_id("D", doc)
        query_text[qid] = claim
        corpus[did] = doc
        if label == 1:
            qrels[(qid, did)] = 1
        else:
            hard_negatives.add((qid, did))

    for row in pairs.itertuples(index=False):
        add(row.anchor, row.positive, 1)
    for row in labelled.itertuples(index=False):
        add(row.text, row.text_b, int(row.label))
    for row in meta.itertuples(index=False):
        add(getattr(row, "claim", ""), getattr(row, "citation", ""),
            int(getattr(row, "invalidates_claim", 1)))

    # A pair cannot be both relevant and a negative; relevance wins.
    hard_negatives = {p for p in hard_negatives if p not in qrels}

    # Drop queries that ended up with no relevant document at all -- they can
    # never be scored, so keeping them would silently depress every metric.
    scorable = {qid for qid, _ in qrels}
    dropped = len(query_text) - len(scorable)
    query_text = {q: t for q, t in query_text.items() if q in scorable}
    hard_negatives = {(q, d) for q, d in hard_negatives if q in scorable}
    print(f"  {len(corpus):,} documents, {len(query_text):,} queries, "
          f"{len(qrels):,} relevance judgements, {len(hard_negatives):,} hard negatives")
    print(f"  dropped {dropped:,} queries that had no relevant document")

    # ---- 4. leakage-free group split -------------------------------------- #
    print("\n[3/5] Group split on the query-document graph (prevents leakage)")
    qrel_rows = [{"query_id": q, "doc_id": d, "relevance": r} for (q, d), r in qrels.items()]
    assignment = group_split(qrel_rows, set(query_text))
    counts: dict[str, int] = defaultdict(int)
    for split in assignment.values():
        counts[split] += 1
    for split in ("train", "dev", "test"):
        print(f"  {split:<6} {counts[split]:>6,} queries")

    # ---- 5. write ---------------------------------------------------------- #
    print("\n[4/5] Writing data/")
    corpus_rows = [
        {
            "doc_id": did,
            "text": text,
            "n_words": len(text.split()),
            **doc_meta.get(did, {}),
        }
        for did, text in sorted(corpus.items())
    ]
    rel_per_query: dict[str, int] = defaultdict(int)
    for q, _ in qrels:
        rel_per_query[q] += 1
    query_rows = [
        {
            "query_id": qid,
            "text": text,
            "split": assignment[qid],
            "n_words": len(text.split()),
            "n_relevant": rel_per_query[qid],
            **claim_meta.get(qid, {}),
        }
        for qid, text in sorted(query_text.items())
    ]

    write_jsonl(DATA / "corpus.jsonl", corpus_rows)
    write_jsonl(DATA / "queries.jsonl", query_rows)
    write_jsonl(DATA / "qrels.jsonl", sorted(qrel_rows, key=lambda r: (r["query_id"], r["doc_id"])))
    write_jsonl(DATA / "hard_negatives.jsonl",
                [{"query_id": q, "doc_id": d} for q, d in sorted(hard_negatives)])

    stats = {
        "source": "PatentMatch (EPO examiner search reports), public HuggingFace mirrors",
        "source_repos": sorted({r for group in SOURCES.values() for r, _ in group}),
        "n_documents": len(corpus_rows),
        "n_queries": len(query_rows),
        "n_qrels": len(qrel_rows),
        "n_hard_negatives": len(hard_negatives),
        "queries_per_split": dict(counts),
        "mean_relevant_per_query": round(len(qrel_rows) / max(1, len(query_rows)), 2),
        "mean_query_words": round(sum(r["n_words"] for r in query_rows) / max(1, len(query_rows)), 1),
        "mean_doc_words": round(sum(r["n_words"] for r in corpus_rows) / max(1, len(corpus_rows)), 1),
        "queries_with_epo_metadata": sum(1 for r in query_rows if r.get("patent_id")),
        "split_method": "connected-component group split on the query-document graph",
        "seed": SEED,
    }
    (DATA / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"  wrote data/dataset_stats.json")

    print("\n[5/5] Done.\n" + "=" * 60)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
