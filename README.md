# Cross-Encoder Reranking Pipeline for Patent Prior-Art Search

**Information Retrieval & Web Search Laboratory**
Shivnaran S (22PD32) · Diravina E I (22PD12)
Integrated M.Sc. Data Science, PSG College of Technology

A two-stage **retrieve-and-rerank** search engine for patent prior-art search, with a
Streamlit front end built to demonstrate and explain every stage.

---

## Quick start

On Windows, just double-click **`run.bat`** — it builds the project on first run and
then starts the app.

To do it by hand, in the project's own virtual environment (recommended, so the project
never fights with whatever Python packages you have installed system-wide):

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/setup.py
.venv\Scripts\python -m streamlit run app.py
```

On macOS or Linux, replace `.venv\Scripts\python` with `.venv/bin/python`.

> **Why a venv.** This project needs `numpy >= 2` (pyarrow, which reads the dataset's
> parquet files, requires it). Some other libraries — `stable-baselines3`, for one — pin
> `numpy < 2`, so installing this project's requirements globally can break them. The
> `.venv` keeps the two apart.

**When submitting or zipping this project, exclude `.venv/` and `artifacts/`.** They are
several GB of installed packages and cached model weights, and both are rebuilt by
`scripts/setup.py`. The source, the dataset in `data/` and the results in `results/` are
all that is needed.

`setup.py` runs the whole build in order and **skips stages that are already done**, so
it is safe to re-run after an interruption. On a typical 8-core laptop expect roughly:

| Stage | Time |
|---|---|
| Download the dataset, build the collection | ~3 min |
| BM25 + TF-IDF + 11,849 dense embeddings | ~10 min |
| Mine hard negatives, train both rerankers | ~55 min |
| Benchmark, 8 configurations × 200 queries | ~30 min |
| Ablations | ~20 min |

So about **two hours** end to end, almost all of it CPU transformer work. Use `--quick`
to skip the cross-encoder fine-tuning and the slow ablations — roughly 15 minutes, and
every page of the app still works (the fine-tuned reranker simply does not appear).

Everything runs on **CPU**. No GPU, no API key, no paid service.

---

## What the project does

Given a patent claim, find the earlier documents that disclose the same invention.

```
query claim
    │
    ▼
┌─────────────────────────────────────────────┐
│ STAGE 1 — candidate retrieval  (high RECALL)│
│   TF-IDF │ BM25 │ Dense │ Hybrid (RRF)      │
│   11,849 documents  ──►  top 100            │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ STAGE 2 — reranking         (high PRECISION)│
│   Learning-to-Rank │ Cross-encoder          │
│   top 100  ──►  reordered                   │
└─────────────────────────────────────────────┘
    │
    ▼
ranked prior art + Recall@k, nDCG@10, MAP, MRR
```

**Why two stages.** The accurate model (a cross-encoder) must read the query and a
document *together*, so it cannot precompute anything — scoring the whole collection
would take minutes per query. The cheap models cover everything; the expensive one sees
only the shortlist.

---

## Dataset

**PatentMatch** — built from real European Patent Office examiner search reports. When
an examiner rejects a claim for lack of novelty they must cite the exact paragraph of an
earlier document that disclosed it. Those citations are our relevance labels.

| | |
|---|---|
| Documents | 11,849 prior-art passages |
| Queries | 14,297 patent claims |
| Relevance judgements | 14,650 examiner citations |
| Examiner-rejected pairs | 246 gold hard negatives |
| Split | connected-component group split (train 70 / dev 10 / test 20) |

**The split prevents leakage.** Several claims of one application are near-duplicates and
cite the *same* prior art. A random split would put twins on both sides and inflate the
test score. We build a bipartite query–document graph and assign whole connected
components to one split, so no test query shares a relevant document with any training
query.

**Why not HUPD.** Our proposal named HUPD as the corpus. HUPD has 4.5M applications with
full text but **no citation field**, and the PatentsView citation API is no longer openly
reachable, so relevance labels cannot be derived from it. PatentMatch provides genuine
examiner judgements at passage level, which is what an evaluation needs. The pipeline
itself is corpus-agnostic — `scripts/prepare_data.py` is the only file that would change.

---

## Methods implemented

**Stage 1 — retrieval**

| Method | Notes |
|---|---|
| TF-IDF | vector space, SMART `ltc` weighting, cosine — written from scratch |
| BM25 | Okapi, from scratch, with per-term score explanation |
| Dense | MiniLM bi-encoder, mean pooling, FAISS `IndexFlatIP` (exact cosine) |
| Hybrid | Reciprocal Rank Fusion, and weighted score fusion |

**Stage 2 — reranking**

| Method | Notes |
|---|---|
| Learning-to-Rank | 9 ranking features, pairwise RankNet loss, from scratch in numpy |
| Cross-encoder | `ms-marco-MiniLM-L-6-v2`, zero-shot |
| Cross-encoder (fine-tuned) | same model trained on PatentMatch with mined hard negatives |
| MaxP | overlapping-window scoring for passages past the input limit |

**Hard negative mining.** Random negatives make the task trivial. For every training
query we run the real first stage and take its top-ranked documents that the examiner did
*not* cite, plus the paragraphs examiners inspected and rejected. The reranker is trained
on exactly the decisions it faces at inference time.

**Evaluation.** Recall@k, Precision@k, MRR, MAP and nDCG@k, all implemented in
`src/metrics.py`, plus a **paired randomisation test** so reported gains are shown not to
be noise, and per-stage latency so the accuracy/cost trade-off can be plotted.

---

## The app

```bash
streamlit run app.py
```

| Page | What it shows |
|---|---|
| **Overview** | the problem, the pipeline, headline numbers |
| **Dataset** | provenance of the labels, the split, browse the collection |
| **Live Search** | the demo — run the pipeline and watch documents move |
| **Evaluation** | every method on the same queries, charts, significance, cost |
| **Ablations** | live sliders for BM25 `k1`/`b`, fusion weight, preprocessing; precomputed depth and MaxP studies |
| **IR Concepts** | every formula with a calculator running on the real index |
| **Error Analysis** | which stage caused each failure, and what correlates with failure |

---

## Repository layout

```
app.py                       Streamlit entry point (st.navigation)
config.toml → .streamlit/    theme

src/                         all logic — one module per concept
  config.py                  every tunable parameter, in one place
  text_utils.py              tokenise → stopwords → stem  (shared by query & doc)
  dataset.py                 loads corpus / queries / qrels
  tfidf_retriever.py         TF-IDF, from scratch
  bm25_retriever.py          BM25, from scratch, with .explain()
  dense_retriever.py         bi-encoder + FAISS
  hybrid_retriever.py        RRF and weighted fusion
  cross_encoder_reranker.py  stage 2 transformer, truncate / MaxP
  features.py                the 9 Learning-to-Rank features
  ltr_reranker.py            pairwise RankNet, from scratch in numpy
  pipeline.py                SearchEngine — ties both stages together, times them
  metrics.py                 Recall / Precision / MRR / MAP / nDCG + significance test
  evaluate.py                the benchmark grid
  explain.py                 highlighting, overlap stats, rank movement

scripts/
  setup.py                   runs everything below, in order
  prepare_data.py            PatentMatch  →  data/corpus|queries|qrels.jsonl
  build_indexes.py           BM25, TF-IDF, embeddings, FAISS  →  artifacts/
  train_reranker.py          LTR weights + fine-tuned cross-encoder
  run_benchmark.py           all methods  →  results/benchmark.json
  run_ablations.py           depth & MaxP studies  →  results/ablations.json
  make_abstract.py           regenerates the abstract .docx from real results

ui/                          Streamlit theme, components, and the seven pages
tests/test_core.py           32 correctness tests, no pytest needed
tests/test_dataset.py        9 integrity checks on the built collection

data/        corpus.jsonl, queries.jsonl, qrels.jsonl, hard_negatives.jsonl
artifacts/   cached indexes, embeddings, trained models
results/     benchmark.json, ablations.json

DEMO_GUIDE.md                what to show and say, page by page, on demo day
```

---

## Running stages individually

```bash
python scripts/prepare_data.py                   # rebuild the collection
python scripts/build_indexes.py                  # rebuild indexes
python scripts/train_reranker.py --ltr           # just the fast linear reranker
python scripts/train_reranker.py --ce --epochs 2 # just the cross-encoder
python scripts/run_benchmark.py --queries 200    # evaluate
python scripts/run_benchmark.py --fast           # evaluate, skipping cross-encoders
python scripts/run_ablations.py --queries 80     # depth + MaxP ablations
python scripts/make_abstract.py                  # regenerate the abstract
python tests/test_core.py                        # 32 correctness tests
python tests/test_dataset.py                     # 9 data-integrity checks
```

---

## Tests

```bash
python tests/test_core.py        # 32 correctness tests
python tests/test_dataset.py     # 9 integrity checks on the built collection
```

41 tests, no pytest required. Every one checks a value **computed by hand from the
formula** on a four-document toy collection — because a retriever that is subtly wrong
still returns a plausible-looking ranked list, so "it looks reasonable" proves nothing.

They cover: the five metrics against hand-computed values, the permutation test at both
extremes, BM25's IDF against the Robertson/Sparck-Jones formula, term-frequency
saturation, the effect of `b` on length normalisation, that `explain()`'s per-term
contributions sum exactly to the document's score, that a saved-and-reloaded index
behaves identically, TF-IDF's cosine range and L2 normalisation, RRF's rank-only
behaviour and scale invariance, and the Learning-to-Rank model's convergence.

`test_dataset.py` validates the collection itself — that every relevance judgement
points at a real query and document, that no query is unscoreable, and, most importantly,
**that no document is relevant to queries in two different splits**. That last one is the
proof that the group split does what it claims; if it ever fails, every number in the
project is inflated.

The test suite caught `explain()` indexing a CSC matrix by row and reading the wrong term
frequencies — so the per-term BM25 contributions shown in the app were simply wrong
numbers, with nothing to indicate it.

**The costliest bug the evaluation harness caught.** Only ~40% of examiner-cited
paragraphs are retrievable in the stage-1 pool. Training the rerankers on *all* cited
paragraphs means "relevant" silently comes to mean *"the retrievers scored this low"* —
the one thing the unretrievable 60% have in common — while every negative is, by
construction, a document the retrievers scored high. Both rerankers learned the inverted
rule. The fine-tuned cross-encoder scored **nDCG@10 = 0.010** on dev against **0.095** for
no reranking at all, and *reversing* its output improved it.

Nothing errored. The loss curve fell smoothly from 1.23 to 0.36 and the model looked
healthy. It was caught only by scoring against a held-out split and asking why the number
moved the wrong way. The fix — train on in-pool positives only, mined at the depth used
at inference — is in `scripts/train_reranker.py`, with the reasoning written out where
the filter is applied.

---

## Notes and honest limitations

- **Absolute scores are modest, and that is expected.** Claim-to-prior-art matching has
  very low lexical overlap by nature, and many genuinely relevant patents are never cited
  by an examiner, so unlabelled correct answers count as errors. Every citation-derived
  prior-art benchmark is pessimistic in absolute terms; the *comparison between methods*
  is the meaningful result.
- **Recall@k is a hard ceiling.** Anything stage 1 misses cannot be recovered by stage 2.
  This is measured and reported explicitly on the Error Analysis page.
- **CPU training was the binding constraint** on the cross-encoder, not data — two epochs
  of a 22M-parameter model. A larger backbone and a listwise loss are the obvious next
  steps.
- **Model downloads.** The first run downloads two models (~180 MB total) from
  HuggingFace. After that everything works offline.
