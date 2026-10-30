"""Overview page: the problem, the pipeline, and the headline result."""

from __future__ import annotations

import streamlit as st

from src import config
from ui import theme
from ui.state import get_collection, get_report

theme.inject()

col = get_collection()
report = get_report()
stats = col.stats

theme.page_header(
    "Cross-Encoder Reranking Pipeline for Patent Prior-Art Search",
    "Finding the earlier patents that prove a new application is not novel, by "
    "pairing a fast high-recall retriever with a slow high-precision transformer.",
)

# --------------------------------------------------------------------------- #
# headline numbers
# --------------------------------------------------------------------------- #
if report:
    methods = report["methods"]
    best_key = max(
        methods,
        key=lambda k: methods[k]["metrics"].get("ndcg@10", 0.0),
    )
    best = methods[best_key]
    base = methods.get("bm25", {}).get("metrics", {})
    gain = best["metrics"]["ndcg@10"] - base.get("ndcg@10", 0.0)
    rel = (gain / base["ndcg@10"] * 100) if base.get("ndcg@10") else 0.0

    theme.tiles(
        [
            {
                "label": "nDCG@10 · best system",
                "value": f"{best['metrics']['ndcg@10']:.3f}",
                "delta": f"{rel:+.0f}% vs BM25",
                "hint": best["label"],
                "color": theme.RERANK,
            },
            {
                "label": "MRR · best system",
                "value": f"{best['metrics']['mrr']:.3f}",
                "hint": "mean reciprocal rank of first hit",
                "color": theme.PRIMARY,
            },
            {
                "label": "Recall@100 · stage 1",
                "value": f"{methods.get('hybrid', best)['metrics']['recall@100']:.3f}",
                "hint": "the ceiling the reranker works within",
                "color": theme.FUSION,
            },
            {
                "label": "Corpus",
                "value": f"{report['corpus_size']:,}",
                "hint": "prior-art passages indexed",
                "color": theme.DENSE,
            },
            {
                "label": "Test queries",
                "value": f"{report['n_queries']:,}",
                "hint": "held-out, examiner-labelled",
                "color": theme.LEXICAL,
            },
        ]
    )
else:
    theme.warn(
        "No benchmark results yet. Run <code>python scripts/run_benchmark.py</code> "
        "to populate the headline metrics on this page."
    )

# --------------------------------------------------------------------------- #
# problem
# --------------------------------------------------------------------------- #
theme.section("The problem")

st.markdown(
    "<p style='color:#5B6478;font-size:.92rem;max-width:80ch;margin-bottom:.9rem'>"
    "Before a patent is granted, an examiner must find <b>prior art</b>: earlier "
    "documents that already disclose the claimed invention. Miss it, and an invalid "
    "patent is granted, which later costs years of litigation. Three things make the "
    "search hard.</p>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    theme.card(
        "Queries are long and technical",
        f"A query here is a patent claim, not a few keywords. In our collection the "
        f"average claim is <b>{stats.get('mean_query_words', 0):.0f} words</b> of dense "
        "legal and engineering prose. Classic keyword search was never designed for this.",
    )
with c2:
    theme.card(
        "The same idea, different words",
        "One applicant writes <i>fastener</i>, another writes <i>screw</i>. Applicants "
        "deliberately coin their own terminology, so a purely lexical matcher like BM25 "
        "cannot see that two documents describe the same mechanism. This is "
        "<b>vocabulary mismatch</b>, and it is the dominant failure mode.",
    )
with c3:
    theme.card(
        "Scale rules out the accurate model",
        "The most accurate model, a cross-encoder, must read the query and a document "
        "<i>together</i>. Running it over the whole collection would need one "
        "transformer pass per document, per query &mdash; minutes per search. It cannot "
        "be the only stage.",
    )

# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
theme.section("Our answer: a two-stage cascade")

st.markdown(
    "<p style='color:#5B6478;font-size:.92rem;max-width:80ch;margin-bottom:.4rem'>"
    "Cheap methods look at everything; the expensive model looks only at the shortlist. "
    "Stage 1 is tuned for <b>recall</b> (do not lose the answer), stage 2 for "
    "<b>precision</b> (put the answer first).</p>",
    unsafe_allow_html=True,
)

theme.stage_flow(
    [
        {"n": "input", "h": "Patent claim", "d": "The new application's claim text is the query.",
         "color": theme.INK},
        {"n": "stage 1 · recall", "h": "Retrieve candidates",
         "d": f"BM25 + dense bi-encoder, fused with Reciprocal Rank Fusion. "
              f"{len(col.doc_ids):,} documents &rarr; top 100.",
         "color": theme.FUSION},
        {"n": "stage 2 · precision", "h": "Rerank with a cross-encoder",
         "d": "Claim and candidate read together by a transformer, fine-tuned on "
              "examiner citations with hard negatives. 100 &rarr; reordered.",
         "color": theme.RERANK},
        {"n": "output", "h": "Ranked prior art",
         "d": "Scored with Recall@k, nDCG@10, MAP and MRR against examiner citations.",
         "color": theme.GOOD},
    ]
)

if report:
    # State what the experiment actually found, rather than assuming the cascade
    # helped. This block reads the measured numbers and adapts.
    m = report["methods"]
    stage1_only = {k: v for k, v in m.items() if v["stage2"] == "none"}
    reranked = {k: v for k, v in m.items() if v["stage2"] != "none"}
    if stage1_only and reranked:
        b1 = max(stage1_only.values(), key=lambda v: v["metrics"]["ndcg@10"])
        b2 = max(reranked.values(), key=lambda v: v["metrics"]["ndcg@10"])
        d1, d2 = b1["metrics"]["ndcg@10"], b2["metrics"]["ndcg@10"]
        if d2 > d1:
            theme.note(
                f"<b>Measured outcome.</b> The best reranked system "
                f"(<b>{b2['label']}</b>, nDCG@10 {d2:.4f}) beats the best "
                f"retrieval-only system (<b>{b1['label']}</b>, {d1:.4f}) by "
                f"{(d2 - d1) / d1 * 100:+.1f}%. The Evaluation page reports whether "
                f"that gain survives a significance test."
            )
        else:
            theme.warn(
                f"<b>Measured outcome, reported as found.</b> On this collection the "
                f"best retrieval-only system (<b>{b1['label']}</b>, nDCG@10 {d1:.4f}) "
                f"was <i>not</i> beaten by any reranker (best: <b>{b2['label']}</b>, "
                f"{d2:.4f}). Reranking is what the project set out to test, and this is "
                f"the answer the data gave; the Error Analysis page breaks down why, and "
                f"the Evaluation page shows the cost each method pays."
            )

c1, c2 = st.columns([1, 1])
with c1:
    theme.card(
        "Why the first stage is a hybrid",
        "BM25 and the dense retriever fail on <i>different</i> queries: BM25 nails exact "
        "technical terms and misses paraphrases, dense retrieval does the opposite. "
        "Because their errors are only weakly correlated, fusing their rankings finds "
        "documents neither one finds alone &mdash; and Recall@100 is what caps the whole "
        "pipeline, since stage 2 can only reorder what stage 1 handed it.",
    )
with c2:
    theme.card(
        "Why the second stage is a cross-encoder",
        "A bi-encoder has to compress a document into one vector <i>before</i> it knows "
        "the query. A cross-encoder puts both texts into the same forward pass, so "
        "attention can compare every claim word against every document word. That is a "
        "real accuracy gain, and limiting it to 100 candidates is what makes it "
        "affordable.",
    )

# --------------------------------------------------------------------------- #
# what we built
# --------------------------------------------------------------------------- #
theme.section("What is implemented")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"**Stage 1 &mdash; retrieval** <br>"
        f"{theme.pill('TF-IDF', 'lex')} vector space baseline, cosine on ltc weights<br>"
        f"{theme.pill('BM25', 'lex')} Okapi, written from scratch with per-term "
        f"score explanation<br>"
        f"{theme.pill('Dense', 'den')} MiniLM bi-encoder, mean pooling, FAISS index<br>"
        f"{theme.pill('Hybrid', 'fus')} Reciprocal Rank Fusion and weighted fusion",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"**Stage 2 &mdash; reranking** <br>"
        f"{theme.pill('Learning-to-Rank', 'pri')} 9 ranking features, pairwise "
        f"RankNet loss, from scratch in numpy<br>"
        f"{theme.pill('Cross-encoder', 'rer')} MS MARCO MiniLM, zero-shot<br>"
        f"{theme.pill('Fine-tuned', 'rer')} same model trained on PatentMatch with "
        f"mined hard negatives<br>"
        f"{theme.pill('MaxP', 'rer')} passage splitting for long documents",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"**Evaluation** <br>"
        f"{theme.pill('Recall@k', 'pri')} stage-1 ceiling<br>"
        f"{theme.pill('nDCG@10', 'pri')} headline ranking quality<br>"
        f"{theme.pill('MAP · MRR', 'pri')} all from scratch in "
        f"<code>src/metrics.py</code><br>"
        f"{theme.pill('Significance', 'pri')} paired randomisation test, so gains are "
        f"not just noise<br>"
        f"{theme.pill('Latency', 'pri')} measured per stage, per query",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# how to demo
# --------------------------------------------------------------------------- #
theme.section("Where to look")

theme.note(
    "<b>Dataset</b> &mdash; what we search and where the ground truth comes from. "
    "<b>Live Search</b> &mdash; the demo: pick a real claim and watch the reranker "
    "move documents. <b>Evaluation</b> &mdash; every method on the same queries. "
    "<b>Ablations</b> &mdash; what each piece is worth. "
    "<b>IR Concepts</b> &mdash; the formulas, with calculators running on the real index. "
    "<b>Error Analysis</b> &mdash; the queries we still get wrong, and the reason."
)

theme.kv_list(
    [
        ("Dataset", "PatentMatch &mdash; European Patent Office examiner search reports"),
        ("Documents indexed", f"{len(col.doc_ids):,} prior-art passages"),
        ("Queries", f"{len(col.queries):,} patent claims, "
                    f"{len(col.split_queries('test')):,} held out for testing"),
        ("Relevance labels", f"{stats.get('n_qrels', 0):,} examiner citations"),
        ("Split", "connected-component group split (no shared documents across splits)"),
        ("Stage-1 depth", f"{config.SETTINGS.pipeline.first_stage_k} candidates"),
        ("Team", "Shivnaran S (22PD32) · Diravina E I (22PD12)"),
    ]
)
