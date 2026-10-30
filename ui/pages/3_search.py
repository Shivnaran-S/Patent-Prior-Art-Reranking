"""
Live Search  --  the demonstration page.

The point of this page is to make the cascade visible: run stage 1, show its
ranking, run stage 2, show how far every document MOVED, and report where the
examiner-cited document ended up in each case.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.explain import best_matching_sentence, overlap_stats
from ui import theme
from ui.components import render_results
from ui.state import availability, get_collection, get_engine, get_report

theme.inject()

col = get_collection()
have = availability()

theme.page_header(
    "Live search",
    "Pick a real patent claim, run the two-stage pipeline, and watch what the "
    "reranker changes.",
)

# --------------------------------------------------------------------------- #
# 1. choose a query
# --------------------------------------------------------------------------- #
theme.section("1 · Choose a query")

source = st.radio(
    "Query source",
    ["Example claims (with known answers)", "Any test-split claim", "Type your own"],
    horizontal=True,
    label_visibility="collapsed",
)

query_text = ""
query_id: str | None = None

if source.startswith("Example"):
    options = col.showcase_queries(40) or col.sample_queries("test", 40)
    query_id = st.selectbox(
        "Claim",
        options,
        format_func=lambda q: (
            f"{col.query(q).get('patent_id') or col.query(q)['split']} · "
            f"{col.query(q)['text'][:95]}..."
        ),
    )
    query_text = col.query(query_id)["text"]
elif source.startswith("Any"):
    pool = col.sample_queries("test", 300)
    query_id = st.selectbox(
        "Test claim",
        pool,
        format_func=lambda q: f"{q} · {col.query(q)['text'][:95]}...",
    )
    query_text = col.query(query_id)["text"]
else:
    query_text = st.text_area(
        "Claim text",
        height=120,
        placeholder="Paste a patent claim, or describe an invention in technical "
                    "language...",
    )

if query_id:
    meta = col.query(query_id)
    bits = [theme.pill(f"split: {meta['split']}", "pri"),
            theme.pill(f"{meta['n_words']} words", "irr"),
            theme.pill(f"{meta['n_relevant']} examiner-cited document(s)", "rel")]
    if meta.get("patent_id"):
        link = meta.get("epo_link")
        label = (f"<a href='{link}' target='_blank' style='color:inherit'>"
                 f"{meta['patent_id']}</a>") if link else meta["patent_id"]
        bits.append(theme.pill(label, "den"))
    st.markdown(" ".join(bits), unsafe_allow_html=True)

if query_text:
    st.markdown(
        f"<div class='card card-scroll' style='border-left:3px solid {theme.PRIMARY}'>"
        f"<p>{query_text}</p></div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# 2. configure the pipeline
# --------------------------------------------------------------------------- #
theme.section("2 · Configure the pipeline")

stage1_options = ["tfidf", "bm25"] + (["dense", "hybrid"] if have["dense"] else [])
stage2_options = ["none"]
if have["ltr"]:
    stage2_options.append("ltr")
stage2_options.append("cross_encoder")
if have["finetuned_ce"]:
    stage2_options.append("cross_encoder_ft")

# Default to the configuration that actually scored best in the benchmark,
# rather than assuming the deepest pipeline wins. Everything stays selectable.
report = get_report()
default_s1, default_s2 = "hybrid", "cross_encoder"
if report and report.get("methods"):
    best = max(report["methods"].values(),
               key=lambda v: v["metrics"].get(f"ndcg@{report['primary_cutoff']}", 0.0))
    default_s1, default_s2 = best["stage1"], best["stage2"]
if default_s1 not in stage1_options:
    default_s1 = "hybrid" if "hybrid" in stage1_options else stage1_options[0]
if default_s2 not in stage2_options:
    default_s2 = "cross_encoder" if "cross_encoder" in stage2_options else "none"

c1, c2, c3, c4 = st.columns([1.1, 1.3, 1, 1])
stage1 = c1.selectbox(
    "Stage 1 · candidate retrieval",
    stage1_options,
    index=stage1_options.index(default_s1),
    format_func=lambda m: config.METHOD_LABELS.get(m, m),
)
stage2 = c2.selectbox(
    "Stage 2 · reranking",
    stage2_options,
    index=stage2_options.index(default_s2),
    format_func=lambda m: "None (stage 1 only)" if m == "none"
    else config.METHOD_LABELS.get(m, m),
)
depth = c3.select_slider("Rerank depth", [20, 50, 100, 200], value=50,
                         help="How many stage-1 candidates stage 2 rescores. "
                              "More candidates means higher recall but more "
                              "transformer passes.")
long_doc = c4.selectbox(
    "Long documents",
    ["truncate", "max_passage"],
    format_func=lambda s: "Truncate" if s == "truncate" else "MaxP (split & best)",
    help="Passages longer than the model's input window must be cut, or split "
         "into overlapping windows and scored separately.",
)

run = st.button("Run search", type="primary", use_container_width=False)

# --------------------------------------------------------------------------- #
# 3. run
# --------------------------------------------------------------------------- #
if run and query_text.strip():
    engine = get_engine()
    status = st.empty()
    bar = st.progress(0.0)

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / max(total, 1))
        status.caption(f"Stage 2: scoring candidate {done} of {total} "
                       f"with the cross-encoder ...")

    with st.spinner("Running the pipeline ..."):
        result = engine.search(
            query_text,
            stage1=stage1,
            stage2=stage2,
            first_stage_k=depth,
            final_k=depth,
            long_doc_strategy=long_doc,
            progress=on_progress if stage2.startswith("cross") else None,
            with_components=(stage1 == "hybrid"),
        )
    bar.empty()
    status.empty()
    st.session_state["last_result"] = result
    st.session_state["last_query_id"] = query_id
    st.session_state["last_query_text"] = query_text

result = st.session_state.get("last_result")
if result is None:
    st.info("Choose a query and press **Run search**.")
    st.stop()

query_text = st.session_state["last_query_text"]
query_id = st.session_state.get("last_query_id")
relevant = col.relevant(query_id) if query_id else set()
engine = get_engine()

# --------------------------------------------------------------------------- #
# 4. outcome
# --------------------------------------------------------------------------- #
theme.section("3 · What happened")


def best_rank(ranking, gold):
    for i, (d, _) in enumerate(ranking, start=1):
        if d in gold:
            return i
    return None


r1 = best_rank(result.stage1, relevant)
r2 = best_rank(result.final, relevant)

tiles = [
    {"label": "Stage 1 latency", "value": f"{result.timings_ms['stage1_ms']:.0f} ms",
     "hint": config.METHOD_LABELS.get(result.stage1_method, ""), "color": theme.FUSION},
    {"label": "Stage 2 latency",
     "value": f"{result.timings_ms['stage2_ms']:.0f} ms",
     "hint": f"{len(result.stage1)} candidates rescored", "color": theme.RERANK},
    {"label": "Candidates", "value": f"{len(result.stage1)}",
     "hint": f"from {len(col.doc_ids):,} documents", "color": theme.DENSE},
]
if relevant:
    if r1:
        tiles.append({"label": "Gold rank · after stage 1", "value": f"#{r1}",
                      "color": theme.LEXICAL})
    else:
        tiles.append({"label": "Gold rank · after stage 1", "value": "not found",
                      "hint": f"outside top {len(result.stage1)}", "color": theme.BAD})
    if r2:
        delta = (r1 - r2) if r1 else None
        tiles.append({
            "label": "Gold rank · after stage 2", "value": f"#{r2}",
            "delta": (f"+{delta} places" if delta and delta > 0
                      else (f"{delta} places" if delta else None)),
            "color": theme.GOOD if r2 <= 10 else theme.LEXICAL,
        })
    else:
        tiles.append({"label": "Gold rank · after stage 2", "value": "not found",
                      "color": theme.BAD})
theme.tiles(tiles)

if relevant and r1 and r2 and r2 < r1:
    theme.note(
        f"The cross-encoder moved the examiner-cited document from "
        f"<b>rank #{r1} to rank #{r2}</b>. Stage 1 had already found it, but buried it "
        f"among lexically similar neighbours; reading the claim and the passage "
        f"<i>together</i> is what separated them."
    )
elif relevant and not r1:
    theme.warn(
        f"Stage 1 did not retrieve the examiner-cited document in its top "
        f"{len(result.stage1)}, so stage 2 could not possibly rank it &mdash; a reranker "
        f"can only reorder what it is given. This is precisely why Recall@k is the "
        f"metric that matters for stage 1, and why we increase the depth or fuse two "
        f"retrievers to raise it."
    )

tab_rank, tab_compare, tab_why, tab_stage1 = st.tabs(
    ["Final ranking", "Before vs after", "Why this document scored", "Stage-1 arms"]
)

# --------------------------------------------------------------------------- #
with tab_rank:
    show_n = st.slider("Results to show", 5, min(30, len(result.final)),
                       min(10, len(result.final)), key="show_final")
    render_results(
        result.final,
        query_text,
        col,
        relevant=relevant,
        stage1_ranks=result.stage1_ranks if result.stage2_method != "none" else {},
        score_label=("cross-encoder" if result.stage2_method.startswith("cross")
                     else ("LTR" if result.stage2_method == "ltr" else "score")),
        idf_lookup=engine.idf,
        limit=show_n,
    )

# --------------------------------------------------------------------------- #
with tab_compare:
    if result.stage2_method == "none":
        st.info("Select a stage-2 reranker and run again to compare the two rankings.")
    else:
        theme.section("Rank movement caused by stage 2")
        rows = result.movement()[:20]
        df = pd.DataFrame(
            [
                {
                    "final rank": r["rank"],
                    "stage-1 rank": r["stage1_rank"],
                    "moved": r["delta"] or 0,
                    "relevant": "yes" if r["doc_id"] in relevant else "",
                    "doc_id": r["doc_id"],
                    "passage": col.doc(r["doc_id"])["text"][:110],
                }
                for r in rows
            ]
        )
        st.dataframe(
            df.style.background_gradient(subset=["moved"], cmap="RdYlGn", vmin=-30, vmax=30),
            use_container_width=True, hide_index=True, height=min(430, 40 + 36 * len(df)),
        )

        chart_df = pd.DataFrame(
            [
                {"doc": r["doc_id"][:8], "stage": "1 · retrieval",
                 "rank": r["stage1_rank"], "relevant": r["doc_id"] in relevant}
                for r in rows if r["stage1_rank"]
            ]
            + [
                {"doc": r["doc_id"][:8], "stage": "2 · reranked",
                 "rank": r["rank"], "relevant": r["doc_id"] in relevant}
                for r in rows if r["stage1_rank"]
            ]
        )
        if not chart_df.empty:
            lines = (
                alt.Chart(chart_df)
                .mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=2)
                .encode(
                    x=alt.X("stage:N", title=None,
                            scale=alt.Scale(padding=0.35),
                            axis=alt.Axis(labelFontSize=12, labelFontWeight=600)),
                    y=alt.Y("rank:Q", title="rank (lower is better)",
                            scale=alt.Scale(reverse=True)),
                    detail="doc:N",
                    color=alt.Color(
                        "relevant:N",
                        scale=alt.Scale(domain=[True, False],
                                        range=[theme.GOOD, "#CBD2E0"]),
                        legend=alt.Legend(orient="top", title="examiner-cited"),
                    ),
                    size=alt.Size("relevant:N",
                                  scale=alt.Scale(domain=[True, False], range=[4, 1.5]),
                                  legend=None),
                    tooltip=["doc", "stage", "rank", "relevant"],
                )
                .properties(height=400)
            )
            st.altair_chart(lines, use_container_width=True)
            st.caption(
                "Each line is one document. Green is the document an examiner actually "
                "cited. A line sloping up to the right means the reranker promoted it."
            )

# --------------------------------------------------------------------------- #
with tab_why:
    theme.section("Per-term BM25 contributions")
    st.markdown(
        "<p style='color:#5B6478;font-size:.88rem;max-width:80ch'>"
        "Because BM25 is implemented from scratch, every document's score can be broken "
        "back down into the contribution of each query term. This is the lexical "
        "evidence &mdash; and seeing what it <i>misses</i> is the clearest argument for "
        "the dense and cross-encoder stages.</p>",
        unsafe_allow_html=True,
    )

    # The top results, with the examiner-cited document(s) pulled to the front
    # and added even if the pipeline ranked them outside the top 15 -- those are
    # the most interesting ones to inspect.
    ranked = [d for d, _ in result.final[:15]]
    gold_first = [d for d in sorted(relevant) if d in engine.bm25._pos]
    choices = gold_first + [d for d in ranked if d not in relevant]
    pick = st.selectbox(
        "Document",
        choices,
        format_func=lambda d: (("[examiner-cited] " if d in relevant else "")
                               + col.doc(d)["text"][:85] + "..."),
    )

    contribs = engine.bm25.explain(query_text, pick)
    c1, c2 = st.columns([1.35, 1])
    with c1:
        theme.term_bars(
            [
                {
                    "term": c.term,
                    "value": c.contribution,
                    "detail": f"<span class='doc-id'>idf {c.idf:.2f} · "
                              f"tf {c.tf_in_doc} · df {c.df}</span>",
                }
                for c in contribs
            ],
            value_fmt="{:.2f}",
        )
        total = sum(c.contribution for c in contribs)
        st.caption(f"Total BM25 score from these terms: {total:.2f}")
    with c2:
        stats = overlap_stats(query_text, col.doc(pick)["text"], engine.idf)
        theme.kv_list([
            ("Query index terms", str(stats["query_terms"])),
            ("Matched", f"{stats['matched_terms']} ({stats['coverage']*100:.0f}%)"),
            ("Document words", str(col.doc(pick)["n_words"])),
            ("Examiner-cited", "yes" if pick in relevant else "no"),
        ])
        if stats.get("missing_rare"):
            st.markdown(
                "<div class='card card-tight'><h4>Rarest query terms NOT matched</h4>"
                "<p>" + ", ".join(f"<code>{t}</code>" for t in stats["missing_rare"])
                + "</p></div>",
                unsafe_allow_html=True,
            )

    theme.section("Closest sentence in the passage")
    st.markdown(
        f"<div class='card'><p style='color:#2A3147'>"
        f"{best_matching_sentence(query_text, col.doc(pick)['text'])}</p></div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_stage1:
    if not result.components:
        st.info(
            "Run the search with **Hybrid (RRF)** as stage 1 to compare the two "
            "retrieval arms and their fusion."
        )
    else:
        theme.section("What each retrieval arm found")
        st.markdown(
            "<p style='color:#5B6478;font-size:.88rem;max-width:80ch'>"
            "BM25 and the dense retriever are shown separately, then their Reciprocal "
            "Rank Fusion. Fusion helps exactly when the two disagree: a document one "
            "arm ranks 60th and the other ranks 3rd ends up near the top.</p>",
            unsafe_allow_html=True,
        )

        summary = []
        for name, hits in result.components.items():
            rank = best_rank(hits, relevant)
            summary.append({
                "arm": config.METHOD_LABELS.get(name, name),
                "gold rank": rank if rank else None,
                "found in top 10": "yes" if rank and rank <= 10 else "no",
                "results": len(hits),
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        cols = st.columns(3)
        for column, (name, hits) in zip(cols, result.components.items()):
            with column:
                kind = {"bm25": "lex", "dense": "den", "hybrid": "fus"}[name]
                st.markdown(
                    f"<div style='margin-bottom:.5rem'>"
                    f"{theme.pill(config.METHOD_LABELS.get(name, name), kind)}</div>",
                    unsafe_allow_html=True,
                )
                for i, (doc_id, score) in enumerate(hits[:6], start=1):
                    is_rel = doc_id in relevant
                    st.markdown(
                        f"<div class='result{' relevant' if is_rel else ''}' "
                        f"style='padding:.5rem .65rem'>"
                        f"<div style='display:flex;gap:.45rem;align-items:center'>"
                        f"<div class='rank-badge{' relevant' if is_rel else ''}' "
                        f"style='min-width:1.6rem;height:1.6rem;font-size:.76rem'>{i}</div>"
                        f"<span class='doc-id'>{score:.3f}</span></div>"
                        f"<div style='font-size:.76rem;color:#2A3147;margin-top:.25rem;"
                        f"line-height:1.5'>{col.doc(doc_id)['text'][:115]}...</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
