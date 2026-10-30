"""
Error Analysis page.

Averages say how well the system does; this page says *why* it fails. Every
failure on this collection falls into one of two buckets, and the distinction
decides what you would fix next:

  STAGE 1 MISS  -- the relevant document never entered the candidate pool.
                   Unrecoverable by any reranker. Fix = better recall.
  STAGE 2 MISS  -- stage 1 found it, but the reranker left it below rank 10.
                   Fix = a better reranker.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.explain import highlight_overlap, overlap_stats
from ui import theme
from ui.state import get_collection, get_engine, get_report, missing_banner

theme.inject()

col = get_collection()
report = get_report()

theme.page_header(
    "Error analysis",
    "Where the pipeline still fails, which stage is responsible, and what the "
    "failures have in common.",
)

if not report:
    missing_banner(["benchmark"])
    st.stop()

methods = report["methods"]
primary = report["primary_cutoff"]
order = [k for k in ["tfidf", "bm25", "dense", "hybrid", "hybrid_ltr", "bm25_ce",
                     "hybrid_ce", "hybrid_ce_ft"] if k in methods]

best_key = max(order, key=lambda k: methods[k]["metrics"][f"ndcg@{primary}"])

tabs = st.tabs(["Which stage is to blame", "Failure cases", "What correlates with failure"])

# =========================================================================== #
with tabs[0]:
    theme.section("Attributing every failure to a stage")

    system = st.selectbox("System to analyse", order,
                          index=order.index(best_key),
                          format_func=lambda k: methods[k]["label"])
    stage1_key = methods[system]["stage1"]
    stage1_run = next((k for k in order if k == stage1_key), None)

    per_q = methods[system]["per_query"]
    # The matching stage-1-only run tells us whether the document was ever found.
    recall_source = methods.get(stage1_run, {}).get("per_query", per_q)

    # When the chosen system has no reranker, "stage 2 miss" would be a
    # misnomer -- the ranking failure is stage 1's own. Label it accordingly.
    has_stage2 = methods[system]["stage2"] != "none"
    RANK_MISS = ("Stage 2 miss · retrieved but ranked low" if has_stage2
                 else "Ranking miss · retrieved but ranked low")
    POOL_MISS = ("Stage 1 miss · never retrieved" if has_stage2
                 else "Retrieval miss · never found")
    SUCCESS = "Success · relevant doc in top 10"

    rows = []
    for qid, m in per_q.items():
        found_in_pool = recall_source.get(qid, {}).get("recall@100", 0.0) > 0
        in_top10 = m["recall@10"] > 0
        if in_top10:
            bucket = SUCCESS
        elif found_in_pool:
            bucket = RANK_MISS
        else:
            bucket = POOL_MISS
        rows.append({"query_id": qid, "bucket": bucket,
                     "ndcg": m[f"ndcg@{primary}"], "rr": m["mrr"],
                     "rank": m["first_relevant_rank"] or None})
    bdf = pd.DataFrame(rows)
    counts = bdf["bucket"].value_counts().reindex([SUCCESS, RANK_MISS, POOL_MISS],
                                                  fill_value=0)

    theme.tiles([
        {"label": "Success", "value": f"{counts.iloc[0]}",
         "hint": f"{counts.iloc[0]/len(bdf)*100:.0f}% of queries", "color": theme.GOOD},
        {"label": "Ranked too low" if not has_stage2 else "Stage 2 miss",
         "value": f"{counts.iloc[1]}",
         "hint": "retrieved, but below rank 10", "color": theme.LEXICAL},
        {"label": "Never retrieved" if not has_stage2 else "Stage 1 miss",
         "value": f"{counts.iloc[2]}",
         "hint": "not in the candidate pool at all", "color": theme.BAD},
        {"label": "Recoverable", "value": f"{counts.iloc[1]/max(len(bdf)-counts.iloc[0],1)*100:.0f}%",
         "hint": "of failures, by better ranking alone", "color": theme.PRIMARY},
    ])

    c1, c2 = st.columns([1, 1.15])
    with c1:
        pie_df = counts.reset_index()
        pie_df.columns = ["bucket", "queries"]
        donut = (
            alt.Chart(pie_df)
            .mark_arc(innerRadius=62, outerRadius=105)
            .encode(
                theta="queries:Q",
                color=alt.Color(
                    "bucket:N",
                    scale=alt.Scale(domain=list(counts.index),
                                    range=[theme.GOOD, theme.LEXICAL, theme.BAD]),
                    legend=alt.Legend(orient="bottom", title=None, columns=1),
                ),
                tooltip=["bucket", "queries"],
            )
            .properties(height=300)
        )
        st.altair_chart(donut, use_container_width=True)
    with c2:
        theme.card(
            "Why the split matters",
            f"<b>{counts.iloc[2]} queries</b> never had the examiner-cited document "
            f"among the {methods[system]['first_stage_k']} candidates at all, so no "
            f"amount of reranking could have saved them. Those are fixed by raising "
            f"recall &mdash; deeper retrieval, a better dense model, more retrieval arms."
            f"<br><br><b>{counts.iloc[1]} queries</b> did have it in the pool and still "
            f"left it below rank 10. "
            + ("Those are the ones a better reranker would win, and they are the honest "
               "measure of how much headroom the current model leaves."
               if has_stage2 else
               "This system does no reranking, so that gap is exactly the opportunity a "
               "stage-2 reranker exists to capture."),
        )
        ceiling = methods.get(stage1_run, {}).get("metrics", {}).get("recall@100")
        if ceiling:
            theme.note(
                f"A perfect ranking of this candidate pool would reach Recall@10 = "
                f"<b>{ceiling:.3f}</b>, because that is the pool's Recall@100. This "
                f"system reaches <b>{methods[system]['metrics']['recall@10']:.3f}</b>, so "
                f"it converts "
                f"<b>{methods[system]['metrics']['recall@10']/ceiling*100:.0f}%</b> of the "
                f"headroom the first stage left it."
            )

    theme.section("Where the first relevant document lands")
    found = bdf.dropna(subset=["rank"])
    if not found.empty:
        hist = (
            alt.Chart(found[found["rank"] <= 100])
            .mark_bar(color=theme.PRIMARY)
            .encode(
                x=alt.X("rank:Q", bin=alt.Bin(maxbins=40),
                        title="rank of the first examiner-cited document"),
                y=alt.Y("count():Q", title="queries"),
                tooltip=["count()"],
            )
            .properties(height=250)
        )
        st.altair_chart(hist, use_container_width=True)

# =========================================================================== #
with tabs[1]:
    theme.section("Individual failures, with the text")
    engine = get_engine()

    system = st.selectbox("System", order, index=order.index(best_key),
                          format_func=lambda k: methods[k]["label"], key="fail_sys")
    per_q = methods[system]["per_query"]
    kind = st.radio(
        "Failure type", ["Stage 2 miss (found but ranked low)", "Stage 1 miss (never found)",
                         "Biggest wins"],
        horizontal=True,
    )

    stage1_key = methods[system]["stage1"]
    recall_source = methods.get(stage1_key, {}).get("per_query", per_q)

    if kind.startswith("Stage 2"):
        pool = [q for q, m in per_q.items()
                if m["recall@10"] == 0 and recall_source.get(q, {}).get("recall@100", 0) > 0]
        pool.sort(key=lambda q: per_q[q]["first_relevant_rank"] or 999)
    elif kind.startswith("Stage 1"):
        pool = [q for q, m in per_q.items()
                if recall_source.get(q, {}).get("recall@100", 0) == 0]
    else:
        base = methods.get("bm25", {}).get("per_query", {})
        pool = sorted(
            [q for q in per_q if q in base],
            key=lambda q: -(per_q[q][f"ndcg@{primary}"] - base[q][f"ndcg@{primary}"]),
        )

    st.caption(f"{len(pool)} queries in this group.")
    if not pool:
        st.info("No queries in this group.")
    else:
        qid = st.selectbox(
            "Query", pool[:120],
            format_func=lambda q: (
                f"rank {int(per_q[q]['first_relevant_rank']) if per_q[q]['first_relevant_rank'] else '-'} · "
                f"{col.query(q)['text'][:85]}..."),
        )
        qtext = col.query(qid)["text"]
        gold = col.relevant(qid)

        st.markdown(
            f"<div class='card' style='border-left:3px solid {theme.PRIMARY}'>"
            f"<h4>Query claim</h4><p style='color:#2A3147'>{qtext}</p></div>",
            unsafe_allow_html=True,
        )

        for doc_id in sorted(gold):
            dtext = col.doc(doc_id)["text"]
            stats = overlap_stats(qtext, dtext, engine.idf)
            bm_rank = next(
                (i for i, (d, _) in enumerate(engine.bm25.search(qtext, 200), 1)
                 if d == doc_id), None)
            dn_rank = None
            if engine.dense is not None:
                dn_rank = next(
                    (i for i, (d, _) in enumerate(engine.dense.search(qtext, 200), 1)
                     if d == doc_id), None)

            st.markdown(
                f"<div class='card' style='border-left:3px solid {theme.GOOD}'>"
                f"<h4>What the examiner cited &mdash; and we missed</h4>"
                f"<div class='result-body'>{highlight_overlap(dtext, qtext, 150)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lexical coverage", f"{stats['coverage']*100:.0f}%",
                      help="Share of query index terms present in this document.")
            c2.metric("BM25 rank", f"#{bm_rank}" if bm_rank else ">200")
            c3.metric("Dense rank", f"#{dn_rank}" if dn_rank else ">200")
            c4.metric("Final rank",
                      f"#{int(per_q[qid]['first_relevant_rank'])}"
                      if per_q[qid]["first_relevant_rank"] else "not in top 100")

            if stats.get("missing_rare"):
                st.markdown(
                    "<div class='card card-tight'><h4>Rarest query terms absent from "
                    "this document</h4><p>"
                    + ", ".join(f"<code>{t}</code>" for t in stats["missing_rare"])
                    + "</p></div>",
                    unsafe_allow_html=True,
                )
            if stats["coverage"] < 0.25:
                theme.warn(
                    f"This document shares only <b>{stats['coverage']*100:.0f}%</b> of "
                    f"the query's index terms, yet an expert judged it novelty-destroying. "
                    f"No lexical method can be expected to find this &mdash; it is a pure "
                    f"vocabulary-mismatch case, and the strongest argument in the project "
                    f"for semantic retrieval."
                )

        if kind == "Biggest wins":
            theme.section("What our system put at the top instead")
            res = get_engine().search(
                qtext, stage1=methods[system]["stage1"], stage2=methods[system]["stage2"],
                first_stage_k=50, final_k=5,
            )
            from ui.components import render_results
            render_results(res.final, qtext, col, relevant=gold,
                           stage1_ranks=res.stage1_ranks, idf_lookup=engine.idf, limit=5)

# =========================================================================== #
with tabs[2]:
    theme.section("What makes a query hard?")
    st.markdown(
        "<p style='color:#5B6478;font-size:.9rem;max-width:82ch'>Rather than guess, we "
        "correlate per-query success against measurable properties of the query and its "
        "answer.</p>",
        unsafe_allow_html=True,
    )

    system = st.selectbox("System", order, index=order.index(best_key),
                          format_func=lambda k: methods[k]["label"], key="corr_sys")
    per_q = methods[system]["per_query"]

    from src.text_utils import analyze

    rows = []
    for qid, m in per_q.items():
        if qid not in col.queries:
            continue
        q_terms = set(analyze(col.query(qid)["text"]))
        gold = col.relevant(qid)
        coverages, lengths = [], []
        for d in gold:
            if d in col.documents:
                d_terms = set(analyze(col.doc(d)["text"]))
                coverages.append(len(q_terms & d_terms) / max(len(q_terms), 1))
                lengths.append(col.doc(d)["n_words"])
        rows.append({
            "query_id": qid,
            "query words": col.query(qid)["n_words"],
            "query index terms": len(q_terms),
            "lexical coverage of answer": max(coverages) if coverages else 0.0,
            "answer length": max(lengths) if lengths else 0,
            f"nDCG@{primary}": m[f"ndcg@{primary}"],
            "success": "found in top 10" if m["recall@10"] > 0 else "missed",
        })
    cdf = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        scatter = (
            alt.Chart(cdf)
            .mark_circle(size=70, opacity=0.6)
            .encode(
                x=alt.X("lexical coverage of answer:Q",
                        title="lexical coverage of the correct answer",
                        axis=alt.Axis(format="%")),
                y=alt.Y(f"nDCG@{primary}:Q"),
                color=alt.Color("success:N",
                                scale=alt.Scale(domain=["found in top 10", "missed"],
                                                range=[theme.GOOD, theme.BAD]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["query words", "lexical coverage of answer",
                         f"nDCG@{primary}"],
            )
            .properties(height=320)
        )
        st.altair_chart(scatter, use_container_width=True)
        st.caption("Each point is one test query.")
    with c2:
        binned = cdf.copy()
        binned["coverage band"] = pd.cut(
            binned["lexical coverage of answer"],
            bins=[-0.01, 0.1, 0.2, 0.3, 0.5, 1.0],
            labels=["0-10%", "10-20%", "20-30%", "30-50%", "50%+"],
        )
        agg = (binned.groupby("coverage band", observed=True)
               .agg(**{f"mean nDCG@{primary}": (f"nDCG@{primary}", "mean"),
                       "queries": ("query_id", "count")})
               .reset_index())
        bars = (
            alt.Chart(agg)
            .mark_bar(cornerRadiusEnd=4, color=theme.PRIMARY)
            .encode(
                x=alt.X("coverage band:N", title="lexical coverage of the answer",
                        sort=["0-10%", "10-20%", "20-30%", "30-50%", "50%+"]),
                y=alt.Y(f"mean nDCG@{primary}:Q"),
                tooltip=["coverage band", f"mean nDCG@{primary}", "queries"],
            )
            .properties(height=320)
        )
        st.altair_chart(bars, use_container_width=True)
        st.dataframe(agg.style.format({f"mean nDCG@{primary}": "{:.4f}"}),
                     use_container_width=True, hide_index=True)

    corr = cdf[["query words", "query index terms", "lexical coverage of answer",
                "answer length", f"nDCG@{primary}"]].corr()[f"nDCG@{primary}"].drop(
        f"nDCG@{primary}")
    theme.section("Correlation with ranking quality")
    corr_df = corr.reset_index()
    corr_df.columns = ["property", "correlation with nDCG"]
    cchart = (
        alt.Chart(corr_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("property:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("correlation with nDCG:Q", scale=alt.Scale(domain=[-0.6, 0.6])),
            color=alt.condition(alt.datum["correlation with nDCG"] > 0,
                                alt.value(theme.GOOD), alt.value(theme.BAD)),
            tooltip=["property", alt.Tooltip("correlation with nDCG:Q", format=".3f")],
        )
        .properties(height=160)
    )
    st.altair_chart(cchart, use_container_width=True)

    strongest = corr.abs().idxmax()
    theme.note(
        f"The strongest single predictor of success is <b>{strongest}</b> "
        f"(r = {corr[strongest]:+.3f}). "
        + ("Lexical coverage dominating confirms the diagnosis in our problem statement: "
           "when the examiner's answer simply does not reuse the claim's vocabulary, the "
           "lexical arm contributes nothing and the whole burden falls on the semantic "
           "components."
           if "coverage" in strongest else
           "This tells us which direction further work should take.")
    )

    theme.section("What we would do next")
    c1, c2, c3 = st.columns(3)
    with c1:
        theme.card("Raise stage-1 recall",
                   "A patent-domain bi-encoder (PatentSBERTa, PaECTER) in place of "
                   "general MiniLM, a third retrieval arm, or query expansion from the "
                   "claim's own description section.")
    with c2:
        theme.card("Stronger reranker",
                   "A larger cross-encoder backbone, a listwise loss instead of "
                   "pointwise, and more than two epochs &mdash; CPU training was the "
                   "binding constraint here, not data.")
    with c3:
        theme.card("Acknowledge the label ceiling",
                   "Many genuinely relevant patents are never cited by an examiner, so "
                   "some 'errors' are unlabelled correct answers. Absolute scores on any "
                   "citation-derived benchmark are therefore pessimistic.")
