"""
Ablations page.

Two kinds of experiment:

  * LIVE  -- anything built on the lexical index or on cached rankings is fast
             enough to recompute while the professor watches, so BM25's k1/b,
             the fusion weight and the preprocessing steps are real sliders.
  * PRECOMPUTED -- anything that needs the cross-encoder is far too slow for a
             slider, so scripts/run_ablations.py measures it once and this page
             reads the result.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import reciprocal_rank_fusion, weighted_score_fusion
from src.metrics import aggregate, evaluate_run
from ui import theme
from ui.state import get_ablations, get_collection, get_engine, missing_banner

theme.inject()

col = get_collection()
engine = get_engine()

theme.page_header(
    "Ablation studies",
    "Removing or changing one component at a time, so every design choice is "
    "backed by a measurement rather than an assumption.",
)

N_QUERIES = 120
# Ablations are TUNING experiments, so they run on the DEV split. The test
# split is touched only by the Evaluation page, and only once -- otherwise
# every slider on this page would be quietly fitting to the test set.
SPLIT = "dev"


@st.cache_data(show_spinner="Caching rankings for the live ablations ...")
def cached_rankings(n: int, depth: int = 300) -> dict:
    """BM25 and dense rankings for a fixed query sample.

    Computed once; after that every fusion experiment on this page is pure
    arithmetic on these lists, which is why the sliders feel instant.
    """
    qids = col.sample_queries(SPLIT, n, config.SETTINGS.eval.seed)
    eng = get_engine()
    out = {"qids": qids, "bm25": {}, "dense": {}}
    for qid in qids:
        text = col.query(qid)["text"]
        out["bm25"][qid] = eng.bm25.search(text, depth)
        out["dense"][qid] = eng.dense.search(text, depth) if eng.dense else []
    return out


tabs = st.tabs([
    "Fusion: how to combine the two retrievers",
    "BM25 parameters",
    "Preprocessing",
    "Rerank depth",
    "Long-document handling",
])

# =========================================================================== #
# 1. fusion
# =========================================================================== #
with tabs[0]:
    if not missing_banner(["dense"]):
        st.stop()
    theme.section("Reciprocal Rank Fusion vs weighted score fusion")
    st.markdown(
        "<p style='color:#5B6478;font-size:.9rem;max-width:82ch;line-height:1.7'>"
        "Both arms of the first stage return a ranking; we have to merge them. "
        "<b>RRF</b> uses only ranks, so it needs no score calibration. <b>Weighted "
        "fusion</b> min-max normalises the raw scores and takes a weighted sum, which "
        "gives more control but is sensitive to outliers. The slider is the weight on "
        "the lexical arm: 0 is pure dense, 1 is pure BM25.</p>",
        unsafe_allow_html=True,
    )

    cache = cached_rankings(N_QUERIES)
    qids = cache["qids"]

    c1, c2 = st.columns([1, 1])
    method = c1.radio("Fusion method", ["rrf", "weighted"], horizontal=True,
                      format_func=lambda m: "Reciprocal Rank Fusion" if m == "rrf"
                      else "Weighted score fusion")
    rrf_k = c2.slider("RRF constant k", 1, 120, config.SETTINGS.fusion.rrf_k,
                      disabled=(method != "rrf"),
                      help="Larger k flattens the influence of the very top ranks.")

    sweep = []
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        per = []
        for qid in qids:
            lists = [cache["bm25"][qid], cache["dense"][qid]]
            if method == "weighted":
                fused = weighted_score_fusion(lists, [alpha, 1 - alpha])
            else:
                fused = reciprocal_rank_fusion(lists, k=rrf_k, weights=[alpha, 1 - alpha])
            per.append(evaluate_run([d for d, _ in fused[:100]], col.relevant(qid)))
        agg = aggregate(per)
        sweep.append({
            "alpha": alpha, "nDCG@10": agg["ndcg@10"], "Recall@100": agg["recall@100"],
            "MRR": agg["mrr"], "Recall@10": agg["recall@10"],
        })
    sdf = pd.DataFrame(sweep)
    best = sdf.loc[sdf["Recall@100"].idxmax()]

    theme.tiles([
        {"label": "Best weight on BM25", "value": f"{best['alpha']:.1f}",
         "hint": "by Recall@100", "color": theme.FUSION},
        {"label": "Best Recall@100", "value": f"{best['Recall@100']:.3f}",
         "color": theme.GOOD},
        {"label": "Pure BM25", "value": f"{sdf.iloc[-1]['Recall@100']:.3f}",
         "hint": "alpha = 1.0", "color": theme.LEXICAL},
        {"label": "Pure dense", "value": f"{sdf.iloc[0]['Recall@100']:.3f}",
         "hint": "alpha = 0.0", "color": theme.DENSE},
    ])

    melted = sdf.melt("alpha", var_name="metric", value_name="value")
    chart = (
        alt.Chart(melted)
        .mark_line(point=alt.OverlayMarkDef(size=55), strokeWidth=2.5)
        .encode(
            x=alt.X("alpha:Q", title="weight on BM25  (0 = dense only, 1 = BM25 only)"),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
            color=alt.Color("metric:N", legend=alt.Legend(orient="top", title=None),
                            scale=alt.Scale(range=[theme.RERANK, theme.LEXICAL,
                                                   theme.DENSE, theme.GOOD])),
            tooltip=["metric", "alpha", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    if best["Recall@100"] > max(sdf.iloc[0]["Recall@100"], sdf.iloc[-1]["Recall@100"]):
        theme.note(
            f"The fused curve peaks <b>above both end points</b>: "
            f"{best['Recall@100']:.3f} at alpha = {best['alpha']:.1f}, against "
            f"{sdf.iloc[-1]['Recall@100']:.3f} for BM25 alone and "
            f"{sdf.iloc[0]['Recall@100']:.3f} for dense alone. That is direct evidence "
            f"the two retrievers find <i>different</i> relevant documents &mdash; if they "
            f"made the same mistakes, fusion could not beat the better of the two."
        )
    else:
        theme.warn(
            "On this query sample fusion does not beat the better single arm. Worth "
            "reporting as-is: it means the two retrievers' errors overlap more than "
            "expected here."
        )
    st.caption(f"Measured live on {len(qids)} held-out DEV queries -- never the test split, because this page is for tuning.")

# =========================================================================== #
# 2. BM25 parameters
# =========================================================================== #
with tabs[1]:
    theme.section("What k1 and b actually do")
    st.markdown(
        "<p style='color:#5B6478;font-size:.9rem;max-width:82ch;line-height:1.7'>"
        "BM25's two parameters are usually left at their defaults. Because our "
        "implementation applies the weighting at query time rather than baking it into "
        "the index, we can sweep them live on the real collection.</p>",
        unsafe_allow_html=True,
    )
    theme.formula(
        "                      f(t,d) <span class='o'>*</span> (<span class='k'>k1</span> + 1)\n"
        "score(t,d) = IDF(t) <span class='o'>*</span> ------------------------------------\n"
        "                      f(t,d) + <span class='k'>k1</span> <span class='o'>*</span> (1 - <span class='k'>b</span> + <span class='k'>b</span> <span class='o'>*</span> |d|/avgdl)\n\n"
        "<span class='k'>k1</span>  term-frequency saturation.  k1 -&gt; 0 makes BM25 almost binary\n"
        "    (does the word occur at all); large k1 lets repetition keep paying.\n"
        "<span class='k'>b</span>   length normalisation.  b = 0 ignores document length;\n"
        "    b = 1 fully divides by the length ratio."
    )

    qids = col.sample_queries(SPLIT, N_QUERIES, config.SETTINGS.eval.seed)
    c1, c2 = st.columns(2)
    with c1:
        k1_live = st.slider("k1", 0.0, 3.0, config.SETTINGS.bm25.k1, 0.1)
    with c2:
        b_live = st.slider("b", 0.0, 1.0, config.SETTINGS.bm25.b, 0.05)

    per = [
        evaluate_run(
            [d for d, _ in engine.bm25.search(col.query(q)["text"], 100,
                                              k1=k1_live, b=b_live)],
            col.relevant(q),
        )
        for q in qids
    ]
    live = aggregate(per)
    default = aggregate([
        evaluate_run([d for d, _ in engine.bm25.search(col.query(q)["text"], 100)],
                     col.relevant(q))
        for q in qids
    ])

    theme.tiles([
        {"label": "nDCG@10", "value": f"{live['ndcg@10']:.4f}",
         "delta": f"{live['ndcg@10'] - default['ndcg@10']:+.4f} vs default",
         "color": theme.LEXICAL},
        {"label": "Recall@100", "value": f"{live['recall@100']:.4f}",
         "delta": f"{live['recall@100'] - default['recall@100']:+.4f}", "color": theme.GOOD},
        {"label": "MRR", "value": f"{live['mrr']:.4f}",
         "delta": f"{live['mrr'] - default['mrr']:+.4f}", "color": theme.PRIMARY},
        {"label": "Default", "value": f"k1={config.SETTINGS.bm25.k1}, "
                                      f"b={config.SETTINGS.bm25.b}",
         "hint": f"nDCG@10 {default['ndcg@10']:.4f}", "color": theme.MUTED},
    ])

    if st.button("Sweep the full k1 x b grid", help="35 configurations on "
                 f"{len(qids)} queries"):
        grid = []
        prog = st.progress(0.0)
        combos = [(k1, b) for k1 in [0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0]
                  for b in [0.0, 0.25, 0.5, 0.75, 1.0]]
        for i, (k1, b) in enumerate(combos):
            scores = aggregate([
                evaluate_run(
                    [d for d, _ in engine.bm25.search(col.query(q)["text"], 100,
                                                      k1=k1, b=b)],
                    col.relevant(q))
                for q in qids
            ])
            grid.append({"k1": k1, "b": b, "nDCG@10": scores["ndcg@10"],
                         "Recall@100": scores["recall@100"]})
            prog.progress((i + 1) / len(combos))
        prog.empty()
        gdf = pd.DataFrame(grid)
        st.session_state["bm25_grid"] = gdf

    if "bm25_grid" in st.session_state:
        gdf = st.session_state["bm25_grid"]
        heat = (
            alt.Chart(gdf)
            .mark_rect()
            .encode(
                x=alt.X("b:O", title="b (length normalisation)"),
                y=alt.Y("k1:O", title="k1 (tf saturation)", sort="descending"),
                color=alt.Color("nDCG@10:Q", scale=alt.Scale(scheme="viridis"),
                                legend=alt.Legend(title="nDCG@10")),
                tooltip=["k1", "b", alt.Tooltip("nDCG@10:Q", format=".4f"),
                         alt.Tooltip("Recall@100:Q", format=".4f")],
            )
            .properties(height=300)
        )
        labels = heat.mark_text(fontSize=10, fontWeight=600).encode(
            text=alt.Text("nDCG@10:Q", format=".3f"),
            color=alt.value("white"),
        )
        st.altair_chart(heat + labels, use_container_width=True)
        best = gdf.loc[gdf["nDCG@10"].idxmax()]
        theme.note(
            f"Best cell: <b>k1 = {best['k1']}, b = {best['b']}</b> "
            f"(nDCG@10 {best['nDCG@10']:.4f}). The surface is fairly flat, which is the "
            f"usual finding &mdash; BM25's defaults are robust, and this is why we did "
            f"not spend the project's effort tuning them."
        )

# =========================================================================== #
# 3. preprocessing
# =========================================================================== #
with tabs[2]:
    theme.section("Is the preprocessing pipeline earning its place?")
    st.markdown(
        "<p style='color:#5B6478;font-size:.9rem;max-width:82ch;line-height:1.7'>"
        "We remove stopwords (including <b>patent-specific</b> boilerplate like "
        "<i>wherein</i>, <i>comprising</i>, <i>said</i>) and apply Porter stemming. "
        "Each run below rebuilds the BM25 index from scratch with one step switched "
        "off, so the comparison is real rather than asserted.</p>",
        unsafe_allow_html=True,
    )

    if st.button("Run the preprocessing ablation",
                 help="Rebuilds four BM25 indexes. Takes about a minute."):
        import src.text_utils as tu

        # This ablation temporarily rewires the shared text pipeline, so BOTH
        # globals are captured up front and restored in `finally`. If an error
        # left stemming switched off, every retriever in the running app would
        # silently start using unstemmed terms.
        original_stopwords = tu.STOPWORDS
        original_stem = tu.stem
        qids = col.sample_queries(SPLIT, N_QUERIES, config.SETTINGS.eval.seed)
        variants = [
            ("Full pipeline (ours)", tu.ENGLISH_STOPWORDS | tu.PATENT_STOPWORDS, True),
            ("No patent stopwords", tu.ENGLISH_STOPWORDS, True),
            ("No stopword removal", set(), True),
            ("No stemming", tu.ENGLISH_STOPWORDS | tu.PATENT_STOPWORDS, False),
        ]
        rows = []
        prog = st.progress(0.0)
        try:
            for i, (label, stops, do_stem) in enumerate(variants):
                tu.STOPWORDS = stops
                tu.stem = original_stem if do_stem else (lambda t: t)  # noqa: E731
                index = BM25Retriever().fit(col.doc_ids, col.doc_texts)
                agg = aggregate([
                    evaluate_run([d for d, _ in index.search(col.query(q)["text"], 100)],
                                 col.relevant(q))
                    for q in qids
                ])
                rows.append({
                    "variant": label, "nDCG@10": agg["ndcg@10"],
                    "Recall@100": agg["recall@100"], "MRR": agg["mrr"],
                    "vocabulary": len(index.vocab),
                })
                prog.progress((i + 1) / len(variants))
        finally:
            tu.STOPWORDS = original_stopwords
            tu.stem = original_stem
        prog.empty()
        st.session_state["prep_ablation"] = pd.DataFrame(rows)

    if "prep_ablation" in st.session_state:
        pdf = st.session_state["prep_ablation"]
        st.dataframe(
            pdf.style.format({"nDCG@10": "{:.4f}", "Recall@100": "{:.4f}",
                              "MRR": "{:.4f}", "vocabulary": "{:,}"})
            .background_gradient(subset=["nDCG@10", "Recall@100", "MRR"], cmap="Greens"),
            use_container_width=True, hide_index=True,
        )
        chart = (
            alt.Chart(pdf.melt("variant", value_vars=["nDCG@10", "MRR"],
                               var_name="metric", value_name="value"))
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                y=alt.Y("variant:N", sort=list(pdf["variant"]), title=None,
                        axis=alt.Axis(labelLimit=200)),
                x=alt.X("value:Q", title=None),
                color=alt.Color("metric:N", scale=alt.Scale(range=[theme.LEXICAL,
                                                                   theme.PRIMARY]),
                                legend=alt.Legend(orient="top", title=None)),
                row=alt.Row("metric:N", title=None,
                            header=alt.Header(labelFontSize=11, labelFontWeight=600)),
                tooltip=["variant", "metric", alt.Tooltip("value:Q", format=".4f")],
            )
            .properties(height=110)
        )
        st.altair_chart(chart, use_container_width=True)
        full = pdf.iloc[0]
        nopat = pdf[pdf["variant"] == "No patent stopwords"]
        if not nopat.empty:
            d = full["nDCG@10"] - nopat.iloc[0]["nDCG@10"]
            theme.note(
                f"Dropping the patent-specific stopword list changes nDCG@10 by "
                f"<b>{-d:+.4f}</b>. Legal boilerplate appears in nearly every claim, so "
                f"it has almost no discriminative power while still inflating similarity "
                f"between unrelated patents &mdash; a domain adaptation a general-purpose "
                f"search engine would not make."
            )

# =========================================================================== #
# 4. rerank depth (precomputed)
# =========================================================================== #
with tabs[3]:
    theme.section("How deep should the reranker go?")
    st.markdown(
        "<p style='color:#5B6478;font-size:.9rem;max-width:82ch;line-height:1.7'>"
        "This is the central trade-off of the cascade. Reranking more candidates can "
        "only help quality (there is more chance the relevant document is in the pool), "
        "but cost grows linearly: every extra candidate is another transformer pass. "
        "Too slow for a slider, so it is measured once by "
        "<code>scripts/run_ablations.py</code>.</p>",
        unsafe_allow_html=True,
    )

    ab = get_ablations()
    if not ab or "depth" not in ab:
        missing_banner(["ablations"])
    else:
        ddf = pd.DataFrame(ab["depth"])
        c1, c2 = st.columns([1.25, 1])
        with c1:
            base = alt.Chart(ddf).encode(
                x=alt.X("depth:Q", title="candidates reranked",
                        scale=alt.Scale(type="log"))
            )
            quality = base.mark_line(point=alt.OverlayMarkDef(size=70),
                                     strokeWidth=3, color=theme.RERANK).encode(
                y=alt.Y("ndcg@10:Q", title="nDCG@10", scale=alt.Scale(zero=False)),
                tooltip=["depth", alt.Tooltip("ndcg@10:Q", format=".4f")],
            )
            latency = base.mark_line(point=alt.OverlayMarkDef(size=70),
                                     strokeWidth=3, color=theme.MUTED,
                                     strokeDash=[6, 4]).encode(
                y=alt.Y("latency_ms:Q", title="latency (ms)"),
                tooltip=["depth", alt.Tooltip("latency_ms:Q", format=".0f")],
            )
            st.altair_chart(
                alt.layer(quality, latency).resolve_scale(y="independent")
                .properties(height=350),
                use_container_width=True,
            )
            st.caption("Solid = nDCG@10 (left axis). Dashed = latency (right axis).")
        with c2:
            show = ddf[["depth", "ndcg@10", "mrr", "recall@10", "recall@100",
                        "latency_ms"]].rename(columns={
                            "ndcg@10": "nDCG@10", "mrr": "MRR", "recall@10": "R@10",
                            # At depth < 100 the run returns only `depth` documents,
                            # so this is really the recall of the candidate pool.
                            "recall@100": "pool recall", "latency_ms": "ms/query"})
            st.dataframe(
                show.style.format({"nDCG@10": "{:.4f}", "MRR": "{:.4f}", "R@10": "{:.4f}",
                                   "pool recall": "{:.4f}", "ms/query": "{:.0f}"}),
                use_container_width=True, hide_index=True,
            )
            if len(ddf) > 2:
                first, last = ddf.iloc[0], ddf.iloc[-1]
                # Find the knee: the smallest depth within 5% of the best score.
                best_q = ddf["ndcg@10"].max()
                knee = ddf[ddf["ndcg@10"] >= best_q * 0.95].iloc[0]
                q_gain = (last["ndcg@10"] - first["ndcg@10"]) / max(first["ndcg@10"], 1e-9)
                t_gain = last["latency_ms"] / max(first["latency_ms"], 1e-9)
                extra_q = (last["ndcg@10"] - knee["ndcg@10"]) / max(knee["ndcg@10"], 1e-9)
                extra_t = last["latency_ms"] / max(knee["latency_ms"], 1e-9)
                theme.note(
                    f"Across the whole sweep, depth {int(first['depth'])} &rarr; "
                    f"{int(last['depth'])} changes nDCG@10 by <b>{q_gain*100:+.1f}%</b> "
                    f"for <b>{t_gain:.1f}x</b> the latency. Most of that is won early: "
                    f"depth <b>{int(knee['depth'])}</b> already reaches within 5% of the "
                    f"best score, and going on to {int(last['depth'])} buys only "
                    f"<b>{extra_q*100:+.1f}%</b> more for <b>{extra_t:.1f}x</b> the cost. "
                    f"That is why we operate at depth {int(knee['depth'])}-100 rather than "
                    f"reranking as deep as possible."
                )

# =========================================================================== #
# 5. long documents (precomputed)
# =========================================================================== #
with tabs[4]:
    theme.section("Truncation vs MaxP for long passages")
    st.markdown(
        "<p style='color:#5B6478;font-size:.9rem;max-width:82ch;line-height:1.7'>"
        "A cross-encoder has a fixed input window. When the claim plus the passage "
        "exceed it, something must give. <b>Truncate</b> simply cuts the passage. "
        "<b>MaxP</b> splits it into overlapping windows, scores each, and keeps the best "
        "&mdash; which matters here because prior art often hinges on one sentence that "
        "may sit past the cut.</p>",
        unsafe_allow_html=True,
    )
    ab = get_ablations()
    if not ab or "long_doc" not in ab:
        missing_banner(["ablations"])
    else:
        ldf = pd.DataFrame(ab["long_doc"])
        theme.tiles([
            {"label": r["strategy"], "value": f"{r['ndcg@10']:.4f}",
             "hint": f"{r['latency_ms']:.0f} ms/query · MRR {r['mrr']:.3f}",
             "color": theme.RERANK if r["strategy"] == "truncate" else theme.FUSION}
            for _, r in ldf.iterrows()
        ])
        st.dataframe(
            ldf.style.format({"ndcg@10": "{:.4f}", "mrr": "{:.4f}",
                              "recall@10": "{:.4f}", "latency_ms": "{:.0f}"}),
            use_container_width=True, hide_index=True,
        )
        if len(ldf) == 2:
            t = ldf[ldf["strategy"] == "truncate"].iloc[0]
            m = ldf[ldf["strategy"] == "max_passage"].iloc[0]
            diff = m["ndcg@10"] - t["ndcg@10"]
            cost = m["latency_ms"] / max(t["latency_ms"], 1e-9)
            verdict = ("worth the cost" if diff > 0.005
                       else "not worth the cost on this collection")
            theme.note(
                f"MaxP changes nDCG@10 by <b>{diff:+.4f}</b> for <b>{cost:.1f}x</b> the "
                f"latency &mdash; {verdict}. Our passages have a median of "
                f"{int(pd.Series([m['n_words'] for m in col.documents.values()]).median())} "
                f"words, so most fit the window already and truncation rarely loses the "
                f"decisive sentence. On full patent descriptions the conclusion would "
                f"very likely flip."
            )
