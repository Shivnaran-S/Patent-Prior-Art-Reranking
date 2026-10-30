"""
Evaluation page: every method, the same test queries, with a significance test.

Numbers are read from results/benchmark.json -- they were measured once by
scripts/run_benchmark.py, not recomputed live, so what the professor sees is
exactly what was recorded.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from ui import theme
from ui.state import get_collection, get_report, missing_banner

theme.inject()

col = get_collection()
report = get_report()

theme.page_header(
    "Evaluation",
    "Every configuration scored on the same held-out queries, with the same "
    "relevance judgements, and a significance test on the headline gain.",
)

if not report:
    missing_banner(["benchmark"])
    st.stop()

methods = report["methods"]
primary = report["primary_cutoff"]
order = [k for k in
         ["tfidf", "bm25", "dense", "hybrid", "hybrid_ltr", "bm25_ce",
          "hybrid_ce", "hybrid_ce_ft"] if k in methods]

st.caption(
    f"{report['n_queries']} test queries · {report['corpus_size']:,} documents · "
    f"measured {report['generated_at']}"
)

# --------------------------------------------------------------------------- #
# headline
# --------------------------------------------------------------------------- #
best_key = max(order, key=lambda k: methods[k]["metrics"][f"ndcg@{primary}"])
best, base = methods[best_key], methods.get("bm25")
b_ndcg = base["metrics"][f"ndcg@{primary}"] if base else 0.0
gain = best["metrics"][f"ndcg@{primary}"] - b_ndcg

theme.tiles([
    {"label": f"Best nDCG@{primary}", "value": f"{best['metrics'][f'ndcg@{primary}']:.3f}",
     "delta": f"{gain/b_ndcg*100:+.0f}% vs BM25" if b_ndcg else None,
     "hint": best["label"], "color": theme.RERANK},
    {"label": "Best MRR", "value": f"{best['metrics']['mrr']:.3f}",
     "hint": "first relevant hit", "color": theme.PRIMARY},
    {"label": "Best MAP", "value": f"{best['metrics']['map']:.3f}",
     "hint": "all relevant hits", "color": theme.FUSION},
    {"label": "BM25 baseline", "value": f"{b_ndcg:.3f}",
     "hint": f"nDCG@{primary}", "color": theme.LEXICAL},
    {"label": "Latency of best", "value": f"{best['mean_latency_ms']:.0f} ms",
     "hint": f"vs {base['mean_latency_ms']:.1f} ms for BM25" if base else None,
     "color": theme.DENSE},
])

tab_table, tab_charts, tab_sig, tab_cost, tab_perq = st.tabs(
    ["Results table", "Metric comparison", "Significance", "Accuracy vs cost",
     "Per-query detail"]
)

# --------------------------------------------------------------------------- #
with tab_table:
    theme.section("All methods, all metrics")
    rows = []
    for key in order:
        m = methods[key]
        met = m["metrics"]
        rows.append({
            "Method": m["label"],
            "Stage 1": config.METHOD_LABELS.get(m["stage1"], m["stage1"]),
            "Stage 2": "-" if m["stage2"] == "none"
                       else config.METHOD_LABELS.get(m["stage2"], m["stage2"]),
            f"nDCG@{primary}": met[f"ndcg@{primary}"],
            "MRR": met["mrr"],
            "MAP": met["map"],
            "R@10": met["recall@10"],
            "R@100": met["recall@100"],
            "P@1": met["precision@1"],
            "ms/query": m["mean_latency_ms"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style
        .background_gradient(subset=[f"nDCG@{primary}", "MRR", "MAP", "R@10", "R@100",
                                     "P@1"], cmap="Greens")
        .format({f"nDCG@{primary}": "{:.4f}", "MRR": "{:.4f}", "MAP": "{:.4f}",
                 "R@10": "{:.4f}", "R@100": "{:.4f}", "P@1": "{:.4f}",
                 "ms/query": "{:.1f}"}),
        use_container_width=True, hide_index=True, height=40 + 36 * len(df),
    )

    theme.section("How to read this table")
    c1, c2 = st.columns(2)
    with c1:
        theme.card(
            "Recall@100 is the stage-1 story",
            "It is identical for a method and that same method plus a reranker, because "
            "a reranker only reorders the candidate set. It is the <b>ceiling</b>: if a "
            "document is not in the top 100, no amount of reranking can rescue it. "
            "That is why the hybrid first stage matters.",
        )
    with c2:
        theme.card(
            f"nDCG@{primary} and MRR are the stage-2 story",
            "They measure how high the relevant document sits. Reranking leaves recall "
            "untouched but should move these up sharply &mdash; that difference, at "
            "constant recall, is the clean measurement of what the cross-encoder adds.",
        )

    with st.expander("Each method, in one sentence"):
        for key in order:
            m = methods[key]
            st.markdown(f"**{m['label']}** &mdash; {m['notes']}")
    if report.get("skipped"):
        with st.expander("Configurations that were skipped"):
            for key, reason in report["skipped"].items():
                st.markdown(f"**{key}** &mdash; {reason}")

# --------------------------------------------------------------------------- #
with tab_charts:
    theme.section("The headline metric")
    chart_rows = [
        {"method": methods[k]["label"], "key": k,
         "value": methods[k]["metrics"][f"ndcg@{primary}"]}
        for k in order
    ]
    cdf = pd.DataFrame(chart_rows)
    bars = (
        alt.Chart(cdf)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("method:N", sort=[r["method"] for r in chart_rows], title=None,
                    axis=alt.Axis(labelFontSize=12, labelLimit=240)),
            x=alt.X("value:Q", title=f"nDCG@{primary}"),
            color=alt.Color("key:N",
                            scale=alt.Scale(domain=order,
                                            range=[theme.METHOD_COLORS.get(k, theme.PRIMARY)
                                                   for k in order]),
                            legend=None),
            tooltip=[alt.Tooltip("method:N"), alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=38 * len(cdf) + 30)
    )
    labels = bars.mark_text(align="left", dx=5, fontSize=11, fontWeight=600).encode(
        text=alt.Text("value:Q", format=".3f"), color=alt.value(theme.INK)
    )
    st.altair_chart(bars + labels, use_container_width=True)

    theme.section("Recall at increasing depth")
    st.markdown(
        "<p style='color:#5B6478;font-size:.88rem;max-width:80ch'>"
        "Only the stage-1 methods are shown, because reranking cannot change recall. "
        "The gap between the hybrid curve and the others is the extra headroom fusion "
        "gives the reranker.</p>",
        unsafe_allow_html=True,
    )
    stage1_only = [k for k in ["tfidf", "bm25", "dense", "hybrid"] if k in methods]
    recall_rows = []
    for k in stage1_only:
        for cut in report["cutoffs"]:
            recall_rows.append({
                "method": methods[k]["label"], "key": k, "k": cut,
                "recall": methods[k]["metrics"][f"recall@{cut}"],
            })
    rdf = pd.DataFrame(recall_rows)
    line = (
        alt.Chart(rdf)
        .mark_line(point=alt.OverlayMarkDef(size=55), strokeWidth=2.5)
        .encode(
            x=alt.X("k:Q", scale=alt.Scale(type="log"), title="k (log scale)"),
            y=alt.Y("recall:Q", title="Recall@k"),
            color=alt.Color("key:N",
                            scale=alt.Scale(domain=stage1_only,
                                            range=[theme.METHOD_COLORS[k] for k in stage1_only]),
                            legend=alt.Legend(orient="top", title=None,
                                              labelExpr="datum.label")),
            tooltip=["method", "k", alt.Tooltip("recall:Q", format=".4f")],
        )
        .properties(height=330)
    )
    st.altair_chart(line, use_container_width=True)

    theme.section("Every metric, side by side")
    metric_keys = [f"ndcg@{primary}", "mrr", "map", "recall@10", "precision@1"]
    nice = {f"ndcg@{primary}": f"nDCG@{primary}", "mrr": "MRR", "map": "MAP",
            "recall@10": "Recall@10", "precision@1": "P@1"}
    grid_rows = [
        {"method": methods[k]["label"], "key": k, "metric": nice[mk],
         "value": methods[k]["metrics"][mk]}
        for k in order for mk in metric_keys
    ]
    gdf = pd.DataFrame(grid_rows)
    grid = (
        alt.Chart(gdf)
        .mark_bar()
        .encode(
            x=alt.X("key:N", title=None, sort=order, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("key:N",
                            scale=alt.Scale(domain=order,
                                            range=[theme.METHOD_COLORS.get(k, theme.PRIMARY)
                                                   for k in order]),
                            legend=alt.Legend(orient="bottom", title=None, columns=4,
                                              labelExpr="datum.label")),
            column=alt.Column("metric:N", title=None, sort=list(nice.values()),
                              header=alt.Header(labelFontSize=12, labelFontWeight=600)),
            tooltip=["method", "metric", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(width=118, height=230)
    )
    st.altair_chart(grid)

# --------------------------------------------------------------------------- #
with tab_sig:
    theme.section("Is the gain real, or is it noise?")
    st.markdown(
        "<p style='color:#5B6478;font-size:.9rem;max-width:82ch;line-height:1.7'>"
        "With a few hundred queries, a small average improvement can easily be luck. "
        "We use a <b>paired randomisation test</b>: take the per-query difference "
        "between the two systems, randomly flip each difference's sign ten thousand "
        "times, and count how often a difference at least as large as the observed one "
        "appears by chance. That proportion is the p-value. It assumes nothing about "
        "the shape of the distribution &mdash; which matters, because per-query IR "
        "metrics are nowhere near normally distributed.</p>",
        unsafe_allow_html=True,
    )
    theme.formula(
        "<span class='c'># observed effect</span>\n"
        "d_i      = metric(B, query i) <span class='o'>-</span> metric(A, query i)\n"
        "observed = <span class='k'>mean</span>(d_i)\n\n"
        "<span class='c'># 10,000 times: flip each sign at random</span>\n"
        "p = P( |<span class='k'>mean</span>(<span class='o'>+/-</span>d_i)| <span class='o'>&gt;=</span> |observed| )\n\n"
        "<span class='c'># p &lt; 0.05  =&gt;  unlikely to be chance</span>"
    )

    sig = report.get("significance", [])
    if sig:
        available = list(dict.fromkeys(r["metric"] for r in sig))
        nice = {f"ndcg@{primary}": f"nDCG@{primary}", "recall@100": "Recall@100",
                "mrr": "MRR", "map": "MAP"}
        chosen = st.radio(
            "Metric to test",
            available,
            index=available.index("recall@100") if "recall@100" in available else 0,
            horizontal=True,
            format_func=lambda m: nice.get(m, m.upper()),
            help="nDCG@10 is the headline but also the noisiest thing we measure. "
                 "Recall@100 is where a first-stage change shows up most cleanly.",
        )
        subset = [r for r in sig if r["metric"] == chosen]

        sdf = pd.DataFrame([
            {
                "Method": r["label"],
                f"Delta vs BM25": r["delta_vs_bm25"],
                "Relative": f"{r['relative_pct']:+.1f}%",
                "p-value": r["p_value"],
                "Significant (p<0.05)": "yes" if r["significant_at_05"] else "no",
            }
            for r in subset
        ])
        st.dataframe(
            sdf.style.format({"Delta vs BM25": "{:+.4f}", "p-value": "{:.4f}"})
            .background_gradient(subset=["Delta vs BM25"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True, height=40 + 36 * len(sdf),
        )

        winners = [r for r in subset if r["significant_at_05"] and r["delta_vs_bm25"] > 0]
        if winners:
            theme.note(
                f"<b>{len(winners)} configuration(s)</b> beat BM25 on "
                f"{nice.get(chosen, chosen)} with p &lt; 0.05. The strongest is "
                f"<b>{winners[0]['label']}</b> at {winners[0]['relative_pct']:+.1f}% "
                f"relative (p = {winners[0]['p_value']:.4f})."
            )
        else:
            theme.warn(
                f"No configuration clears p &lt; 0.05 against BM25 on "
                f"{nice.get(chosen, chosen)} with these {report['n_queries']} queries. "
                "We report that as measured rather than quietly dropping the test."
            )

        theme.section("What this means, across metrics")
        wins = {m: [r for r in sig
                    if r["metric"] == m and r["significant_at_05"] and r["delta_vs_bm25"] > 0]
                for m in available}
        c1, c2 = st.columns(2)
        with c1:
            theme.card(
                "Where the evidence is strong",
                "The <b>Recall@100</b> gain from hybrid fusion is large and holds up "
                "under the test. That is the result this project can state without "
                "hedging: combining a lexical and a semantic retriever finds "
                "substantially more of the prior art than either alone."
                if wins.get("recall@100") else
                "On this sample the recall gain did not clear the significance "
                "threshold either.",
            )
        with c2:
            theme.card(
                "Where it is only suggestive",
                f"The <b>nDCG@{primary}</b> differences point the right way but do not "
                f"reach p &lt; 0.05. With ~{report['n_queries']} queries and roughly one "
                "relevant document each, nDCG is zero for most queries and the "
                "per-query variance is large. More test queries, not a different model, "
                "is what would settle it &mdash; and saying so is more useful than "
                "quoting a percentage that the data does not support."
                if not wins.get(f"ndcg@{primary}") else
                f"The nDCG@{primary} gain also clears the threshold.",
            )

# --------------------------------------------------------------------------- #
with tab_cost:
    theme.section("What the accuracy costs")
    st.markdown(
        "<p style='color:#5B6478;font-size:.9rem;max-width:82ch'>"
        "This is the plot that justifies the cascade. Up and to the left is better: "
        "high quality, low latency. The cross-encoder buys accuracy with time &mdash; "
        "but only on 100 documents, not 11,849.</p>",
        unsafe_allow_html=True,
    )
    cost_rows = [
        {
            "method": methods[k]["label"], "key": k,
            "latency": max(methods[k]["mean_latency_ms"], 0.05),
            "quality": methods[k]["metrics"][f"ndcg@{primary}"],
            "stage1_ms": methods[k]["stage1_ms"], "stage2_ms": methods[k]["stage2_ms"],
        }
        for k in order
    ]
    costdf = pd.DataFrame(cost_rows)
    pts = (
        alt.Chart(costdf)
        .mark_circle(size=320, opacity=0.9)
        .encode(
            x=alt.X("latency:Q", scale=alt.Scale(type="log"),
                    title="mean latency per query, ms (log scale)"),
            y=alt.Y("quality:Q", title=f"nDCG@{primary}",
                    scale=alt.Scale(zero=False, padding=20)),
            color=alt.Color("key:N",
                            scale=alt.Scale(domain=order,
                                            range=[theme.METHOD_COLORS.get(k, theme.PRIMARY)
                                                   for k in order]),
                            legend=None),
            tooltip=["method", alt.Tooltip("latency:Q", format=".1f"),
                     alt.Tooltip("quality:Q", format=".4f")],
        )
        .properties(height=400)
    )
    text = pts.mark_text(align="left", dx=14, dy=-4, fontSize=11, fontWeight=600).encode(
        text="method:N", color=alt.value(theme.INK)
    )
    st.altair_chart(pts + text, use_container_width=True)

    theme.section("Where the time goes")
    split_rows = []
    for r in cost_rows:
        split_rows.append({"method": r["method"], "stage": "Stage 1 · retrieval",
                           "ms": r["stage1_ms"]})
        split_rows.append({"method": r["method"], "stage": "Stage 2 · reranking",
                           "ms": r["stage2_ms"]})
    sdf = pd.DataFrame(split_rows)
    stacked = (
        alt.Chart(sdf)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("method:N", sort=[r["method"] for r in cost_rows], title=None,
                    axis=alt.Axis(labelLimit=240)),
            x=alt.X("ms:Q", title="milliseconds per query"),
            color=alt.Color("stage:N",
                            scale=alt.Scale(range=[theme.FUSION, theme.RERANK]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["method", "stage", alt.Tooltip("ms:Q", format=".1f")],
        )
        .properties(height=38 * len(cost_rows) + 30)
    )
    st.altair_chart(stacked, use_container_width=True)

    ce = methods.get("hybrid_ce") or methods.get("bm25_ce")
    if ce:
        per_doc = ce["stage2_ms"] / max(ce["first_stage_k"], 1)
        full = per_doc * report["corpus_size"] / 1000
        theme.note(
            f"The cross-encoder takes about <b>{per_doc:.1f} ms per document</b> on this "
            f"machine. Scoring all {report['corpus_size']:,} documents for a single query "
            f"would therefore take roughly <b>{full:.0f} seconds</b>. Reranking only the "
            f"top {ce['first_stage_k']} takes <b>{ce['stage2_ms']:.0f} ms</b> &mdash; about "
            f"<b>{full*1000/max(ce['stage2_ms'],1):.0f}x faster</b>. That ratio is the "
            f"entire argument for retrieve-then-rerank."
        )

# --------------------------------------------------------------------------- #
with tab_perq:
    theme.section("Per-query comparison")
    st.markdown(
        "<p style='color:#5B6478;font-size:.88rem;max-width:82ch'>"
        "Averages hide the distribution. Pick two systems: each row is one query, and "
        "the sign of the difference shows whether reranking helped or hurt it.</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    a_key = c1.selectbox("System A (baseline)", order,
                         index=order.index("bm25") if "bm25" in order else 0,
                         format_func=lambda k: methods[k]["label"])
    b_key = c2.selectbox("System B", order, index=len(order) - 1,
                         format_func=lambda k: methods[k]["label"])
    metric = c3.selectbox(f"Metric", [f"ndcg@{primary}", "mrr", "map", "recall@10"],
                          format_func=lambda m: m.upper())

    a_pq = methods[a_key]["per_query"]
    b_pq = methods[b_key]["per_query"]
    shared = [q for q in a_pq if q in b_pq]
    diff_rows = [
        {
            "query_id": q,
            methods[a_key]["label"]: a_pq[q][metric],
            methods[b_key]["label"]: b_pq[q][metric],
            "difference": b_pq[q][metric] - a_pq[q][metric],
            "claim": col.query(q)["text"][:90] if q in col.queries else "",
        }
        for q in shared
    ]
    ddf = pd.DataFrame(diff_rows)
    better = int((ddf["difference"] > 1e-9).sum())
    worse = int((ddf["difference"] < -1e-9).sum())
    same = len(ddf) - better - worse

    theme.tiles([
        {"label": "B better", "value": str(better), "color": theme.GOOD},
        {"label": "B worse", "value": str(worse), "color": theme.BAD},
        {"label": "Tied", "value": str(same), "color": theme.MUTED},
        {"label": "Mean difference", "value": f"{ddf['difference'].mean():+.4f}",
         "color": theme.PRIMARY},
    ])

    hist = (
        alt.Chart(ddf)
        .mark_bar()
        .encode(
            x=alt.X("difference:Q", bin=alt.Bin(maxbins=35),
                    title=f"{metric.upper()} difference  (B - A)"),
            y=alt.Y("count():Q", title="queries"),
            color=alt.condition(alt.datum.difference > 0, alt.value(theme.GOOD),
                                alt.value(theme.BAD)),
            tooltip=["count()"],
        )
        .properties(height=260)
    )
    st.altair_chart(hist, use_container_width=True)

    st.dataframe(
        ddf.sort_values("difference", ascending=False)
        .style.format({methods[a_key]["label"]: "{:.4f}",
                       methods[b_key]["label"]: "{:.4f}", "difference": "{:+.4f}"})
        .background_gradient(subset=["difference"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True, height=400,
    )
