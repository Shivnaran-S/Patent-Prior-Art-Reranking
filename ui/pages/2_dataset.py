"""Dataset page: the test collection, its provenance, and how it is split."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.text_utils import analyze
from ui import theme
from ui.state import get_collection

theme.inject()

col = get_collection()
stats = col.stats

theme.page_header(
    "The test collection",
    "An IR experiment is only as trustworthy as its ground truth. Ours comes from "
    "real European Patent Office examiners, not a heuristic.",
)

theme.tiles(
    [
        {"label": "Documents", "value": f"{len(col.doc_ids):,}",
         "hint": "prior-art passages", "color": theme.DENSE},
        {"label": "Queries", "value": f"{len(col.queries):,}",
         "hint": "patent claims", "color": theme.PRIMARY},
        {"label": "Relevance labels", "value": f"{stats.get('n_qrels', 0):,}",
         "hint": "examiner citations", "color": theme.GOOD},
        {"label": "Avg claim length", "value": f"{stats.get('mean_query_words', 0):.0f}",
         "hint": "words per query", "color": theme.LEXICAL},
        {"label": "Avg passage length", "value": f"{stats.get('mean_doc_words', 0):.0f}",
         "hint": "words per document", "color": theme.FUSION},
    ]
)

tab_source, tab_split, tab_browse, tab_shape = st.tabs(
    ["Where the labels come from", "How we split it", "Browse the collection",
     "Shape of the data"]
)

# --------------------------------------------------------------------------- #
with tab_source:
    theme.section("PatentMatch: relevance from examiner search reports")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(
            "<p style='color:#5B6478;font-size:.9rem;line-height:1.7'>"
            "When a European patent examiner rejects a claim for lack of novelty, the "
            "search report must cite the <b>exact paragraph</b> of an earlier document "
            "that already disclosed the invention. PatentMatch collected those "
            "citations. That gives us a test collection in the classic three-part "
            "form, with labels that are expert human judgements:"
            "</p>",
            unsafe_allow_html=True,
        )
        theme.formula(
            "<span class='k'>query</span>  = a claim from a new patent application\n"
            "<span class='k'>doc</span>    = a paragraph from an earlier patent document\n"
            "<span class='k'>label</span>  = <span class='o'>1</span> if an examiner cited that paragraph\n"
            "              against that claim  (novelty-destroying)\n"
            "         <span class='o'>0</span> if the examiner inspected it and\n"
            "              decided it was <span class='o'>not</span> novelty-destroying"
        )
        st.markdown(
            "<p style='color:#5B6478;font-size:.9rem;line-height:1.7'>"
            "The label-0 pairs matter as much as the label-1 pairs: they are documents "
            "an expert looked at and <i>rejected</i>, which makes them gold-standard "
            "<b>hard negatives</b> for training the reranker."
            "</p>",
            unsafe_allow_html=True,
        )
    with c2:
        theme.kv_list(
            [
                ("Source", "PatentMatch (EPO search reports)"),
                ("Judgements", f"{stats.get('n_qrels', 0):,} examiner citations"),
                ("Examiner hard negatives", f"{stats.get('n_hard_negatives', 0):,}"),
                ("Relevant docs per query", f"{stats.get('mean_relevant_per_query', 0):.2f}"),
                ("Claims with real EPO numbers",
                 f"{stats.get('queries_with_epo_metadata', 0)}"),
            ]
        )
        theme.card(
            "Why not HUPD?",
            "Our proposal listed HUPD as the corpus. HUPD has 4.5M applications with "
            "full text but <b>no citation field</b>, and the PatentsView citation API "
            "is no longer openly reachable, so relevance labels cannot be derived from "
            "it. PatentMatch gives genuine examiner judgements at passage level, which "
            "is what an evaluation actually needs, so we built the benchmark on it "
            "instead and kept the pipeline corpus-agnostic.",
        )

    theme.section("A real labelled example")
    showcase = col.showcase_queries(40)
    if showcase:
        pick = st.selectbox(
            "Claim (these are the ones that kept their real EPO publication number)",
            showcase,
            format_func=lambda q: f"{col.query(q).get('patent_id', q)} - "
                                  f"{col.query(q)['text'][:70]}...",
        )
        qmeta = col.query(pick)
        c1, c2 = st.columns(2)
        with c1:
            link = qmeta.get("epo_link")
            head = f"Query claim · {qmeta.get('patent_id', '')}"
            if link:
                head += f" · <a href='{link}' target='_blank'>EPO document</a>"
            st.markdown(
                f"<div class='card'><h4>{head}</h4>"
                f"<p style='color:#2A3147'>{qmeta['text']}</p>"
                f"<div style='margin-top:.5rem'><span class='doc-id'>"
                f"filed {qmeta.get('filing_date', 'n/a')} · "
                f"{qmeta['n_words']} words · split: {qmeta['split']}</span></div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            for doc_id in sorted(col.relevant(pick)):
                dmeta = col.doc(doc_id)
                dlink = dmeta.get("epo_link")
                dhead = f"Cited prior art · {dmeta.get('patent_id', doc_id)}"
                if dlink:
                    dhead += f" · <a href='{dlink}' target='_blank'>EPO document</a>"
                st.markdown(
                    f"<div class='card' style='border-left:3px solid {theme.GOOD}'>"
                    f"<h4>{dhead}</h4>"
                    f"<p style='color:#2A3147'>{dmeta['text'][:900]}</p>"
                    f"<div style='margin-top:.5rem'><span class='doc-id'>"
                    f"{dmeta['n_words']} words</span></div></div>",
                    unsafe_allow_html=True,
                )

        q_terms = set(analyze(qmeta["text"]))
        d_terms = set()
        for doc_id in col.relevant(pick):
            d_terms |= set(analyze(col.doc(doc_id)["text"]))
        shared = q_terms & d_terms
        coverage = len(shared) / max(len(q_terms), 1)
        theme.note(
            f"This claim shares only <b>{len(shared)} of its {len(q_terms)} index terms "
            f"({coverage*100:.0f}%)</b> with the paragraph an examiner cited against it. "
            "That is vocabulary mismatch in one number, and it is why a purely lexical "
            "retriever struggles on this task."
        )

# --------------------------------------------------------------------------- #
with tab_split:
    theme.section("A group split, not a random split")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            "<p style='color:#5B6478;font-size:.9rem;line-height:1.7'>"
            "A random train/test split would <b>leak</b>. Several claims of the same "
            "application are near-duplicates of each other (<i>\"The device of claim 1, "
            "wherein...\"</i>) and they cite the <b>same</b> prior-art paragraphs. If one "
            "lands in training and its twin in test, the test score is inflated and the "
            "whole evaluation is worthless."
            "</p>",
            unsafe_allow_html=True,
        )
        theme.formula(
            "<span class='c'># build a bipartite graph</span>\n"
            "for each (query, relevant doc) pair:\n"
            "    add edge  query <span class='o'>---</span> doc\n\n"
            "<span class='c'># a connected component is a cluster of queries that</span>\n"
            "<span class='c'># share prior art, directly or transitively</span>\n"
            "for each <span class='k'>connected component</span>:\n"
            "    send the <span class='o'>whole</span> component to ONE split\n\n"
            "<span class='c'># result: no test query shares a relevant document</span>\n"
            "<span class='c'># with any training query</span>"
        )
        st.markdown(
            "<p style='color:#5B6478;font-size:.9rem;line-height:1.7'>"
            "The split is seeded, so every run of every experiment sees exactly the same "
            "test queries and the numbers are directly comparable."
            "</p>",
            unsafe_allow_html=True,
        )
    with c2:
        split_counts = stats.get("queries_per_split", {})
        df = pd.DataFrame(
            [{"split": k, "queries": v} for k, v in split_counts.items()]
        )
        if not df.empty:
            chart = (
                alt.Chart(df)
                .mark_bar(cornerRadiusEnd=4, size=38)
                .encode(
                    y=alt.Y("split:N", sort=["train", "dev", "test"], title=None),
                    x=alt.X("queries:Q", title="queries"),
                    color=alt.Color(
                        "split:N",
                        scale=alt.Scale(
                            domain=["train", "dev", "test"],
                            range=[theme.PRIMARY, theme.DENSE, theme.RERANK],
                        ),
                        legend=None,
                    ),
                    tooltip=["split", "queries"],
                )
                .properties(height=170)
            )
            st.altair_chart(chart, use_container_width=True)
        theme.kv_list(
            [
                ("Split method", "connected components"),
                ("Seed", str(stats.get("seed", 42))),
                ("Train", f"{split_counts.get('train', 0):,} queries"),
                ("Dev (model selection)", f"{split_counts.get('dev', 0):,} queries"),
                ("Test (reported)", f"{split_counts.get('test', 0):,} queries"),
            ]
        )
        theme.card(
            "What the splits are used for",
            "<b>Train</b> mines hard negatives and fits the rerankers. <b>Dev</b> is the "
            "only split we look at while choosing settings. <b>Test</b> is scored once "
            "and is what the Evaluation page reports.",
        )

# --------------------------------------------------------------------------- #
with tab_browse:
    theme.section("Browse")
    mode = st.radio("", ["Queries (patent claims)", "Documents (prior-art passages)"],
                    horizontal=True, label_visibility="collapsed")

    if mode.startswith("Queries"):
        c1, c2, c3 = st.columns([1, 1, 2])
        split = c1.selectbox("Split", ["all", "train", "dev", "test"])
        min_words = c2.slider("Minimum words", 0, 200, 0, step=10)
        search = c3.text_input("Contains text", placeholder="e.g. optical filter")

        rows = []
        for qid, meta in col.queries.items():
            if split != "all" and meta["split"] != split:
                continue
            if meta["n_words"] < min_words:
                continue
            if search and search.lower() not in meta["text"].lower():
                continue
            rows.append(
                {
                    "query_id": qid,
                    "patent": meta.get("patent_id") or "",
                    "split": meta["split"],
                    "words": meta["n_words"],
                    "relevant docs": meta["n_relevant"],
                    "claim": meta["text"][:160],
                }
            )
            if len(rows) >= 400:
                break
        st.caption(f"Showing {len(rows)} of {len(col.queries):,} queries")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=430,
                     hide_index=True)
    else:
        c1, c2 = st.columns([1, 3])
        min_words = c1.slider("Minimum words", 0, 400, 0, step=20)
        search = c2.text_input("Contains text", placeholder="e.g. rubber composition")
        rows = []
        for did, meta in col.documents.items():
            if meta["n_words"] < min_words:
                continue
            if search and search.lower() not in meta["text"].lower():
                continue
            rows.append(
                {
                    "doc_id": did,
                    "patent": meta.get("patent_id") or "",
                    "words": meta["n_words"],
                    "passage": meta["text"][:200],
                }
            )
            if len(rows) >= 400:
                break
        st.caption(f"Showing {len(rows)} of {len(col.doc_ids):,} documents")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=430,
                     hide_index=True)

# --------------------------------------------------------------------------- #
with tab_shape:
    theme.section("Length distributions")
    st.markdown(
        "<p style='color:#5B6478;font-size:.89rem;max-width:80ch'>"
        "Document length matters directly: BM25's <code>b</code> parameter penalises long "
        "documents, and the cross-encoder has a fixed input window, so any passage past "
        "it must be truncated or split. These are the distributions those decisions are "
        "based on.</p>",
        unsafe_allow_html=True,
    )

    q_df = pd.DataFrame({"words": [m["n_words"] for m in col.queries.values()],
                         "kind": "Query (claim)"})
    d_df = pd.DataFrame({"words": [m["n_words"] for m in col.documents.values()],
                         "kind": "Document (passage)"})
    both = pd.concat([q_df, d_df])

    chart = (
        alt.Chart(both[both["words"] <= 600])
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("words:Q", bin=alt.Bin(maxbins=55), title="words"),
            y=alt.Y("count():Q", title="count", stack=None),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(range=[theme.DENSE, theme.LEXICAL]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["kind", "count()"],
        )
        .properties(height=290)
    )
    st.altair_chart(chart, use_container_width=True)

    c1, c2 = st.columns(2)
    q_words = sorted(m["n_words"] for m in col.queries.values())
    d_words = sorted(m["n_words"] for m in col.documents.values())

    def pct(xs, p):
        return xs[int(len(xs) * p / 100)]

    with c1:
        theme.kv_list([
            ("Query median", f"{pct(q_words, 50)} words"),
            ("Query 90th percentile", f"{pct(q_words, 90)} words"),
            ("Query 99th percentile", f"{pct(q_words, 99)} words"),
            ("Longest query", f"{q_words[-1]} words"),
        ])
    with c2:
        # The cross-encoder window is 256 tokens shared between the claim and
        # the passage; patent prose runs about 1.4 tokens per word, so a
        # passage past ~150 words will be cut once a claim is prepended.
        window_words = int(config.SETTINGS.cross_encoder.max_length / 1.4) - 30
        over = sum(1 for w in d_words if w > window_words)
        theme.kv_list([
            ("Document median", f"{pct(d_words, 50)} words"),
            ("Document 90th percentile", f"{pct(d_words, 90)} words"),
            ("Document 99th percentile", f"{pct(d_words, 99)} words"),
            (f"Longer than ~{window_words} words",
             f"{over:,} docs ({over/len(d_words)*100:.0f}%)"),
        ])

    theme.note(
        "About <b>" f"{over/len(d_words)*100:.0f}%" "</b> of passages are long enough "
        f"that the {config.SETTINGS.cross_encoder.max_length}-token cross-encoder has to cut "
        "them. That is exactly why the "
        "<b>MaxP</b> strategy (split into overlapping windows, score each, keep the best) "
        "is implemented as an option &mdash; prior art often hinges on one sentence that "
        "may sit past the truncation point. The Ablations page measures whether it helps."
    )
