"""
IR Concepts page -- the teaching surface.

Every formula on this page has a calculator next to it that runs on the real
index, so a question like "what is IDF?" can be answered by typing a word and
watching the number appear, rather than by pointing at a slide.
"""

from __future__ import annotations

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.metrics import (average_precision, dcg, ndcg_at_k, precision_at_k,
                         recall_at_k, reciprocal_rank)
from src.text_utils import PATENT_STOPWORDS, analyze, analyze_steps, stem
from ui import theme
from ui.state import get_collection, get_engine

theme.inject()

col = get_collection()
engine = get_engine()

theme.page_header(
    "The IR concepts behind the project",
    "Each idea with its formula, and a calculator running on the real 11,849-document "
    "index underneath it.",
)

tabs = st.tabs([
    "1 · Preprocessing",
    "2 · TF-IDF & the vector space",
    "3 · BM25",
    "4 · Dense retrieval",
    "5 · Rank fusion",
    "6 · Reranking",
    "7 · Metrics",
])

# =========================================================================== #
with tabs[0]:
    theme.section("From raw text to index terms")
    st.markdown(
        "<p style='color:#5B6478;font-size:.9rem;max-width:82ch;line-height:1.7'>"
        "Raw text cannot be matched directly. <i>Comprising</i>, <i>comprises</i> and "
        "<i>comprised</i> are one concept, and <i>the</i> carries no information. The "
        "same function processes queries and documents &mdash; if it did not, a query "
        "term could never match its own document term.</p>",
        unsafe_allow_html=True,
    )
    theme.stage_flow([
        {"n": "step 1", "h": "Tokenise", "d": "Lowercase, split on non-alphanumerics, "
         "but keep internal hyphens and dots so <code>poly-ethylene</code> and "
         "<code>co2</code> survive.", "color": theme.INK},
        {"n": "step 2", "h": "Remove stopwords", "d": "English function words, plus "
         "patent boilerplate: <i>wherein</i>, <i>comprising</i>, <i>said</i>, "
         "<i>thereof</i>.", "color": theme.LEXICAL},
        {"n": "step 3", "h": "Drop bare numerals", "d": "Reference numerals like "
         "<code>175</code> are noise; alphanumerics like <code>175c</code> are kept.",
         "color": theme.DENSE},
        {"n": "step 4", "h": "Stem", "d": "Porter stemmer: <i>amplification</i> "
         "&rarr; <code>amplif</code>, which now matches <i>amplifier</i>.",
         "color": theme.GOOD},
    ])

    sample = st.text_area(
        "Try it -- type or paste any claim",
        value="The optical amplification stage according to any one of the preceding "
              "claims, wherein the optical filter 10 comprises a multilayer "
              "interferential filter.",
        height=90,
    )
    steps = analyze_steps(sample)

    c1, c2, c3, c4 = st.columns(4)
    for column, (label, key) in zip(
        [c1, c2, c3, c4],
        [("1 · tokenised", "tokenized"), ("2 · stopwords gone", "stopwords_removed"),
         ("3 · numerals gone", "numerals_removed"), ("4 · stemmed (final terms)", "stemmed")],
    ):
        with column:
            tokens = steps[key]
            chips = " ".join(
                f"<code style='background:#F5F6FA;padding:.1rem .3rem;border-radius:4px;"
                f"font-size:.76rem;display:inline-block;margin:.1rem'>{t}</code>"
                for t in tokens[:40]
            )
            st.markdown(
                f"<div class='card card-tight'><h4>{label} "
                f"<span class='muted' style='font-weight:500'>({len(tokens)})</span></h4>"
                f"<div style='line-height:2'>{chips}</div></div>",
                unsafe_allow_html=True,
            )

    if steps["removed_as_stopword"]:
        removed = sorted(set(steps["removed_as_stopword"]))
        patent_only = [t for t in removed if t in PATENT_STOPWORDS]
        theme.note(
            f"Removed <b>{len(removed)}</b> stopwords, of which "
            f"<b>{len(patent_only)}</b> are patent-specific: "
            + ", ".join(f"<code>{t}</code>" for t in patent_only[:14])
            + ". These appear in nearly every claim, so they cannot help tell claims "
              "apart &mdash; but left in, they make unrelated patents look similar."
        )

# =========================================================================== #
with tabs[1]:
    theme.section("TF-IDF and the vector space model")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        theme.formula(
            "<span class='c'># term weight (SMART: ltc)</span>\n"
            "w(t,d) = (1 + log f(t,d))  <span class='o'>*</span>  log(N / df(t))\n"
            "           <span class='c'>^ term frequency</span>      <span class='c'>^ inverse doc frequency</span>\n\n"
            "<span class='c'># every document becomes a vector; normalise to unit length</span>\n"
            "d_hat = d / ||d||\n\n"
            "<span class='c'># similarity is the angle between query and document</span>\n"
            "<span class='k'>cosine</span>(q,d) = q_hat <span class='o'>.</span> d_hat"
        )
        st.markdown(
            "<p style='color:#5B6478;font-size:.9rem;line-height:1.7'>"
            "<b>TF</b> says a word repeated in a document matters more. <b>IDF</b> says a "
            "word found in every document matters less. <b>Cosine</b> compares direction "
            "rather than magnitude, so a long document is not favoured just for being "
            "long.</p>",
            unsafe_allow_html=True,
        )
    with c2:
        theme.card(
            "Where it falls short",
            "<code>1 + log(tf)</code> keeps growing without bound, and cosine normalises "
            "by <i>vector</i> length rather than <i>document</i> length. BM25 fixes both: "
            "a saturating term frequency, and length normalisation against the collection "
            "average with a tunable strength. That is why BM25 is the stronger baseline "
            "on collections with uneven document lengths &mdash; like patents.",
        )
        theme.kv_list([
            ("Documents (N)", f"{len(col.doc_ids):,}"),
            ("Vocabulary", f"{len(engine.bm25.vocab):,} terms"),
            ("Average document length", f"{engine.bm25.avgdl:.0f} terms"),
        ])

    theme.section("IDF calculator")
    st.markdown(
        "<p style='color:#5B6478;font-size:.88rem'>Type a word to see how rare it is in "
        "our collection, and therefore how much weight a match on it earns.</p>",
        unsafe_allow_html=True,
    )
    words = st.text_input("Words (space separated)",
                          value="filter amplifier polymer wherein tire semiconductor")
    rows = []
    for raw in words.split():
        stats = engine.bm25.term_stats(stem(raw.lower()))
        if stats:
            rows.append({
                "word": raw, "stem": stats["term"], "df (documents containing it)":
                stats["df"], "IDF": stats["idf"],
                "share of corpus": stats["df"] / stats["n_docs"],
            })
        else:
            rows.append({"word": raw, "stem": "-", "df (documents containing it)": 0,
                         "IDF": float("nan"), "share of corpus": 0.0})
    idf_df = pd.DataFrame(rows)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(
            idf_df.style.format({"IDF": "{:.3f}", "share of corpus": "{:.2%}"}),
            use_container_width=True, hide_index=True,
        )
    with c2:
        valid = idf_df.dropna(subset=["IDF"])
        if not valid.empty:
            chart = (
                alt.Chart(valid)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    y=alt.Y("word:N", sort="-x", title=None),
                    x=alt.X("IDF:Q", title="IDF (higher = rarer = more informative)"),
                    color=alt.Color("IDF:Q", scale=alt.Scale(scheme="oranges"),
                                    legend=None),
                    tooltip=["word", "stem", alt.Tooltip("IDF:Q", format=".3f")],
                )
                .properties(height=max(150, 36 * len(valid)))
            )
            st.altair_chart(chart, use_container_width=True)

# =========================================================================== #
with tabs[2]:
    theme.section("Okapi BM25")
    theme.formula(
        "                              f(t,d) <span class='o'>*</span> (k1 + 1)\n"
        "score(q,d) = <span class='k'>SUM</span>   IDF(t) <span class='o'>*</span>  ------------------------------------\n"
        "             t in q            f(t,d) + k1 <span class='o'>*</span> (1 - b + b <span class='o'>*</span> |d|/avgdl)\n\n"
        "                       N - df(t) + 0.5\n"
        "       IDF(t) = ln( 1 + ----------------- )\n"
        "                         df(t) + 0.5"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        theme.card("IDF(t)", "Rare terms carry more evidence. A match on "
                   "<i>interferential</i> says far more than a match on <i>filter</i>.")
    with c2:
        theme.card("Saturating tf", "The 20th occurrence of a word adds almost nothing. "
                   "<b>k1</b> sets how quickly the reward flattens &mdash; this is the "
                   "key difference from TF-IDF's unbounded log.")
    with c3:
        theme.card("Length normalisation", "Long documents match more words simply by "
                   "being long. <b>b</b> sets how hard to penalise that, measured "
                   "against the collection's average length.")

    theme.section("Watch term-frequency saturation")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        k1_demo = st.slider("k1", 0.1, 3.0, 1.2, 0.1, key="k1demo")
        b_demo = st.slider("b", 0.0, 1.0, 0.75, 0.05, key="bdemo")
        len_ratio = st.slider("document length / average", 0.2, 3.0, 1.0, 0.1)
        st.caption(
            f"With these settings, a document {len_ratio:.1f}x the average length has "
            f"its term frequencies divided by a factor of "
            f"{1 - b_demo + b_demo * len_ratio:.2f} before saturation applies."
        )
    with c2:
        tfs = np.arange(0, 21)
        norm = k1_demo * (1 - b_demo + b_demo * len_ratio)
        bm = tfs * (k1_demo + 1) / (tfs + norm)
        tfidf_curve = np.where(tfs > 0, 1 + np.log(np.maximum(tfs, 1)), 0)
        cdf = pd.DataFrame(
            [{"tf": int(t), "weight": float(v), "model": "BM25 (saturating)"}
             for t, v in zip(tfs, bm)]
            + [{"tf": int(t), "weight": float(v), "model": "TF-IDF (1 + log tf)"}
               for t, v in zip(tfs, tfidf_curve)]
        )
        chart = (
            alt.Chart(cdf)
            .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.5)
            .encode(
                x=alt.X("tf:Q", title="f(t,d) -- times the term appears"),
                y=alt.Y("weight:Q", title="term weight (before IDF)"),
                color=alt.Color("model:N", scale=alt.Scale(range=[theme.LEXICAL,
                                                                 theme.MUTED]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["model", "tf", alt.Tooltip("weight:Q", format=".3f")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    theme.note(
        "BM25's curve flattens out; TF-IDF's never does. On a long patent description "
        "that repeats its own jargon dozens of times, the unbounded curve lets one "
        "common word dominate the whole score &mdash; which is exactly the failure "
        "saturation prevents."
    )

    theme.section("Score a real query against a real document")
    qid = st.selectbox(
        "Query", col.sample_queries("test", 25),
        format_func=lambda q: col.query(q)["text"][:95] + "...", key="bm25_demo_q",
    )
    qtext = col.query(qid)["text"]
    hits = engine.bm25.search(qtext, 8)
    if hits:
        pick = st.selectbox(
            "Document", [d for d, _ in hits],
            format_func=lambda d: (("[examiner-cited] " if d in col.relevant(qid) else "")
                                   + col.doc(d)["text"][:85] + "..."),
            key="bm25_demo_d",
        )
        contribs = engine.bm25.explain(qtext, pick)
        theme.term_bars([
            {"term": c.term, "value": c.contribution,
             "detail": f"<span class='doc-id'>idf {c.idf:.2f} x tf {c.tf_in_doc} "
                       f"(df {c.df})</span>"}
            for c in contribs
        ])
        st.caption(f"These terms sum to {sum(c.contribution for c in contribs):.3f} "
                   f"of this document's BM25 score.")

# =========================================================================== #
with tabs[3]:
    theme.section("Dense retrieval with a bi-encoder")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        theme.formula(
            "<span class='c'># one vector per token, then average them</span>\n"
            "H          = <span class='k'>transformer</span>(text)        <span class='c'># (tokens, 384)</span>\n"
            "pooled     = <span class='k'>SUM</span>(H <span class='o'>*</span> mask) / <span class='k'>SUM</span>(mask)  <span class='c'># mean pooling</span>\n"
            "e           = pooled / ||pooled||       <span class='c'># unit length</span>\n\n"
            "<span class='c'># unit vectors =&gt; inner product IS cosine</span>\n"
            "score(q,d) = e_q <span class='o'>.</span> e_d\n\n"
            "<span class='c'># FAISS IndexFlatIP: exhaustive, so recall is exact</span>"
        )
        st.markdown(
            "<p style='color:#5B6478;font-size:.9rem;line-height:1.7'>"
            "The crucial property is that query and document are encoded "
            "<b>separately</b>. The document never sees the query, so all "
            f"{len(col.doc_ids):,} document vectors are computed once, offline, and a "
            "search is a single matrix multiply.</p>",
            unsafe_allow_html=True,
        )
    with c2:
        theme.card(
            "Why we did not use an approximate index",
            "FAISS offers IVF and HNSW indexes that trade exactness for speed. At "
            f"{len(col.doc_ids):,} documents an exhaustive <code>IndexFlatIP</code> "
            "already answers in milliseconds, so approximation would only add error. "
            "At millions of documents the choice would reverse &mdash; and that is the "
            "only change needed to scale this stage.",
        )
        if engine.dense is not None:
            theme.kv_list([
                ("Model", engine.dense.model_name.split("/")[-1]),
                ("Dimensions", str(engine.dense.embeddings.shape[1])),
                ("Vectors stored", f"{engine.dense.embeddings.shape[0]:,}"),
                ("Index", "FAISS IndexFlatIP (exact cosine)"),
                ("Max input", f"{engine.dense.max_length} tokens"),
            ])

    theme.section("Semantic vs lexical: see the difference")
    st.markdown(
        "<p style='color:#5B6478;font-size:.88rem;max-width:82ch'>Type two phrases that "
        "mean the same thing but share no words. BM25 scores them at zero; the "
        "bi-encoder does not.</p>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    text_a = c1.text_input("Phrase A", "a threaded fastener secures the two plates")
    text_b = c2.text_input("Phrase B", "a screw holds the panels together")

    if text_a and text_b and engine.dense is not None:
        emb = engine.dense.encode([text_a, text_b])
        cos = float(emb[0] @ emb[1])
        terms_a, terms_b = set(analyze(text_a)), set(analyze(text_b))
        shared = terms_a & terms_b
        jaccard = len(shared) / max(len(terms_a | terms_b), 1)
        theme.tiles([
            {"label": "Dense cosine similarity", "value": f"{cos:.3f}",
             "hint": "semantic, 0 to 1", "color": theme.DENSE},
            {"label": "Shared index terms", "value": f"{len(shared)}",
             "hint": ", ".join(sorted(shared)) if shared else "none at all",
             "color": theme.LEXICAL},
            {"label": "Jaccard overlap", "value": f"{jaccard:.3f}",
             "hint": "what a lexical matcher sees", "color": theme.MUTED},
        ])
        if cos > 0.4 and jaccard < 0.2:
            theme.note(
                f"Cosine similarity <b>{cos:.3f}</b> on a Jaccard overlap of only "
                f"<b>{jaccard:.3f}</b>. BM25 would score this pair near zero. This gap "
                f"is vocabulary mismatch, and closing it is what the dense arm is for."
            )

# =========================================================================== #
with tabs[4]:
    theme.section("Reciprocal Rank Fusion")
    c1, c2 = st.columns([1, 1])
    with c1:
        theme.formula(
            "                        1\n"
            "RRF(d) = <span class='k'>SUM</span>    ------------------\n"
            "         rankers   k + rank_i(d)\n\n"
            "<span class='c'># k = 60, from the original RRF paper</span>\n"
            "<span class='c'># only RANKS are used -- never raw scores</span>"
        )
        st.markdown(
            "<p style='color:#5B6478;font-size:.9rem;line-height:1.7'>"
            "That last line is the whole trick. BM25 returns unbounded scores like "
            "18.4; cosine returns 0.71. Any attempt to add them directly needs "
            "calibration that breaks whenever the score distribution shifts. RRF throws "
            "the scores away and keeps the ordering, so there is nothing left to "
            "calibrate.</p>",
            unsafe_allow_html=True,
        )
    with c2:
        k_demo = st.slider("RRF constant k", 1, 120, 60, key="rrfk")
        ranks = np.arange(1, 21)
        contrib = 1.0 / (k_demo + ranks)
        rdf = pd.DataFrame({"rank": ranks, "contribution": contrib})
        chart = (
            alt.Chart(rdf)
            .mark_bar(cornerRadiusEnd=3, color=theme.FUSION)
            .encode(
                x=alt.X("rank:O", title="rank in one ranker's list"),
                y=alt.Y("contribution:Q", title="1 / (k + rank)"),
                tooltip=["rank", alt.Tooltip("contribution:Q", format=".5f")],
            )
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"With k={k_demo}, rank 1 is worth "
            f"{(1/(k_demo+1))/(1/(k_demo+10)):.2f}x rank 10. A small k makes the top "
            f"rank dominate; a large k flattens the list."
        )

    theme.section("Worked example")
    st.markdown(
        "<p style='color:#5B6478;font-size:.88rem;max-width:82ch'>Document D is ranked "
        "8th by BM25 and 2nd by the dense retriever. Document E is 3rd and 3rd. Which "
        "should win?</p>",
        unsafe_allow_html=True,
    )
    demo = pd.DataFrame([
        {"document": "D", "BM25 rank": 8, "Dense rank": 2},
        {"document": "E", "BM25 rank": 3, "Dense rank": 3},
        {"document": "F", "BM25 rank": 1, "Dense rank": 40},
    ])
    demo["RRF score"] = 1 / (60 + demo["BM25 rank"]) + 1 / (60 + demo["Dense rank"])
    demo["fused rank"] = demo["RRF score"].rank(ascending=False).astype(int)
    st.dataframe(
        demo.style.format({"RRF score": "{:.5f}"})
        .background_gradient(subset=["RRF score"], cmap="Purples"),
        use_container_width=True, hide_index=True,
    )
    theme.note(
        "E wins: being consistently good in both lists beats being excellent in one and "
        "mediocre in the other. F, ranked first by BM25 but 40th by the dense arm, "
        "falls &mdash; which is the behaviour we want, because one ranker loving a "
        "document while the other ignores it is weak evidence."
    )

# =========================================================================== #
with tabs[5]:
    theme.section("Bi-encoder vs cross-encoder")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='card' style='border-top:2px solid {theme.DENSE}'>"
            f"<h4>Bi-encoder &mdash; stage 1</h4>"
            f"<div class='formula' style='margin:.4rem 0'>"
            f"query <span class='o'>--&gt;</span> [transformer] <span class='o'>--&gt;</span> vector  <span class='o'>\\</span>\n"
            f"                                  <span class='o'>&gt;</span> cosine\n"
            f"doc   <span class='o'>--&gt;</span> [transformer] <span class='o'>--&gt;</span> vector  <span class='o'>/</span></div>"
            f"<p>The two texts never meet. The document must be compressed into one "
            f"vector <i>before</i> the model knows what will be asked of it. Cheap, "
            f"because every document vector is precomputed once.</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='card' style='border-top:2px solid {theme.RERANK}'>"
            f"<h4>Cross-encoder &mdash; stage 2</h4>"
            f"<div class='formula' style='margin:.4rem 0'>"
            f"[CLS] query [SEP] doc [SEP]\n"
            f"        <span class='o'>|</span>\n"
            f"   [transformer]  <span class='c'># full self-attention</span>\n"
            f"        <span class='o'>|</span>\n"
            f"   relevance score</div>"
            f"<p>Both texts enter the same forward pass, so attention compares every "
            f"query word against every document word. Much more accurate &mdash; and "
            f"nothing can be precomputed, because the input depends on the query.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    theme.section("Why the cascade is not optional")
    c1, c2 = st.columns([1, 1])
    with c1:
        per_doc_ms = st.slider("Cross-encoder cost per document (ms)", 1, 60, 12)
        depth = st.select_slider("Candidates reranked", [10, 20, 50, 100, 200, 500],
                                 value=100)
        n_docs = len(col.doc_ids)
        full_s = per_doc_ms * n_docs / 1000
        casc_s = per_doc_ms * depth / 1000
        theme.tiles([
            {"label": "Rerank everything", "value": f"{full_s:.0f} s",
             "hint": f"all {n_docs:,} documents, per query", "color": theme.BAD},
            {"label": "Rerank the shortlist", "value": f"{casc_s*1000:.0f} ms",
             "hint": f"top {depth} only", "color": theme.GOOD},
            {"label": "Speed-up", "value": f"{full_s/max(casc_s,1e-9):.0f}x",
             "color": theme.PRIMARY},
        ])
    with c2:
        theme.card(
            "The bargain",
            f"Stage 1 costs a few milliseconds and keeps the relevant document inside "
            f"its top {depth} most of the time. Stage 2 then pays the transformer price "
            f"on {depth} documents instead of {n_docs:,}. The accuracy is nearly that of "
            f"reranking everything, at a tiny fraction of the cost &mdash; and the "
            f"Evaluation page measures both halves of that claim.",
        )
        theme.card(
            "The price of the bargain",
            "Recall@k becomes a hard ceiling. Anything stage 1 misses is unrecoverable, "
            "no matter how good stage 2 is. That is why the first stage is a hybrid and "
            "why Recall@100 is reported as prominently as nDCG@10.",
        )
        theme.card(
            "Why we also built a cheap reranker",
            "The Learning-to-Rank model is the control. It sees the same retrieval "
            "signals the cross-encoder does &mdash; scores, ranks, term overlap &mdash; "
            "but not the text, and it trains in under a second. If it matched the "
            "transformer, the transformer would not be worth its cost. Comparing the "
            "two is what turns \"the cross-encoder helps\" into a measured claim rather "
            "than an assumption.",
        )

    theme.section("Hard negatives: why training data choice decides everything")
    theme.formula(
        "<span class='c'># random negatives -- the task is trivial, nothing is learned</span>\n"
        "query: optical amplifier claim\n"
        "neg:   <span class='o'>\"a peanut planter comprising a mounting frame\"</span>   <span class='c'># obviously not it</span>\n\n"
        "<span class='c'># hard negatives -- the decisions the reranker actually faces</span>\n"
        "query: optical amplifier claim\n"
        "neg:   <span class='o'>\"an optical fiber coupler branches the output light\"</span>\n"
        "       <span class='c'># same field, high BM25 score, but NOT cited by the examiner</span>"
    )
    st.markdown(
        "<p style='color:#5B6478;font-size:.9rem;max-width:82ch;line-height:1.7'>"
        "We mine negatives by running the real first stage on each training query and "
        "taking its top-ranked documents that the examiner did <b>not</b> cite, plus the "
        "paragraphs examiners inspected and explicitly rejected. The model is then "
        "trained on exactly the discriminations it will have to make at inference "
        "time.</p>",
        unsafe_allow_html=True,
    )

    theme.section("The trap we fell into, and how we found it")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(
            "<p style='color:#5B6478;font-size:.9rem;line-height:1.7'>"
            "Choosing the positives turned out to matter as much as choosing the "
            "negatives, and our first attempt got it wrong in a way that is worth "
            "showing.<br><br>"
            "Only about <b>40%</b> of examiner-cited paragraphs are inside the stage-1 "
            "pool at all. We initially trained on <i>all</i> cited paragraphs. But the "
            "other 60% have exactly one thing in common: <b>the retrievers scored them "
            "low</b> &mdash; that is the only reason they are missing from the pool. "
            "Meanwhile every negative was, by construction, a document the retrievers "
            "scored <b>high</b>.<br><br>"
            "So inside the training set, \"relevant\" silently came to mean \"looks "
            "dissimilar to the query\". The model learned that rule faithfully, and "
            "inverted the ranking.</p>",
            unsafe_allow_html=True,
        )
    with c2:
        theme.formula(
            "<span class='c'># dev nDCG@10, before the fix</span>\n"
            "hybrid, no reranking      <span class='k'>0.0950</span>\n"
            "+ cross-encoder, zero-shot  0.0509\n"
            "+ cross-encoder, fine-tuned <span class='o'>0.0104</span>  <span class='c'>&lt;- 9x WORSE</span>\n\n"
            "<span class='c'># the giveaway: reversing its ranking HELPED</span>\n"
            "fine-tuned ranking          0.0187\n"
            "fine-tuned <span class='k'>reversed</span>         0.0317"
        )
        theme.card(
            "The fix",
            "Train only on positives that stage 1 actually retrieved, mined at the same "
            "depth used at inference. Training and inference then see the same "
            "distribution, and the task becomes the real one: tell the cited paragraph "
            "apart from other highly-ranked paragraphs in the same technical field.",
        )
    theme.note(
        "Worth stressing: this bug produced <b>no error and no warning</b>. The loss "
        "curve fell smoothly from 1.23 to 0.36 and the model looked perfectly healthy. "
        "It was only caught by evaluating against a held-out split and asking why the "
        "number moved the wrong way &mdash; which is the entire argument for building "
        "the evaluation harness before trusting any model."
    )

# =========================================================================== #
with tabs[6]:
    theme.section("The evaluation metrics, computed live")
    st.markdown(
        "<p style='color:#5B6478;font-size:.88rem;max-width:82ch'>Toggle which positions "
        "hold a relevant document and watch every metric respond. This is the fastest "
        "way to see what each one actually rewards.</p>",
        unsafe_allow_html=True,
    )

    n_positions = 10
    cols = st.columns(n_positions)
    pattern = []
    defaults = [False, False, True, False, False, False, True, False, False, False]
    for i, column in enumerate(cols):
        with column:
            pattern.append(st.checkbox(f"#{i+1}", value=defaults[i], key=f"relpos{i}"))

    n_relevant = st.slider("Total relevant documents in the collection for this query",
                           1, 6, max(1, sum(pattern)))
    run = [f"d{i+1}" for i in range(n_positions)]
    relevant = {f"d{i+1}" for i, is_rel in enumerate(pattern) if is_rel}
    # Any relevant documents not in the top 10 are unretrieved but still counted.
    gold = relevant | {f"missing{j}" for j in range(max(0, n_relevant - len(relevant)))}

    m = {
        "Precision@10": precision_at_k(run, gold, 10),
        "Recall@10": recall_at_k(run, gold, 10),
        "MRR": reciprocal_rank(run, gold),
        "MAP (AP)": average_precision(run, gold),
        "nDCG@10": ndcg_at_k(run, gold, 10),
    }
    theme.tiles([
        {"label": k, "value": f"{v:.4f}",
         "color": [theme.LEXICAL, theme.GOOD, theme.PRIMARY, theme.FUSION,
                   theme.RERANK][i]}
        for i, (k, v) in enumerate(m.items())
    ])

    c1, c2 = st.columns([1.1, 1])
    with c1:
        rows = []
        running_hits = 0
        for i, is_rel in enumerate(pattern, start=1):
            running_hits += 1 if is_rel else 0
            rows.append({
                "rank": i,
                "relevant": "yes" if is_rel else "",
                "gain / log2(rank+1)": (1 / math.log2(i + 1)) if is_rel else 0.0,
                "precision@rank": running_hits / i,
            })
        rdf = pd.DataFrame(rows)
        st.dataframe(
            rdf.style.format({"gain / log2(rank+1)": "{:.4f}",
                              "precision@rank": "{:.4f}"})
            .background_gradient(subset=["gain / log2(rank+1)"], cmap="Purples"),
            use_container_width=True, hide_index=True, height=390,
        )
    with c2:
        disc = pd.DataFrame([
            {"rank": i, "discount": 1 / math.log2(i + 1),
             "relevant": pattern[i - 1]}
            for i in range(1, n_positions + 1)
        ])
        chart = (
            alt.Chart(disc)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("rank:O", title="rank"),
                y=alt.Y("discount:Q", title="1 / log2(rank + 1)"),
                color=alt.condition(alt.datum.relevant, alt.value(theme.GOOD),
                                    alt.value("#DCE1EC")),
                tooltip=["rank", alt.Tooltip("discount:Q", format=".4f")],
            )
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"DCG@10 = {dcg(run, gold, 10):.4f}. Dividing by the ideal DCG "
            f"(all {min(n_relevant, 10)} relevant documents at the very top) gives "
            f"nDCG@10 = {m['nDCG@10']:.4f}."
        )

    theme.section("What each metric is for")
    c1, c2 = st.columns(2)
    with c1:
        theme.card("Recall@k", "<i>Did we find it at all?</i> The metric for stage 1, "
                   "because the reranker can only reorder what stage 1 returned. A "
                   "document missing from the candidate pool is lost for good.")
        theme.card("MRR", "<i>How soon is the first hit?</i> Matches how an examiner "
                   "works: they read down the list and stop at the first piece of "
                   "blocking prior art, so later hits add nothing.")
        theme.card("Precision@k", "<i>How much of what we showed is useful?</i> Easy to "
                   "read, but it ignores the ordering inside the top k.")
    with c2:
        theme.card("MAP", "<i>Are ALL the relevant documents high?</i> Averages "
                   "precision at every relevant position, so finding a second and third "
                   "piece of prior art is rewarded &mdash; which matters for building a "
                   "complete invalidity case.")
        theme.card("nDCG@k", "<i>The headline number.</i> Discounts each hit by "
                   "<code>log2(rank+1)</code>, then divides by the best possible score. "
                   "The normalisation is what lets queries with different numbers of "
                   "relevant documents be averaged fairly.")
        theme.card("Why so many", "Each one hides something. Recall ignores order, MRR "
                   "ignores everything after the first hit, precision ignores order "
                   "within k. Reporting them together is what makes a ranking claim "
                   "credible.")
