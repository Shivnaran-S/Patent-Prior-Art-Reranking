"""
Shared rendering of search results.

The result list appears on three different pages (Live Search, Ablations, Error
Analysis), so it is written once here.  Each row shows:

  rank  |  relevance badge  |  rank movement from stage 1  |  score
  the document text with query terms highlighted
  how much of the query it covers lexically
"""

from __future__ import annotations

import streamlit as st

from src.explain import explain_movement, highlight_overlap, overlap_stats
from . import theme


def result_row(
    doc_id: str,
    text: str,
    query: str,
    rank: int,
    score: float,
    is_relevant: bool,
    stage1_rank: int | None = None,
    delta: int | None = None,
    score_label: str = "score",
    patent_id: str | None = None,
    epo_link: str | None = None,
    idf_lookup=None,
    max_words: int = 110,
) -> None:
    badge_cls = "rank-badge relevant" if is_relevant else "rank-badge"
    row_cls = "result relevant" if is_relevant else "result"

    bits = [f"<div class='{badge_cls}'>{rank}</div>"]
    if is_relevant:
        bits.append(theme.pill("examiner-cited prior art", "rel"))
    else:
        bits.append(theme.pill("not cited", "irr"))

    if delta is not None or stage1_rank is not None:
        label, cls = explain_movement(delta)
        was = f" (was #{stage1_rank})" if stage1_rank else ""
        bits.append(f"<span class='move {cls}'>{label}{was}</span>")

    bits.append(f"<span class='doc-id'>{score_label} {score:+.3f}</span>")
    if patent_id:
        link = (f"<a href='{epo_link}' target='_blank' style='text-decoration:none'>"
                f"{patent_id}</a>") if epo_link else patent_id
        bits.append(f"<span class='doc-id'>· {link}</span>")
    bits.append(f"<span class='doc-id'>· {doc_id}</span>")

    stats = overlap_stats(query, text, idf_lookup)
    body = highlight_overlap(text, query, max_words=max_words)
    coverage = (
        f"<span class='doc-id'>lexical coverage "
        f"{stats['matched_terms']}/{stats['query_terms']} query terms "
        f"({stats['coverage']*100:.0f}%)</span>"
    )

    st.markdown(
        f"<div class='{row_cls}'>"
        f"<div class='result-head'>{''.join(bits)}</div>"
        f"<div class='result-body'>{body}</div>"
        f"<div style='margin-top:.35rem'>{coverage}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_results(
    results: list[tuple[str, float]],
    query: str,
    collection,
    relevant: set[str] | None = None,
    stage1_ranks: dict[str, int] | None = None,
    score_label: str = "score",
    idf_lookup=None,
    limit: int = 10,
) -> None:
    relevant = relevant or set()
    stage1_ranks = stage1_ranks or {}
    if not results:
        st.info("No documents matched this query.")
        return

    for rank, (doc_id, score) in enumerate(results[:limit], start=1):
        meta = collection.documents.get(doc_id, {})
        old = stage1_ranks.get(doc_id)
        result_row(
            doc_id=doc_id,
            text=meta.get("text", ""),
            query=query,
            rank=rank,
            score=score,
            is_relevant=doc_id in relevant,
            stage1_rank=old,
            delta=(old - rank) if old else None,
            score_label=score_label,
            patent_id=meta.get("patent_id"),
            epo_link=meta.get("epo_link"),
            idf_lookup=idf_lookup,
        )


def gold_summary(results: list[tuple[str, float]], relevant: set[str], label: str) -> dict:
    """Where did the examiner-cited document(s) land in this ranking?"""
    ranks = [i for i, (d, _) in enumerate(results, start=1) if d in relevant]
    return {
        "label": label,
        "found": bool(ranks),
        "best_rank": min(ranks) if ranks else None,
        "n_found": len(ranks),
        "n_relevant": len(relevant),
    }
