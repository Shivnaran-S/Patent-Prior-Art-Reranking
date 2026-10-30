"""
EXPLAINABILITY HELPERS  --  turning scores into something a human can check.

A ranked list on its own is not evidence that the system works.  These helpers
let the UI answer three questions for any single result:

  1. WHICH WORDS matched?          -> highlight_overlap()
  2. HOW MUCH did each contribute? -> BM25Retriever.explain() (in bm25_retriever)
  3. WHY did the reranker MOVE it? -> compare_rankings(), explain_movement()

Point 3 is the demo money shot: a document that BM25 put at rank 40 because it
shares few words, which the cross-encoder lifts to rank 1 because it actually
describes the same mechanism.
"""

from __future__ import annotations

import html
import re

from .text_utils import STOPWORDS, analyze, stem


def highlight_overlap(text: str, query: str, max_words: int = 120) -> str:
    """Return HTML with query terms marked.

    Matching happens on STEMS (so "amplification" in the query highlights
    "amplifier" in the document), but the original surface word is displayed --
    which is what makes it readable.
    """
    q_stems = set(analyze(query))
    words = text.split()
    truncated = len(words) > max_words
    words = words[:max_words]

    out: list[str] = []
    for word in words:
        core = re.sub(r"[^A-Za-z0-9\-.]", "", word).lower()
        is_match = bool(core) and core not in STOPWORDS and stem(core) in q_stems
        safe = html.escape(word)
        out.append(f"<mark class='hit'>{safe}</mark>" if is_match else safe)
    return " ".join(out) + (" <span class='muted'>...</span>" if truncated else "")


def overlap_stats(query: str, doc_text: str, idf_lookup=None) -> dict:
    """Lexical overlap between a query and a document, for the result card."""
    q_terms = set(analyze(query))
    d_terms = set(analyze(doc_text))
    shared = q_terms & d_terms
    missing = q_terms - d_terms

    stats = {
        "query_terms": len(q_terms),
        "matched_terms": len(shared),
        "coverage": len(shared) / max(len(q_terms), 1),
        "shared": sorted(shared),
        "missing": sorted(missing),
    }
    if idf_lookup is not None:
        # The rarest query terms we FAILED to match: these are usually the
        # reason a relevant document was missed by a lexical retriever.
        stats["missing_rare"] = sorted(
            missing, key=lambda t: -idf_lookup(t)
        )[:8]
        stats["matched_rare"] = sorted(shared, key=lambda t: -idf_lookup(t))[:8]
    return stats


def compare_rankings(
    before: list[tuple[str, float]],
    after: list[tuple[str, float]],
    top_k: int = 10,
) -> list[dict]:
    """Align two rankings so the UI can draw the rank-movement arrows."""
    before_rank = {d: i + 1 for i, (d, _) in enumerate(before)}
    rows = []
    for new_rank, (doc_id, score) in enumerate(after[:top_k], start=1):
        old = before_rank.get(doc_id)
        rows.append(
            {
                "doc_id": doc_id,
                "new_rank": new_rank,
                "old_rank": old,
                "delta": (old - new_rank) if old is not None else None,
                "score": score,
            }
        )
    return rows


def explain_movement(delta: int | None) -> tuple[str, str]:
    """(symbol, css class) for a rank change."""
    if delta is None:
        return "new", "move-new"
    if delta > 0:
        return f"up {delta}", "move-up"
    if delta < 0:
        return f"down {abs(delta)}", "move-down"
    return "same", "move-flat"


def vocabulary_mismatch_example(query: str, doc_text: str, idf_lookup) -> dict | None:
    """Detect the textbook failure case: a relevant document that shares almost
    no rare vocabulary with the query.  Used on the Error Analysis page to show
    WHY dense retrieval is needed, with a real example rather than an assertion.
    """
    stats = overlap_stats(query, doc_text, idf_lookup)
    if stats["coverage"] > 0.35:
        return None
    return {
        "coverage": stats["coverage"],
        "matched": stats.get("matched_rare", []),
        "missed": stats.get("missing_rare", []),
    }


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def best_matching_sentence(query: str, doc_text: str) -> str:
    """The single sentence of the document with the most query-term overlap --
    a cheap, readable stand-in for 'here is the passage that matters'."""
    q = set(analyze(query))
    best, best_score = "", -1.0
    for sent in sentence_split(doc_text) or [doc_text]:
        terms = set(analyze(sent))
        if not terms:
            continue
        score = len(q & terms) / (len(terms) ** 0.5)
        if score > best_score:
            best, best_score = sent, score
    return best or doc_text[:300]
