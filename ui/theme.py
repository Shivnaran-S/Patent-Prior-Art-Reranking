"""
Visual theme + reusable UI pieces for the Streamlit app.

All CSS lives here so the page modules stay readable and contain only logic.
The palette is deliberately high-contrast: the app is meant to be shown on a
classroom projector, where low-contrast dark themes wash out.
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #
INK = "#141A2E"
MUTED = "#5B6478"
LINE = "#E3E6F0"
SURFACE = "#FFFFFF"
SOFT = "#F5F6FA"

PRIMARY = "#4338CA"      # indigo   - the pipeline / our method
LEXICAL = "#B45309"      # amber    - BM25 / TF-IDF (lexical)
DENSE = "#0E7490"        # teal     - dense / semantic
FUSION = "#7C3AED"       # violet   - hybrid fusion
RERANK = "#BE185D"       # magenta  - stage 2 reranking
GOOD = "#047857"         # green    - relevant / improvement
BAD = "#B91C1C"          # red      - irrelevant / regression

METHOD_COLORS = {
    "tfidf": "#D97706",
    "bm25": LEXICAL,
    "dense": DENSE,
    "hybrid": FUSION,
    "hybrid_ltr": "#4F46E5",
    "bm25_ce": "#DB2777",
    "hybrid_ce": RERANK,
    "hybrid_ce_ft": "#86198F",
    "ltr": "#4F46E5",
    "cross_encoder": RERANK,
    "cross_encoder_ft": "#86198F",
    "none": MUTED,
}

CSS = f"""
<style>
/* ---------- layout ---------- */
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
#MainMenu, footer {{ visibility: hidden; }}

html, body, [class*="st-"] {{
  -webkit-font-smoothing: antialiased;
}}

/* ---------- headings ---------- */
.page-title {{
  font-size: 1.95rem; font-weight: 740; letter-spacing: -0.025em;
  color: {INK}; margin: 0 0 .15rem 0; line-height: 1.15;
}}
.page-sub {{ color: {MUTED}; font-size: .97rem; margin: 0 0 1.4rem 0; max-width: 70ch; }}
.section-title {{
  font-size: 1.08rem; font-weight: 680; color: {INK};
  margin: 1.9rem 0 .7rem 0; display: flex; align-items: center; gap: .55rem;
}}
.section-title::before {{
  content: ""; width: 3px; height: 1.05em; background: {PRIMARY}; border-radius: 2px;
}}

/* ---------- cards ---------- */
.card {{
  background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px;
  padding: 1.05rem 1.2rem; margin-bottom: .85rem;
}}
.card-tight {{ padding: .8rem .95rem; }}
.card-scroll {{ max-height: 260px; overflow-y: auto; }}
.card-scroll p {{ color: #2A3147 !important; }}
.card h4 {{ margin: 0 0 .45rem 0; font-size: .95rem; font-weight: 670; color: {INK}; }}
.card p {{ margin: 0; color: {MUTED}; font-size: .89rem; line-height: 1.6; }}

/* ---------- metric tiles ---------- */
.tiles {{ display: flex; gap: .7rem; flex-wrap: wrap; margin-bottom: .4rem; }}
.tile {{
  flex: 1 1 150px; background: {SURFACE}; border: 1px solid {LINE};
  border-radius: 11px; padding: .8rem .95rem; position: relative; overflow: hidden;
}}
.tile::before {{
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--accent, {PRIMARY});
}}
.tile .label {{
  font-size: .685rem; text-transform: uppercase; letter-spacing: .07em;
  color: {MUTED}; font-weight: 640; margin-bottom: .2rem;
}}
.tile .value {{
  font-size: 1.52rem; font-weight: 710; color: {INK};
  font-variant-numeric: tabular-nums; line-height: 1.1;
}}
.tile .delta {{ font-size: .76rem; font-weight: 620; margin-top: .12rem; }}
.tile .hint {{ font-size: .72rem; color: {MUTED}; margin-top: .18rem; }}
.up {{ color: {GOOD}; }}
.down {{ color: {BAD}; }}

/* ---------- pills / badges ---------- */
.pill {{
  display: inline-block; padding: .14rem .55rem; border-radius: 999px;
  font-size: .72rem; font-weight: 650; letter-spacing: .01em;
  border: 1px solid transparent; white-space: nowrap;
}}
.pill-rel  {{ background: #ECFDF5; color: {GOOD}; border-color: #A7F3D0; }}
.pill-irr  {{ background: {SOFT};  color: {MUTED}; border-color: {LINE}; }}
.pill-lex  {{ background: #FFFBEB; color: {LEXICAL}; border-color: #FDE68A; }}
.pill-den  {{ background: #ECFEFF; color: {DENSE}; border-color: #A5F3FC; }}
.pill-fus  {{ background: #F5F3FF; color: {FUSION}; border-color: #DDD6FE; }}
.pill-rer  {{ background: #FDF2F8; color: {RERANK}; border-color: #FBCFE8; }}
.pill-pri  {{ background: #EEF2FF; color: {PRIMARY}; border-color: #C7D2FE; }}

/* ---------- result rows ---------- */
.result {{
  background: {SURFACE}; border: 1px solid {LINE}; border-left: 3px solid {LINE};
  border-radius: 10px; padding: .75rem .9rem; margin-bottom: .55rem;
}}
.result.relevant {{ border-left-color: {GOOD}; background: #FAFFFD; }}
.result-head {{
  display: flex; align-items: center; gap: .6rem; margin-bottom: .35rem;
  flex-wrap: wrap;
}}
.rank-badge {{
  min-width: 2.1rem; height: 2.1rem; border-radius: 8px; background: {SOFT};
  color: {INK}; display: inline-flex; align-items: center; justify-content: center;
  font-weight: 710; font-size: .9rem; border: 1px solid {LINE};
  font-variant-numeric: tabular-nums;
}}
.rank-badge.relevant {{ background: {GOOD}; color: #fff; border-color: {GOOD}; }}
.doc-id {{
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: .76rem; color: {MUTED};
}}
.result-body {{
  font-size: .855rem; line-height: 1.62; color: #2A3147; margin-top: .2rem;
}}
mark.hit {{
  background: #FEF3C7; color: #7C2D12; padding: 0 .12rem;
  border-radius: 3px; font-weight: 600;
}}
.muted {{ color: {MUTED}; }}

/* ---------- rank movement ---------- */
.move {{
  font-size: .72rem; font-weight: 680; padding: .1rem .42rem;
  border-radius: 5px; font-variant-numeric: tabular-nums; white-space: nowrap;
}}
.move-up   {{ background: #ECFDF5; color: {GOOD}; }}
.move-down {{ background: #FEF2F2; color: {BAD}; }}
.move-flat {{ background: {SOFT}; color: {MUTED}; }}
.move-new  {{ background: #EEF2FF; color: {PRIMARY}; }}

/* ---------- formula block ---------- */
.formula {{
  background: #0F172A; color: #E2E8F0; border-radius: 10px;
  padding: .95rem 1.15rem; font-family: ui-monospace, Consolas, monospace;
  font-size: .845rem; line-height: 1.85; overflow-x: auto; margin: .5rem 0 .9rem 0;
  white-space: pre;
}}
.formula .k {{ color: #7DD3FC; }}   /* variable  */
.formula .c {{ color: #86EFAC; }}   /* comment   */
.formula .o {{ color: #FDBA74; }}   /* operator  */

/* ---------- term contribution bars ---------- */
.termbar {{ display: flex; align-items: center; gap: .6rem; margin-bottom: .3rem; }}
.termbar .t {{
  font-family: ui-monospace, Consolas, monospace; font-size: .8rem;
  min-width: 8.5rem; color: {INK}; font-weight: 600;
}}
.termbar .b {{ flex: 1; height: 15px; background: {SOFT}; border-radius: 4px; overflow: hidden; }}
.termbar .b > span {{ display: block; height: 100%; background: {LEXICAL}; border-radius: 4px; }}
.termbar .v {{
  font-size: .76rem; color: {MUTED}; min-width: 7.5rem; text-align: right;
  font-variant-numeric: tabular-nums;
}}

/* ---------- stage diagram ---------- */
.stageflow {{ display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; margin: .3rem 0 1rem 0; }}
.stagebox {{
  flex: 1 1 165px; border: 1px solid {LINE}; border-radius: 10px;
  padding: .7rem .8rem; background: {SURFACE}; position: relative;
}}
.stagebox .n {{
  font-size: .66rem; text-transform: uppercase; letter-spacing: .08em;
  font-weight: 680; color: {MUTED};
}}
.stagebox .h {{ font-size: .92rem; font-weight: 690; color: {INK}; margin: .12rem 0; }}
.stagebox .d {{ font-size: .76rem; color: {MUTED}; line-height: 1.5; }}
.arrow {{
  align-self: center; color: {MUTED}; font-size: 1.2rem; padding: 0 .5rem;
  font-weight: 700;
}}

/* ---------- misc ---------- */
.kv {{ display: flex; justify-content: space-between; padding: .3rem 0; border-bottom: 1px dashed {LINE}; font-size: .85rem; }}
.kv:last-child {{ border-bottom: none; }}
.kv .k {{ color: {MUTED}; }}
.kv .v {{ color: {INK}; font-weight: 620; font-variant-numeric: tabular-nums; }}

.note {{
  background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 10px;
  padding: .7rem .9rem; font-size: .85rem; color: #312E81; line-height: 1.6;
  margin: .5rem 0 .9rem 0;
}}
.warn {{
  background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px;
  padding: .7rem .9rem; font-size: .85rem; color: #78350F; line-height: 1.6;
  margin: .5rem 0 .9rem 0;
}}

div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 620; font-size: .9rem; }}
hr {{ margin: 1.4rem 0; border-color: {LINE}; }}
</style>
"""


def inject() -> None:
    """Called once per page render."""
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# small components
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='page-sub'>{subtitle}</div>", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def tiles(items: list[dict]) -> None:
    """items: [{label, value, delta?, hint?, color?}]"""
    html = ["<div class='tiles'>"]
    for it in items:
        colour = it.get("color", PRIMARY)
        delta = ""
        if it.get("delta"):
            text = str(it["delta"])
            cls = "down" if text.lstrip().startswith(("-", "−")) else "up"
            delta = f"<div class='delta {cls}'>{text}</div>"
        hint = f"<div class='hint'>{it['hint']}</div>" if it.get("hint") else ""
        html.append(
            f"<div class='tile' style='--accent:{colour}'>"
            f"<div class='label'>{it['label']}</div>"
            f"<div class='value'>{it['value']}</div>{delta}{hint}</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def card(title: str, body: str, extra_class: str = "") -> None:
    st.markdown(
        f"<div class='card {extra_class}'><h4>{title}</h4><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def warn(text: str) -> None:
    st.markdown(f"<div class='warn'>{text}</div>", unsafe_allow_html=True)


def formula(text: str) -> None:
    st.markdown(f"<div class='formula'>{text}</div>", unsafe_allow_html=True)


def pill(text: str, kind: str = "pri") -> str:
    return f"<span class='pill pill-{kind}'>{text}</span>"


def kv_list(pairs: list[tuple[str, str]]) -> None:
    rows = "".join(
        f"<div class='kv'><span class='k'>{k}</span><span class='v'>{v}</span></div>"
        for k, v in pairs
    )
    st.markdown(f"<div class='card card-tight'>{rows}</div>", unsafe_allow_html=True)


def stage_flow(stages: list[dict]) -> None:
    """stages: [{n, h, d, color}] rendered as boxes joined by arrows."""
    parts = ["<div class='stageflow'>"]
    for i, s in enumerate(stages):
        if i:
            parts.append("<div class='arrow'>&rarr;</div>")
        colour = s.get("color", LINE)
        parts.append(
            f"<div class='stagebox' style='border-top:2px solid {colour}'>"
            f"<div class='n'>{s['n']}</div>"
            f"<div class='h'>{s['h']}</div>"
            f"<div class='d'>{s['d']}</div></div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def term_bars(rows: list[dict], value_fmt: str = "{:.2f}") -> None:
    """rows: [{term, value, detail}] -- horizontal contribution bars."""
    if not rows:
        st.caption("No overlapping terms.")
        return
    top = max((r["value"] for r in rows), default=1.0) or 1.0
    html = []
    for r in rows:
        pct = max(2.0, r["value"] / top * 100)
        html.append(
            f"<div class='termbar'><div class='t'>{r['term']}</div>"
            f"<div class='b'><span style='width:{pct:.1f}%'></span></div>"
            f"<div class='v'>{value_fmt.format(r['value'])}"
            f"{'  ' + r['detail'] if r.get('detail') else ''}</div></div>"
        )
    st.markdown("".join(html), unsafe_allow_html=True)
