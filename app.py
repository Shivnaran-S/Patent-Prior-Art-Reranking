"""
PATENT PRIOR-ART SEARCH  --  Streamlit application entry point.

    streamlit run app.py

Seven pages, in the order you would present them:

    1  Overview        the problem, the pipeline, the headline numbers
    2  Dataset         what we search, and where the relevance labels come from
    3  Live Search     the demo: run the two-stage pipeline on a real claim
    4  Evaluation      every method on the same test queries, with significance
    5  Ablations       what each component is actually worth
    6  Concepts        the IR theory, with live calculators
    7  Error Analysis  where it still fails, and why

Run `python scripts/setup.py` once before the first launch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Patent Prior-Art Search",
    page_icon="assets/favicon.png" if (ROOT / "assets/favicon.png").exists() else ":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src import config  # noqa: E402
from ui import theme  # noqa: E402
from ui.state import availability  # noqa: E402

theme.inject()

# The app is useless without the test collection, and a raw FileNotFoundError
# traceback is a poor way to say so.
if not config.CORPUS_FILE.exists():
    st.markdown(
        "<div class='page-title'>Patent Prior-Art Search</div>"
        "<div class='page-sub'>The project has not been built yet.</div>",
        unsafe_allow_html=True,
    )
    st.error("`data/corpus.jsonl` is missing, so there is nothing to search.")
    st.markdown("Build everything with one command, then reload this page:")
    st.code("python scripts/setup.py", language="bash")
    st.caption(
        "That downloads the PatentMatch dataset, builds the BM25, TF-IDF and dense "
        "indexes, trains both rerankers and runs the evaluation. Add `--quick` to skip "
        "the cross-encoder fine-tuning and the slow ablations."
    )
    st.stop()

PAGES = [
    ("ui/pages/1_overview.py", "Overview", ":material/home:"),
    ("ui/pages/2_dataset.py", "Dataset", ":material/database:"),
    ("ui/pages/3_search.py", "Live Search", ":material/search:"),
    ("ui/pages/4_evaluation.py", "Evaluation", ":material/bar_chart:"),
    ("ui/pages/5_ablations.py", "Ablations", ":material/tune:"),
    ("ui/pages/6_concepts.py", "IR Concepts", ":material/school:"),
    ("ui/pages/7_errors.py", "Error Analysis", ":material/bug_report:"),
]

pages = [
    st.Page(path, title=title, icon=icon, default=(i == 0))
    for i, (path, title, icon) in enumerate(PAGES)
]

with st.sidebar:
    st.markdown(
        "<div style='font-weight:740;font-size:1.02rem;letter-spacing:-.02em;"
        "line-height:1.25'>Patent Prior-Art Search</div>"
        "<div style='color:#5B6478;font-size:.78rem;margin:.1rem 0 .1rem 0'>"
        "Two-stage retrieve &amp; rerank</div>"
        "<div style='color:#5B6478;font-size:.74rem;margin-bottom:.6rem'>"
        "Shivnaran S (22PD32) · Diravina E I (22PD12)</div>",
        unsafe_allow_html=True,
    )

nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.divider()
    have = availability()
    st.markdown(
        "<div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;"
        "color:#5B6478;font-weight:650;margin-bottom:.35rem'>Components built</div>",
        unsafe_allow_html=True,
    )
    labels = {
        "dense": "Dense index",
        "ltr": "Learning-to-Rank",
        "finetuned_ce": "Fine-tuned cross-encoder",
        "benchmark": "Benchmark results",
        "ablations": "Ablation results",
    }
    rows = []
    for key, label in labels.items():
        built = have.get(key)
        colour = theme.GOOD if built else "#C2C8D6"
        mark = "&#9679;" if built else "&#9675;"     # filled vs hollow dot
        text = theme.INK if built else "#9AA2B4"
        rows.append(
            f"<div style='font-size:.76rem;padding:.1rem 0;color:{text}'>"
            f"<span style='color:{colour};font-size:.7rem'>{mark}</span> {label}</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption("BM25 and TF-IDF are always available.")

nav.run()
