"""
Cached application state.

Streamlit re-runs the whole script on every interaction, so anything expensive
must be cached or the app would rebuild the indexes on every click.
`@st.cache_resource` keeps ONE copy of the search engine alive for the whole
session, which is why clicking around the app stays instant.
"""

from __future__ import annotations

import streamlit as st

from src.dataset import load_collection
from src.evaluate import load_report
from src.pipeline import DENSE_PATH, LTR_PATH, SearchEngine
from src import config


@st.cache_resource(show_spinner=False)
def get_engine() -> SearchEngine:
    """Build (or load from artifacts/) every index, once per session.

    The transformers are also WARMED UP here -- loaded and run on one dummy
    input. Without that, the first search of the session would include ~30
    seconds of model loading in its reported latency, which would make the
    timing numbers on the Live Search page meaningless.
    """
    with st.spinner("Loading indexes ..."):
        engine = SearchEngine(load_collection())
        engine.build(include_dense=DENSE_PATH.exists())

    if engine.dense is not None:
        with st.spinner("Warming up the bi-encoder ..."):
            try:
                engine.dense.encode(["warm up"])
            except Exception:
                pass

    ft = config.SETTINGS.cross_encoder.finetuned_dir
    variant = "finetuned" if (ft / "config.json").exists() else "base"
    with st.spinner(f"Warming up the cross-encoder ({variant}) ..."):
        try:
            engine.cross_encoder(variant).score("warm up", ["x"], {"x": "warm up"})
        except Exception:
            pass

    return engine


@st.cache_resource(show_spinner=False)
def get_collection():
    return load_collection()


@st.cache_data(show_spinner=False)
def get_report() -> dict | None:
    return load_report()


@st.cache_data(show_spinner=False)
def get_ablations() -> dict | None:
    return load_report("ablations.json")


def availability() -> dict:
    """What has been built so far -- pages use this to stay usable even if the
    user has not run every script yet."""
    ft = config.SETTINGS.cross_encoder.finetuned_dir
    return {
        "dense": DENSE_PATH.exists(),
        "ltr": LTR_PATH.exists(),
        "finetuned_ce": (ft / "config.json").exists(),
        "benchmark": (config.RESULTS_DIR / "benchmark.json").exists(),
        "ablations": (config.RESULTS_DIR / "ablations.json").exists(),
    }


def missing_banner(requirements: list[str]) -> bool:
    """Show a friendly instruction if a prerequisite is missing.
    Returns True when everything needed is present."""
    have = availability()
    cmds = {
        "dense": ("the dense index", "python scripts/build_indexes.py"),
        "ltr": ("the Learning-to-Rank weights", "python scripts/train_reranker.py --ltr"),
        "finetuned_ce": ("the fine-tuned cross-encoder", "python scripts/train_reranker.py --ce"),
        "benchmark": ("the benchmark results", "python scripts/run_benchmark.py"),
        "ablations": ("the ablation results", "python scripts/run_ablations.py"),
    }
    missing = [r for r in requirements if not have.get(r)]
    if not missing:
        return True
    for key in missing:
        what, cmd = cmds[key]
        st.warning(f"This section needs {what}. Run:  `{cmd}`")
    return False
