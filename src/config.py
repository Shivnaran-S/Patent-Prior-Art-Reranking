"""
Every tunable number in the project lives here, in one place.

Nothing else in the codebase hard-codes a path, a model name or a parameter --
so when the professor asks "where do you set k1?", the answer is one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"      # cached indexes + embeddings
RESULTS_DIR = ROOT / "results"         # benchmark output

CORPUS_FILE = DATA_DIR / "corpus.jsonl"
QUERIES_FILE = DATA_DIR / "queries.jsonl"
QRELS_FILE = DATA_DIR / "qrels.jsonl"
HARD_NEG_FILE = DATA_DIR / "hard_negatives.jsonl"
STATS_FILE = DATA_DIR / "dataset_stats.json"

for _d in (ARTIFACT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# retrieval parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BM25Config:
    """Okapi BM25.

    k1  controls term-frequency saturation.  A word appearing 20 times is not
        20x more important than once; k1 decides how fast the reward flattens.
    b   controls length normalisation.  b=1 fully penalises long documents,
        b=0 ignores length entirely.
    """
    k1: float = 1.2
    b: float = 0.75


@dataclass(frozen=True)
class DenseConfig:
    """Dense bi-encoder: query and document are embedded SEPARATELY, so all
    document vectors can be computed once, offline, and reused forever."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 256
    batch_size: int = 64
    normalize: bool = True       # unit vectors => inner product == cosine


@dataclass(frozen=True)
class CrossEncoderConfig:
    """Cross-encoder: query and document go in TOGETHER, so the transformer's
    attention can compare every query word against every document word.  Far
    more accurate, far too slow to run on the whole corpus."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    finetuned_dir: Path = ARTIFACT_DIR / "crossencoder-patent-finetuned"
    # 256 tokens covers a median claim + passage pair in full; the MaxP
    # strategy exists for the tail that does not fit.
    max_length: int = 256
    batch_size: int = 32


@dataclass(frozen=True)
class FusionConfig:
    """Reciprocal Rank Fusion:  score(d) = sum_i 1 / (k + rank_i(d))

    RRF combines ranked lists using only RANKS, never raw scores -- which is
    why it can mix BM25 (unbounded) with cosine similarity (0..1) without any
    score calibration."""
    rrf_k: int = 60              # the standard constant from the RRF paper
    alpha: float = 0.5           # weight for score-level fusion (0=dense, 1=bm25)


@dataclass(frozen=True)
class PipelineConfig:
    first_stage_k: int = 100     # how many candidates stage 1 hands to stage 2
    final_k: int = 10            # how many results we finally show / score
    stage1: str = "hybrid"       # tfidf | bm25 | dense | hybrid
    stage2: str = "cross_encoder"  # none | ltr | cross_encoder


@dataclass(frozen=True)
class EvalConfig:
    cutoffs: tuple[int, ...] = (1, 5, 10, 20, 50, 100)
    primary_cutoff: int = 10
    n_eval_queries: int = 300    # sampled from the test split, seeded
    seed: int = 42


@dataclass(frozen=True)
class Settings:
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


SETTINGS = Settings()

# Human-readable labels used by the UI so method names stay consistent.
METHOD_LABELS = {
    "tfidf": "TF-IDF (vector space)",
    "bm25": "BM25 (probabilistic)",
    "dense": "Dense bi-encoder",
    "hybrid": "Hybrid (RRF)",
    "ltr": "Learning-to-Rank",
    "cross_encoder": "Cross-encoder",
    "cross_encoder_ft": "Cross-encoder (fine-tuned)",
}
