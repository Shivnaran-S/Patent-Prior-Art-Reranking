"""
STAGE 2  --  THE CROSS-ENCODER RERANKER.  This is the heart of the project.

BI-ENCODER vs CROSS-ENCODER
---------------------------
Bi-encoder (stage 1):
        query ---> [transformer] ---> vector  \
                                               >--- cosine
        doc   ---> [transformer] ---> vector  /

    The two texts NEVER meet.  Fast (documents precomputed), but the model has
    to compress a whole document into one vector before it knows what will be
    asked of it.

Cross-encoder (stage 2):
        [CLS] query [SEP] doc [SEP] ---> [transformer] ---> relevance score

    Both texts enter the SAME forward pass, so self-attention lets every query
    token attend to every document token.  The model can check "does this
    paragraph actually disclose *this* mechanism?" instead of comparing two
    summaries.  Typically a large accuracy gain.

THE COST, AND WHY THE CASCADE EXISTS
------------------------------------
A cross-encoder cannot precompute anything: its input depends on the query, so
scoring the full corpus would mean 11,849 transformer passes PER QUERY.
Measured on this machine that is minutes per query.  Scoring the top 100 from
stage 1 is ~100 passes -- about a second -- for nearly all of the benefit.
That trade-off is the entire justification for retrieve-then-rerank.

LONG DOCUMENTS
--------------
A patent paragraph can exceed the model's 512-token window.  Two standard
options, both implemented:
  * "truncate"  -- cut the document at max_length (fast, may lose the key
                   sentence if it appears late)
  * "max_passage" -- split the document into overlapping windows, score each,
                   keep the best ("MaxP").  Slower but robust, because prior
                   art often hinges on one specific sentence.
"""

from __future__ import annotations

import numpy as np

from . import config


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(
        self,
        model_name: str | None = None,
        long_doc_strategy: str = "truncate",
        passage_words: int = 180,
        passage_stride: int = 120,
    ) -> None:
        cfg = config.SETTINGS.cross_encoder
        self.model_name = model_name or cfg.model_name
        self.max_length = cfg.max_length
        self.batch_size = cfg.batch_size
        self.long_doc_strategy = long_doc_strategy
        self.passage_words = passage_words
        self.passage_stride = passage_stride
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------ model #
    @property
    def is_finetuned(self) -> bool:
        return str(self.model_name) == str(config.SETTINGS.cross_encoder.finetuned_dir)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()

    @classmethod
    def finetuned(cls, **kwargs) -> "CrossEncoderReranker | None":
        """Load our fine-tuned checkpoint if scripts/train_reranker.py has run."""
        path = config.SETTINGS.cross_encoder.finetuned_dir
        if not (path / "config.json").exists():
            return None
        return cls(model_name=str(path), **kwargs)

    # ------------------------------------------------------------------ score #
    def _score_pairs(self, pairs: list[tuple[str, str]], progress=None) -> np.ndarray:
        """Run the transformer over (query, doc) pairs and return one logit each.

        LENGTH-SORTED BATCHING
        ----------------------
        Every sequence in a batch is padded to the longest one in that batch,
        and the transformer does the full attention computation over the padding
        too. Our passages range from ~30 to ~380 words, so a batch mixing a
        short passage with a long one wastes most of its work on padding.

        Sorting the pairs by length first means each batch holds
        similar-length sequences, so there is far less padding to compute.
        The scores are IDENTICAL either way -- each pair is scored
        independently and the attention mask already excludes padding -- so
        this is free speed. We undo the sort at the end to return scores in
        the caller's original order.
        """
        import torch

        self._ensure_model()
        total = len(pairs)
        if total == 0:
            return np.zeros(0, dtype=np.float32)

        order = sorted(range(total), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        scores = np.zeros(total, dtype=np.float32)

        done = 0
        for start in range(0, total, self.batch_size):
            idx = order[start : start + self.batch_size]
            chunk = [pairs[i] for i in idx]
            with torch.no_grad():
                enc = self._tokenizer(
                    [a for a, _ in chunk],
                    [b for _, b in chunk],
                    padding=True,
                    truncation="longest_first",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                logits = self._model(**enc).logits
                # ms-marco cross-encoders are single-logit regressors; some
                # checkpoints are 2-class, so handle both.
                vals = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, 1]
            scores[idx] = vals.numpy().astype(np.float32)
            done += len(idx)
            if progress is not None:
                progress(done, total)
        return scores

    def _windows(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.passage_words:
            return [text]
        return [
            " ".join(words[i : i + self.passage_words])
            for i in range(0, len(words), self.passage_stride)
            if i < len(words)
        ]

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        doc_texts: dict[str, str],
        top_k: int | None = None,
        progress=None,
    ) -> list[tuple[str, float]]:
        """Rescore stage-1 candidates and return them in the new order."""
        if not candidates:
            return []
        doc_ids = [d for d, _ in candidates]

        if self.long_doc_strategy == "max_passage":
            pairs: list[tuple[str, str]] = []
            owner: list[int] = []
            for i, d in enumerate(doc_ids):
                for window in self._windows(doc_texts.get(d, "")):
                    pairs.append((query, window))
                    owner.append(i)
            flat = self._score_pairs(pairs, progress=progress)
            scores = np.full(len(doc_ids), -1e9, dtype=np.float32)
            for value, i in zip(flat, owner):           # MaxP: best window wins
                scores[i] = max(scores[i], value)
        else:
            pairs = [(query, doc_texts.get(d, "")) for d in doc_ids]
            scores = self._score_pairs(pairs, progress=progress)

        order = np.argsort(-scores, kind="stable")
        out = [(doc_ids[i], float(scores[i])) for i in order]
        return out[:top_k] if top_k else out

    def score(self, query: str, doc_ids: list[str], doc_texts: dict[str, str]) -> dict[str, float]:
        if not doc_ids:
            return {}
        scores = self._score_pairs([(query, doc_texts.get(d, "")) for d in doc_ids])
        return dict(zip(doc_ids, (float(s) for s in scores)))
