"""
STEP 3 OF THE PIPELINE  --  TRAIN THE TWO STAGE-2 RERANKERS.

    python scripts/train_reranker.py            # both
    python scripts/train_reranker.py --ltr      # just the fast linear model
    python scripts/train_reranker.py --ce       # just the cross-encoder

HARD NEGATIVE MINING  (the idea that makes reranker training work)
------------------------------------------------------------------
If we train with random negatives, the task is trivial: a paragraph about
peanut planters is obviously not prior art for an optical amplifier, and the
model learns nothing useful.  The documents the reranker actually has to judge
are the ones stage 1 ranked highly -- near-misses from the same technical
field.  So for every training query we run the real first stage, take its
top-ranked NON-relevant documents, and use those as negatives:

    positives = examiner-cited paragraphs THAT STAGE 1 ACTUALLY RETRIEVED
    negatives = top hybrid-retrieval results that the examiner did NOT cite
                (+ the examiner-inspected negatives PatentMatch gives us)

The model is then trained on exactly the decisions it will face at inference.

The capitalised part is the subtle bit, and we learned it the hard way. Only
~40% of cited paragraphs are inside the stage-1 pool at all. Train on the rest
as positives and "relevant" silently comes to mean "the retrievers scored this
low", because that is the only thing those documents have in common -- while
every negative is a document the retrievers scored high. The model dutifully
learns the inverted rule. Our first attempt did exactly that and scored
nDCG@10 = 0.010 on dev, against 0.095 for doing no reranking at all.

TWO MODELS, TWO LOSSES
----------------------
1. Learning-to-Rank: pairwise logistic loss on feature differences (RankNet).
   Trains in under a second.
2. Cross-encoder: binary cross-entropy on the relevance logit, fine-tuning all
   22M parameters of MiniLM-L6 for a couple of epochs on CPU.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config                              # noqa: E402
from src.dataset import load_collection             # noqa: E402
from src.ltr_reranker import PairwiseLTR            # noqa: E402
from src.metrics import aggregate, evaluate_run     # noqa: E402
from src.pipeline import LTR_PATH, SearchEngine     # noqa: E402

SEED = 42
N_TRAIN_QUERIES = 8000        # training queries used for mining
MINE_DEPTH = 100              # depth of the stage-1 pool; matches inference
NEG_PER_POS = 3               # negatives sampled per positive


# --------------------------------------------------------------------------- #
# shared: mine (query, positive, negatives) triples from the training split
# --------------------------------------------------------------------------- #
def mine_training_examples(engine: SearchEngine, n_queries: int = N_TRAIN_QUERIES) -> list[dict]:
    col = engine.collection
    qids = col.sample_queries("train", n_queries, SEED)
    rng = random.Random(SEED)
    examples: list[dict] = []

    print(f"Mining hard negatives from {len(qids):,} training queries "
          f"(stage 1 = hybrid, depth {MINE_DEPTH}) ...")
    t0 = time.perf_counter()
    for i, qid in enumerate(qids, 1):
        query = col.query(qid)["text"]
        gold = col.relevant(qid)
        if not gold:
            continue

        ranked = [d for d, _ in engine.retrieve(query, "hybrid", MINE_DEPTH)]
        # Hard negatives: highly ranked but not cited by the examiner.
        hard = [d for d in ranked if d not in gold]
        # Plus the negatives the examiner actually looked at and rejected.
        hard += [d for d in col.hard_negatives.get(qid, []) if d not in gold]
        if not hard:
            continue

        in_pool = set(ranked)
        for pos in gold:
            # Sample negatives from ACROSS the whole candidate pool, not just
            # its top. Taking only the highest-ranked negatives looks like the
            # "hardest" choice, but it teaches inversion: if the positive sits
            # at rank 20 and every negative comes from ranks 1-8, the only rule
            # that fits the training data is "a worse rank means more relevant".
            # Sampling uniformly from the pool puts negatives both above and
            # below the positive, so rank and score stay usable signals.
            negs = rng.sample(hard, NEG_PER_POS) if len(hard) > NEG_PER_POS else list(hard)
            examples.append({
                "query_id": qid, "query": query, "pos": pos, "negs": negs,
                "pos_in_pool": pos in in_pool,
            })

        if i % 500 == 0:
            print(f"  {i:,}/{len(qids):,} queries, {len(examples):,} examples "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- keep only positives that stage 1 actually retrieved ---------------- #
    #
    # THIS FILTER IS NOT OPTIONAL, and it took a failed training run to learn why.
    #
    # Only ~40% of examiner-cited paragraphs are inside the stage-1 pool. If the
    # other 60% are used as training positives, then "positive" correlates with
    # "the retrievers scored this LOW" -- because that is the only reason those
    # documents are missing from the pool -- while every negative is by
    # construction a document the retrievers scored HIGH.
    #
    # A model fed that data learns the inverted rule "looks dissimilar to the
    # query => relevant". We measured exactly this: a cross-encoder fine-tuned
    # on the unfiltered set scored nDCG@10 = 0.010 on dev against 0.095 for no
    # reranking at all, and REVERSING its ranking improved it.
    #
    # Restricting to in-pool positives makes the training distribution identical
    # to the inference one: discriminate the cited paragraph from other
    # highly-ranked paragraphs in the same technical field, which is the actual
    # reranking task.
    usable = [ex for ex in examples if ex["pos_in_pool"]]
    print(f"  mined {len(examples):,} groups in {time.perf_counter() - t0:.0f}s; "
          f"{len(usable):,} have the positive inside the stage-1 top-{MINE_DEPTH} "
          f"({len(usable)/max(len(examples),1)*100:.0f}%) and are kept for training\n")
    return usable


# --------------------------------------------------------------------------- #
# 1. Learning-to-Rank
# --------------------------------------------------------------------------- #
def train_ltr(engine: SearchEngine, examples: list[dict]) -> PairwiseLTR:
    print("=" * 62)
    print("TRAINING 1/2  --  pairwise Learning-to-Rank (9 features)")
    print("=" * 62)

    # mine_training_examples() has already restricted these to in-pool
    # positives, which the feature-based model needs even more than the
    # cross-encoder does: an out-of-pool positive has all-zero score features.
    usable = examples
    print(f"  {len(usable):,} training groups")

    pos_rows: list[np.ndarray] = []
    neg_rows: list[np.ndarray] = []

    t0 = time.perf_counter()
    for i, ex in enumerate(usable, 1):
        candidates = [ex["pos"]] + ex["negs"]
        feats = engine._ltr_features(ex["query"], candidates)
        for j in range(1, len(candidates)):
            pos_rows.append(feats[0])
            neg_rows.append(feats[j])
        if i % 500 == 0:
            print(f"  featurised {i:,}/{len(usable):,} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)

    if not pos_rows:
        raise RuntimeError("No usable LTR training pairs were mined.")

    pos = np.vstack(pos_rows)
    neg = np.vstack(neg_rows)
    print(f"  {len(pos):,} training pairs, {pos.shape[1]} features")

    model = PairwiseLTR().fit(pos, neg, verbose=True)
    acc = model.training_accuracy(pos, neg)
    print(f"  pairwise training accuracy: {acc:.3f}")
    print("  learned weights:")
    for row in model.weight_table():
        bar = "#" * int(abs(row["weight"]) * 8)
        print(f"    {row['feature']:<18} {row['weight']:+8.3f}  {bar}")

    model.save(LTR_PATH)
    print(f"  saved -> {LTR_PATH.relative_to(ROOT)}\n")
    return model


# --------------------------------------------------------------------------- #
# 2. Cross-encoder fine-tuning
# --------------------------------------------------------------------------- #
def train_cross_encoder(
    engine: SearchEngine,
    examples: list[dict],
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_pairs: int = 12_000,
) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 62)
    print("TRAINING 2/2  --  fine-tuning the cross-encoder")
    print("=" * 62)

    col = engine.collection
    cfg = config.SETTINGS.cross_encoder
    rng = random.Random(SEED)

    # ---- build the labelled pair set --------------------------------------- #
    pairs: list[tuple[str, str, float]] = []
    for ex in examples:
        pairs.append((ex["query"], col.documents[ex["pos"]]["text"], 1.0))
        for neg in ex["negs"]:
            pairs.append((ex["query"], col.documents[neg]["text"], 0.0))
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    n_pos = sum(1 for *_, y in pairs if y == 1.0)
    print(f"  {len(pairs):,} training pairs  ({n_pos:,} positive / "
          f"{len(pairs) - n_pos:,} negative)")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    # Fine-tuning is the one place worth using every core.
    import os
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    def collate(batch):
        enc = tokenizer(
            [b[0] for b in batch], [b[1] for b in batch],
            padding=True, truncation="longest_first",
            max_length=cfg.max_length, return_tensors="pt",
        )
        enc["labels"] = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        return enc

    loader = DataLoader(pairs, batch_size=batch_size, shuffle=True, collate_fn=collate)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr, total_steps=total_steps, pct_start=0.1, anneal_strategy="linear"
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"  {epochs} epochs x {len(loader)} steps = {total_steps} steps "
          f"(batch {batch_size}, lr {lr}, max_len {cfg.max_length})")
    print("  training on CPU -- expect roughly 15-40 minutes\n")

    model.train()
    history: list[dict] = []
    t0 = time.perf_counter()
    step = 0
    for epoch in range(1, epochs + 1):
        running = 0.0
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            step += 1
            running += loss.item()
            if step % 25 == 0:
                mean = running / 25
                running = 0.0
                elapsed = time.perf_counter() - t0
                eta = elapsed / step * (total_steps - step)
                history.append({"step": step, "loss": mean})
                print(f"  epoch {epoch}  step {step:>4}/{total_steps}  "
                      f"loss {mean:.4f}  elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m",
                      flush=True)

    out = cfg.finetuned_dir
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "training_history.json").write_text(
        json.dumps(
            {
                "history": history,
                "n_pairs": len(pairs),
                "n_positive": n_pos,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_length": cfg.max_length,
                "base_model": cfg.model_name,
                "minutes": round((time.perf_counter() - t0) / 60, 1),
                "mining": {"stage1": "hybrid", "depth": MINE_DEPTH,
                           "neg_per_pos": NEG_PER_POS, "train_queries": N_TRAIN_QUERIES},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  saved -> {out.relative_to(ROOT)}  "
          f"({(time.perf_counter() - t0)/60:.1f} minutes)\n")


# --------------------------------------------------------------------------- #
# dev-set sanity check
# --------------------------------------------------------------------------- #
def check_on_dev(engine: SearchEngine, n: int = 60) -> None:
    """Did training actually help?  Measured on the DEV split, never on test."""
    print("=" * 62)
    print(f"DEV CHECK  --  {n} held-out dev queries, nDCG@10")
    print("=" * 62)
    col = engine.collection

    # Always score what is actually SAVED, not whatever this process happens to
    # hold in memory. A long run can have its artifacts replaced by a separate
    # training invocation while the cross-encoder is still training, and
    # reporting the stale in-memory model would be badly misleading.
    if LTR_PATH.exists():
        engine.ltr = PairwiseLTR.load(LTR_PATH)
    engine._cross.clear()

    qids = col.sample_queries("dev", n, SEED)

    configs = [("hybrid", "none"), ("hybrid", "ltr"),
               ("hybrid", "cross_encoder"), ("hybrid", "cross_encoder_ft")]
    for s1, s2 in configs:
        try:
            per = []
            for qid in qids:
                res = engine.search(col.query(qid)["text"], stage1=s1, stage2=s2,
                                    first_stage_k=100, final_k=100)
                per.append(evaluate_run([d for d, _ in res.final], col.relevant(qid)))
            a = aggregate(per)
            print(f"  {s1} + {s2:<18} nDCG@10={a['ndcg@10']:.4f}  "
                  f"MRR={a['mrr']:.4f}  R@10={a['recall@10']:.4f}")
        except FileNotFoundError as exc:
            print(f"  {s1} + {s2:<18} skipped ({exc})")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ltr", action="store_true", help="train only the LTR model")
    ap.add_argument("--ce", action="store_true", help="train only the cross-encoder")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--queries", type=int, default=N_TRAIN_QUERIES)
    ap.add_argument("--no-dev-check", action="store_true")
    args = ap.parse_args()

    do_ltr = args.ltr or not (args.ltr or args.ce)
    do_ce = args.ce or not (args.ltr or args.ce)

    engine = SearchEngine(load_collection()).build(progress=lambda m: print(f"  {m}"))
    print()
    examples = mine_training_examples(engine, args.queries)

    if do_ltr:
        engine.ltr = train_ltr(engine, examples)
    if do_ce:
        train_cross_encoder(engine, examples, epochs=args.epochs)

    if not args.no_dev_check:
        check_on_dev(engine)


if __name__ == "__main__":
    main()
