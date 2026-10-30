# Demo guide

A 10–12 minute walkthrough of the app, in order, with the line to say at each step and
the question it answers. Start the app a few minutes early — the first launch loads the
indexes and warms up both transformers, which takes about half a minute.

```bash
streamlit run app.py
```

---

## Before you start

| | |
|---|---|
| Open on | **Overview** page |
| Browser | full screen, zoom 90–100% |
| Have ready | the **Live Search** page already run once, so the models are warm |

If the professor only has five minutes, show **Live Search → Before vs after** and
**Evaluation → Accuracy vs cost**. Those two carry the whole project.

---

## 1 · Overview (1 min) — frame the problem

> "Our task is patent prior-art search: given a new patent claim, find the earlier
> documents that show it is not novel. Three things make it hard — the query is a
> 60-word technical claim rather than keywords, the same invention gets described in
> completely different words, and the most accurate model is far too slow to run over a
> whole collection."

Point at the four-box pipeline diagram.

> "So we built a cascade. Stage one is cheap and looks at everything; stage two is
> expensive and looks only at the shortlist. Stage one is tuned for recall, stage two for
> precision."

**If asked "why two stages?"** → the Evaluation page has the measured answer; promise it
and move on.

---

## 2 · Dataset (1.5 min) — establish the ground truth

Tab: **Where the labels come from**

> "The ground truth is not a heuristic. When a European patent examiner rejects a claim
> for lack of novelty, the search report has to cite the exact paragraph that disclosed
> it. Those citations are our relevance labels — expert human judgements."

Scroll to **A real labelled example** and read the blue box (query claim) and the green
box (what the examiner cited).

Point at the note underneath:

> "This claim shares only a small fraction of its index terms with the paragraph the
> examiner cited against it. That one number is the whole argument for the project —
> keyword matching alone cannot find this."

Tab: **How we split it**

> "One detail we are proud of. Several claims of the same application are near-duplicates
> and cite the same prior art, so a random split would leak. We build a graph of queries
> and documents and send whole connected components to one split, so no test query shares
> a relevant document with any training query."

**If asked "why not HUPD, you proposed it?"** → right-hand card answers it: HUPD has no
citation field and the PatentsView API is no longer openly reachable, so labels cannot be
derived from it. The pipeline is corpus-agnostic; only `prepare_data.py` would change.

---

## 3 · Live Search (4 min) — the demo

This is the centrepiece. Give it the most time.

1. Leave the default example claim (a real EPO patent, ~60 words).
2. Stage 1 = **Hybrid (RRF)**, Stage 2 = **Cross-encoder**, depth **50**.
3. Press **Run search**.

While it runs:

> "Stage one just scored all 11,849 documents in a few milliseconds and handed the top 50
> to the cross-encoder, which is now reading the claim and each candidate together — one
> transformer pass per candidate."

When it finishes, point at the five tiles:

> "Stage one took milliseconds, stage two took about a second for 50 documents. And here
> is the examiner-cited document's rank before and after reranking."

Tab: **Before vs after** — the slope chart.

> "Every line is one document. The green one is what the examiner actually cited. You can
> see the reranker pulling it up past documents that merely shared vocabulary."

Tab: **Why this document scored**

> "Because we implemented BM25 ourselves rather than calling a library, we can break any
> document's score back into the contribution of each query term — IDF times a saturating
> term frequency. And this box lists the rarest query terms the document does *not*
> contain, which is usually exactly why a lexical method missed it."

Tab: **Stage-1 arms**

> "Here are BM25 and the dense retriever separately, then their fusion. They disagree
> often, and that disagreement is what fusion exploits."

**Live variation to try if there is time:** set Stage 2 to **None**, run again, and show
the gold document falling back down. That is the single most convincing before/after.

---

## 4 · Evaluation (2.5 min) — prove it

Tab: **Results table**

> "Every method on the same held-out queries with the same judgements."

Point at the Recall@100 column:

> "Notice Recall@100 is identical for a method and that method plus a reranker — because
> a reranker only reorders what stage one found. It is the ceiling. nDCG and MRR are
> where reranking shows up."

Tab: **Significance**

> "A small average gain can be luck, so we ran a paired randomisation test: flip the sign
> of each query's difference ten thousand times and see how often chance produces a
> difference this large. That is the p-value."

Tab: **Accuracy vs cost**

> "This is the plot that justifies the whole design. The cross-encoder costs about ten
> milliseconds per document, so scoring all 11,849 for one query would take a couple of
> minutes. Reranking the top 50 takes about a second, for nearly all of the benefit."

---

## 5 · Ablations (1.5 min) — show it is measured, not assumed

Tab: **Fusion** — point at the curve.

> "This is a live sweep, computed right now on the dev split. The fused curve peaks
> *above* both end points. That is direct evidence the two retrievers fail on different
> queries — if they made the same mistakes, fusion could not beat the better of the two."

Drag the **RRF constant k** slider to show the numbers move.

Tab: **BM25 parameters** — drag `k1` and `b`.

> "Because we apply the BM25 weighting at query time instead of baking it into the index,
> both parameters can be changed live on the real collection."

Tab: **Rerank depth**

> "And this is the central trade-off, measured: quality flattens out well before cost
> does, which is why we operate at depth 50 to 100."

---

## 6 · IR Concepts (1.5 min) — the theory, on demand

Open the tab that matches whatever was asked.

- Tab 1 **Preprocessing** — type into the box, watch 21 tokens become 8 index terms, and
  point at the patent-specific stopwords (*wherein*, *comprising*, *said*).
- Tab 3 **BM25** — the saturation chart. "BM25 flattens, TF-IDF never does."
- Tab 4 **Dense retrieval** — type "a threaded fastener secures the two plates" vs
  "a screw holds the panels together". Zero shared terms, high cosine. That *is*
  vocabulary mismatch in one screen.
- Tab 7 **Metrics** — tick relevance boxes and watch all five metrics update. Best way to
  answer "what is nDCG?" without a slide.

---

## 7 · Error Analysis (1 min) — close honestly

> "Finally, where it still fails. Every failure is one of two kinds: stage one never
> retrieved the document, which no reranker can fix, or stage one found it and stage two
> left it too low. The split tells us what to work on next."

> "And the honest caveat: absolute scores on any citation-derived prior-art benchmark are
> pessimistic, because many genuinely relevant patents are never cited by an examiner, so
> unlabelled correct answers count against us. The comparison between methods is the
> meaningful result."

---

## Likely questions, with short answers

**Why is nDCG low in absolute terms?**
Claim-to-prior-art pairs have very low lexical overlap by construction, each query has
about one relevant document among 11,849, and unlabelled relevant documents count as
errors. Published results on PatentMatch are in the same range. The relative comparison
is what the experiment is for.

**Why RRF instead of just adding the scores?**
BM25 returns unbounded scores, cosine returns 0–1. Adding them needs calibration that
breaks whenever the distribution shifts. RRF uses ranks only, so there is nothing to
calibrate. Both are implemented — the Ablations page compares them.

**Did you implement BM25 yourself or use a library?**
From scratch, in `src/bm25_retriever.py`, which is why the app can show per-term score
attribution and sweep `k1`/`b` live. TF-IDF, the metrics and the Learning-to-Rank model
are also from scratch; only the transformers come from HuggingFace.

**How do you know your implementations are correct?**
41 tests, runnable in front of them:

```bash
python tests/test_core.py
python tests/test_dataset.py
```

Every one checks a value computed **by hand from the formula** on a four-document toy
collection — a retriever that is subtly wrong still returns a plausible ranked list, so
"it looks reasonable" proves nothing. `test_dataset.py` also proves the split does not
leak: no document is relevant to queries in two different splits.

Worth mentioning if asked what went wrong: this suite caught two real bugs. `explain()`
was indexing a sparse matrix by row in the wrong format and reading the wrong term
frequencies, and the Learning-to-Rank model was being trained on positives that stage 1
never retrieved — whose score features are all zero, so it learned "missing score =
relevant" and inverted the ranking. Both are fixed, and both are the kind of bug that
produces plausible-looking output rather than a crash.

**What are hard negatives and why do they matter?**
Random negatives make the task trivial — a peanut planter is obviously not prior art for
an optical amplifier. We run the real first stage on each training query and take its
top-ranked documents the examiner did *not* cite, so the model trains on exactly the
decisions it faces at inference.

**Why is the Learning-to-Rank model there at all?**
As the honest middle rung. It learns from the same data in under a second and its
weights are printable. If a nine-feature linear model got most of the gain, the
transformer would not be justified — so it sets the bar the cross-encoder has to clear.

**Could this scale to millions of patents?**
Stage 1 is the only part that touches the whole collection. Swap FAISS `IndexFlatIP` for
an IVF or HNSW index and the dense arm scales; BM25 is already an inverted index. Stage 2
cost does not change at all, because it only ever sees the top k. That is the structural
advantage of a cascade.

**What was the hardest part / what went wrong?**
This is the best question you can get — answer it with the training-distribution bug,
and show **IR Concepts → tab 6 → "The trap we fell into"**.

Only ~40% of examiner-cited paragraphs are retrievable in the stage-1 pool. We first
trained the cross-encoder on *all* cited paragraphs, so "relevant" silently came to mean
"the retrievers scored this low" — that is the one thing the unretrievable 60% share —
while every negative was a document the retrievers scored high. The model learned the
inverted rule and scored nDCG@10 = 0.010 on dev, against 0.095 for doing no reranking at
all. The giveaway was that *reversing* its output improved it.

The point worth making: the loss curve fell smoothly from 1.23 to 0.36 and the model
looked completely healthy. Nothing errored. It was caught only because we had a held-out
dev split and checked whether the number moved the right way. The fix is to train on
in-pool positives only, mined at the same depth used at inference.

**What would you do next?**
A patent-domain bi-encoder (PatentSBERTa, PaECTER) to raise stage-1 recall, a larger
cross-encoder backbone with a listwise loss, and more epochs — CPU was the binding
constraint, not data.
