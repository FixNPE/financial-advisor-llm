# RAG Explorer & RAG Quality Evaluation

## 1. What RAG Explorer is

`app/pages/6_RAG_Explorer.py` is an in-app debugging/inspection page for the
Chroma vector store that backs Financial Q&A retrieval — the same collection
`HybridRetriever` (`src/advisor/rag/retrieve.py`) queries at answer time.
It has three panels:

1. **Overview** — chunk count, embedding model (`BAAI/bge-small-en-v1.5`),
   and the on-disk Chroma path. Confirms the index is actually built before
   you go looking for a bug elsewhere.
2. **Browse chunks** — a filterable table (by source document, by text
   substring) over every indexed chunk, so you can confirm a given fact
   from `corpus/` actually made it into the index, in one piece, with the
   metadata (`source`, `chunk_index`) the retriever relies on for citations.
3. **Inspect a chunk's embedding** — picks one chunk and shows its full
   384-dimension vector (as a bar chart + raw values), its L2 norm, and its
   metadata. Useful for sanity-checking that embeddings aren't degenerate
   (e.g. all-zero) after a re-index.
4. **Semantic search test** — runs a live query through the same
   `collection.query()` call path the app uses, and shows the nearest
   chunks with their vector distance. This is the fastest way to answer
   "why didn't the advisor cite the right source" without going through
   the full ReAct loop — you can see exactly what retrieval returned before
   the LLM ever sees it.

In short: RAG Explorer isolates the **retrieval** half of the RAG pipeline
from the **generation** half, so retrieval problems and generation problems
don't get diagnosed as the same thing.

## 2. RAGAS evaluation

`scripts/evaluate_rag.py` runs a fixed, corpus-grounded question set through
the **real** pipeline — `HybridRetriever.search()` for retrieval and
`advisor.agents.advisor.answer_question()` for generation (the exact code
path the app uses, ReAct loop included) — then scores the result with
[RAGAS](https://docs.ragas.io/) using the same LLM the app already runs
(`meta-llama/Llama-3.3-70B-Instruct` via Groq) as the judge, and the app's
own embedder (`BAAI/bge-small-en-v1.5`) for the embedding-based metric.

Reproduce with:

```bash
python scripts/evaluate_rag.py
```

This writes `rag_eval_results.csv` (raw per-question scores + generated
answers) and regenerates this file's results section.

### Methodology notes

- **n = 8 questions**, hand-written against specific facts in `corpus/`
  (glossary.md, IRS Publication 463, the FINRA scams guide), each paired
  with a reference answer written from the source text — not from the
  model. This is what `context_recall` / `context_precision` score
  retrieval against.
- 8 questions is enough to see a real pattern (below), not enough to treat
  any single decimal as statistically precise — treat this as a
  qualitative diagnostic, re-run with a larger set before citing these
  exact numbers as a KPI.
- The four RAGAS metrics map directly to the guideline's ask:
  **retrieval accuracy** → `context_precision` + `context_recall`;
  **faithfulness** → `faithfulness`; **answer relevance** →
  `answer_relevancy`.

### Aggregate scores

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | **0.616** | Fraction of claims in the answer supported by the retrieved context (hallucination check) |
| Answer relevancy | **0.969** | How well the answer addresses the actual question asked |
| Context precision | **0.889** | Of the retrieved chunks, how many were actually relevant |
| Context recall | **1.000** | Whether retrieval surfaced everything needed to answer |

### Per-question results

| # | Question | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|
| 1 | What is dollar-cost averaging and how does it compare to lump-sum investing? | 0.43 | 0.97 | 1.00 | 1.00 |
| 2 | What's the difference between a Traditional IRA and a Roth IRA? | 0.76 | 0.99 | 0.50 | 1.00 |
| 3 | How does an expense ratio affect long-term investment returns? | 1.00 | 0.97 | 1.00 | 1.00 |
| 4 | What is the wash-sale rule and how does it affect tax-loss harvesting? | 0.60 | 1.00 | 1.00 | 1.00 |
| 5 | What does an inverted yield curve historically signal? | 0.50 | 0.93 | 1.00 | 1.00 |
| 6 | What is the 2025 contribution limit for a 401(k) for someone under 50? | 0.25 | 1.00 | 1.00 | 1.00 |
| 7 | How can I tell if an investment opportunity is a scam? | 0.79 | 0.99 | 0.80 | 1.00 |
| 8 | What records do I need to keep for travel expense deductions? | 0.60 | 0.90 | 0.80 | 1.00 |

## 3. Analysis

**Retrieval is strong.** Context recall is a perfect 1.000 across all 8
questions — the retriever never missed a chunk it needed. Context precision
(0.889) is high, with the two soft spots (#2 IRA question, 0.50; #7/#8,
0.80) explained by close-topic chunks competing for the same top-k slot
(e.g. Traditional vs. Roth IRA content sits in adjacent glossary entries,
so both get retrieved even when only one is strictly needed for a given
sub-question). This validates the hybrid Chroma+BM25/RRF retrieval design —
retrieval is not the bottleneck.

**Answer relevancy is strong.** 0.969 average — the ReAct advisor
consistently answers the question actually asked, not a nearby one.

**Faithfulness is the real finding here (0.616), and it's not simply "the
model hallucinates."** Reading the raw generated answers
(`rag_eval_results.csv`) shows two specific, fixable causes:

1. **The mandatory disclaimer counts against the score.** Every advisor
   answer ends with guardrail-injected text
   ("*This is educational information, not personalized investment
   advice...*" — `src/advisor/guardrails.py`). RAGAS's faithfulness metric
   checks whether every claim in the answer is backed by the retrieved
   context; the disclaimer is policy text, not a retrieved fact, so it
   always scores as "unsupported." On short, factual answers this has an
   outsized effect — question #6 ("2025 401(k) limit") answers with the
   *exact correct number* cited to `glossary.md`, plus the disclaimer, and
   still scores 0.25 because the disclaimer sentence is 1 of only ~2-4
   total claims in a short answer. **This is a metric artifact, not a
   quality problem** — but it means faithfulness should be read net of the
   fixed disclaimer sentence, or the eval harness should be extended to
   strip it before scoring.
2. **The model adds reasonable elaboration beyond the literal retrieved
   sentence.** Question #1 (dollar-cost averaging) scores 0.43 because the
   answer expands "smooths entry timing, removes emotion" (glossary.md)
   into explanatory framing like "helps reduce the impact of market
   volatility" and "can be risky if the market declines" — true, standard
   financial reasoning, but not a verbatim claim in the retrieved chunk, so
   the strict judge flags it as unsupported. Question #5 (yield curve)
   shows the same pattern plus a generated "Caveats & Next Steps" section
   that is house style, not retrieved content.

**Net read:** retrieval and relevance are not the risk here — generation
*style* is. The ReAct advisor is grounded in the right sources but writes
with advisor-style elaboration and a fixed compliance disclaimer, both of
which a strict faithfulness judge penalizes even when the underlying facts
are correct. For a compliance-sensitive product this is arguably the
*safer* failure mode (extra caveats and disclaimers, not fabricated facts),
but it does mean the raw faithfulness number understates actual factual
accuracy and shouldn't be quoted without this context.

## 4. Recommendations

- Re-run the harness excluding the fixed disclaimer sentence from the
  faithfulness input, to separate "the model added the mandatory
  disclaimer" from "the model asserted something ungrounded."
- Expand the eval set past n=8 (ideally 25-50, covering all four corpus
  sources plus the two PDF-only ones) before treating any of these numbers
  as a tracked metric across app changes.
- Add a lightweight regression gate: re-run `scripts/evaluate_rag.py` in CI
  whenever `corpus/`, the retriever, or the system prompt changes, and flag
  if faithfulness or context recall drops below their current baseline.
