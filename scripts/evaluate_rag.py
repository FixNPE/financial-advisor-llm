#!/usr/bin/env python3
"""RAGAS evaluation of the Financial Q&A RAG pipeline (retrieval + generation).

Runs a fixed set of corpus-grounded questions through the real
HybridRetriever + advisor.answer_question(), scores the result with RAGAS
(faithfulness, answer_relevancy, context_precision, context_recall), and
writes both a CSV of raw scores and a markdown summary.

Usage:
    python scripts/evaluate_rag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ragas 0.2.x's ragas.llms.base unconditionally imports
# langchain_community.chat_models.vertexai, which current langchain-community
# no longer ships (it moved to the standalone langchain-google-vertexai
# package). We don't use Vertex AI, so stub the symbol rather than pull in
# google-cloud-aiplatform just to satisfy an unused import.
import types  # noqa: E402
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class _UnusedChatVertexAI:  # pragma: no cover - never instantiated
    pass


_vertexai_stub.ChatVertexAI = _UnusedChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

import pandas as pd  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    context_precision, context_recall, faithfulness, answer_relevancy,
)

from advisor.agents.advisor import answer_question  # noqa: E402
from advisor.config import settings  # noqa: E402
from advisor.rag.retrieve import HybridRetriever  # noqa: E402

# Corpus-grounded eval set — question + a reference answer written from the
# actual corpus text (glossary.md / SEC guide / IRS pubs / FINRA scams doc),
# not from the model. This is what context_recall / context_precision score
# retrieval against.
EVAL_SET = [
    {
        "question": "What is dollar-cost averaging and how does it compare to lump-sum investing?",
        "reference": "Dollar-cost averaging (DCA) means investing a fixed dollar amount on a "
                      "regular schedule regardless of price. It smooths entry timing and removes "
                      "emotion, but typically underperforms lump-sum investing in rising markets "
                      "because money sits in cash longer.",
    },
    {
        "question": "What's the difference between a Traditional IRA and a Roth IRA?",
        "reference": "Traditional IRA contributions may be tax-deductible and withdrawals are "
                      "taxed in retirement, with Required Minimum Distributions starting at age 73. "
                      "Roth IRA contributions are post-tax, qualified withdrawals after age 59.5 "
                      "and the 5-year rule are tax-free, and there are no RMDs for the original "
                      "owner, though income phase-outs apply.",
    },
    {
        "question": "How does an expense ratio affect long-term investment returns?",
        "reference": "The expense ratio is the annual fee a fund charges as a percentage of "
                      "assets — a 0.10% ratio costs $10/year per $10,000 invested. Over 30 years, "
                      "a 1% gap in expense ratios can cut final wealth by more than 25% due to "
                      "compounding drag.",
    },
    {
        "question": "What is the wash-sale rule and how does it affect tax-loss harvesting?",
        "reference": "Tax-loss harvesting sells a losing position to realize a capital loss that "
                      "offsets capital gains or up to $3,000/year of ordinary income. The wash-sale "
                      "rule disallows that loss if you rebuy a substantially identical security "
                      "within 30 days.",
    },
    {
        "question": "What does an inverted yield curve historically signal?",
        "reference": "The yield curve plots interest rates against maturity for bonds of "
                      "equivalent credit quality. A normal curve slopes upward; an inverted curve, "
                      "where short-term rates exceed long-term rates, has historically preceded "
                      "US recessions.",
    },
    {
        "question": "What is the 2025 contribution limit for a 401(k) for someone under 50?",
        "reference": "The 2025 contribution limit for a 401(k) is $23,500 for someone under age 50.",
    },
    {
        "question": "How can I tell if an investment opportunity is a scam?",
        "reference": "FINRA's investor guidance highlights common red flags of investment "
                      "scams: promises of guaranteed high returns with little or no risk, "
                      "pressure to act immediately, unregistered or unlicensed sellers, and "
                      "overly complex or secretive strategies that are hard to verify.",
    },
    {
        "question": "What records do I need to keep for travel expense deductions?",
        "reference": "IRS Publication 463 explains that taxpayers must keep records substantiating "
                      "the amount, time, place, and business purpose of travel expenses in order "
                      "to claim them as deductions.",
    },
]


def build_llm() -> ChatOpenAI:
    """RAGAS needs an LLM judge. HF's Inference Providers router is OpenAI-
    compatible, so we point ChatOpenAI at it using the same HF_TOKEN and
    model already configured for the app (no separate provider/key needed).
    """
    return ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=settings.hf_token,
        model=f"{settings.llm_model_id}:{settings.llm_provider}",
        temperature=0,
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    """Reuse the app's own embedder so retrieval-quality metrics are judged
    in the same embedding space the app actually retrieves in."""
    return HuggingFaceEmbeddings(model_name=settings.embed_model_id)


def run() -> pd.DataFrame:
    retriever = HybridRetriever()
    rows = []
    for case in EVAL_SET:
        q = case["question"]
        snippets = retriever.search(q, k=6)
        contexts = [s["text"] for s in snippets]
        answer = answer_question(q, profile=None, k=6)
        rows.append({
            "question": q,
            "contexts": contexts,
            "answer": answer.answer_markdown,
            "ground_truth": case["reference"],
            "_stopped_reason": answer.stopped_reason,
            "_provider": answer.provider,
        })
        print(f"  ran: {q[:60]}  (stopped_reason={answer.stopped_reason})")

    ds = Dataset.from_list([
        {k: v for k, v in r.items() if not k.startswith("_")} for r in rows
    ])

    llm = LangchainLLMWrapper(build_llm())
    embeddings = LangchainEmbeddingsWrapper(build_embeddings())
    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    df["stopped_reason"] = [r["_stopped_reason"] for r in rows]
    df["provider"] = [r["_provider"] for r in rows]
    return df


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    metric_cols = ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    means = df[metric_cols].mean(numeric_only=True)

    lines = [
        "# RAG Evaluation — RAGAS",
        "",
        f"Model: `{settings.llm_model_id}` via `{settings.llm_provider}` "
        f"(same judge LLM as the app, routed through HF Inference Providers). "
        f"Embedder: `{settings.embed_model_id}`. n={len(df)} corpus-grounded questions.",
        "",
        "## Aggregate scores",
        "",
        "| Metric | Score | What it measures |",
        "|---|---|---|",
    ]
    descriptions = {
        "faithfulness": "Fraction of claims in the answer that are supported by the retrieved context (hallucination check).",
        "answer_relevancy": "How well the answer addresses the actual question (semantic similarity to a reverse-generated question).",
        "context_precision": "Of the retrieved chunks, how many were actually relevant (retrieval precision).",
        "context_recall": "Whether the retrieved chunks covered everything needed to produce the reference answer (retrieval recall).",
    }
    for c in metric_cols:
        lines.append(f"| {c} | {means[c]:.3f} | {descriptions.get(c, '')} |")

    lines += [
        "",
        "## Per-question results",
        "",
        "| # | Question | Faithfulness | Answer relevancy | Context precision | Context recall |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in df.iterrows():
        q = row["user_input"][:70].replace("|", "/")
        vals = [f"{row[c]:.2f}" if c in row and pd.notna(row[c]) else "—" for c in metric_cols]
        lines.append(f"| {i+1} | {q} | " + " | ".join(vals) + " |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(f"Running RAGAS eval — provider={settings.llm_provider}, model={settings.llm_model_id}")
    df = run()
    csv_path = ROOT / "rag_eval_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    write_markdown(df, ROOT / "RAG_EVALUATION.md")
    print("Wrote RAG_EVALUATION.md")
