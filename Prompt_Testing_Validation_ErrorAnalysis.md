# FinAdvisor Testing, Validation & Error-Analysis Workstream


---

You are the QA and evaluation engineer for **FinAdvisor**, an agentic financial-guidance assistant built on an open-weights LLM with this architecture: intent classification → agent orchestrator → specialized agents (Risk Profiling, Market Intelligence, Portfolio Analysis, Goal Planning) → investment proposal engine → LLM answer generation grounded by hybrid RAG (BM25 + dense retrieval with RRF fusion over SEC/IRS/FINRA education corpus) → safety & guardrails layer (hallucination check, suitability check, PII masking, mandatory disclaimers) → response with citations. Live data comes from market-data APIs with a SQLite cache; deterministic tools handle SIP/401(k)/retirement/EMI/CAGR math.

Deliver the following three workstreams. Work through them in order, showing your reasoning and producing every artifact as a concrete, ready-to-use file or table — not a description of one.

## 1. Prompt testing with multi-scenario inputs and few-shot refinement

Build a test-prompt suite covering at minimum these input scenario classes, with 3–5 concrete test prompts each:

- Simple factual/glossary questions (expense ratio, Roth vs Traditional IRA)
- Personalized goal planning (retirement corpus, college savings) with full, partial, and missing user profiles
- Live-data questions (quotes, fundamentals, news sentiment) including symbols the API won't recognize
- Directive-advice traps the system must deflect ("Should I buy X?", "Guaranteed 20% return?")
- Ambiguous or multi-intent queries ("I'm 40 with $50k, diabetes, and want to retire early — what do I do?")
- Adversarial/safety inputs: prompt injection, PII in the query, requests for tax evasion, distressed users
- Out-of-domain queries (crypto tips, medical advice, homework)

For each scenario class: state the expected intent classification, expected agent routing, expected tool/RAG calls, and pass criteria. Then write **few-shot examples** to be embedded in the intent classifier and answer-generation prompts — include both positive exemplars and hard negatives — and show a before/after refinement of at least two system prompts using failures you anticipate, explaining why each added exemplar fixes the failure mode.

## 2. Validation of structured outputs, dashboard updates, and API responses

- Define JSON Schemas for every structured artifact the system emits: intent classification result, agent routing decision, gap-analysis object (life cover, savings rate, insurance gaps), recommendation list with rank/impact/rationale, and the final response envelope (answer, citations[], disclaimers[], confidence).
- Write validation checks (pseudocode or Python) for each schema, including semantic checks — numbers reconcile (weights sum to 100%, gap = need − have), citations resolve to retrieved chunks, every quantitative claim traces to a tool call or source, disclaimer present on any advice-adjacent answer.
- Specify dashboard-update validation: after a chat interaction mutates state (new risk profile, updated goal), assert that dashboard widgets (readiness %, gap cards, allocation bars) reflect the new state and remain internally consistent.
- Specify API-response validation: schema conformance for market-data payloads, staleness rules against cache TTL, rate-limit and timeout behavior, and golden-file contract tests for each endpoint used.
- Produce a test matrix (component × check × method × pass threshold) I can drop into a test plan.

## 3. Error analysis and improvement loop

- Define an error taxonomy: intent misclassification, wrong agent routing, retrieval miss (relevant chunk not in top-k), retrieval noise (irrelevant chunk cited), hallucinated number, stale market data, tool-math error, guardrail false positive/negative, formatting/schema failure.
- For each category: how to detect it (automatic checks and LLM-as-judge rubrics), which pipeline stage owns it, and the primary remediation lever — prompt edit, few-shot addition, retrieval parameter (chunk size, top-k, RRF weights, similarity threshold), or fallback logic.
- Recommend concrete starting thresholds and what evidence justifies moving them: retrieval similarity cutoff, minimum citation coverage per answer, confidence threshold below which the system falls back.
- Design fallback handling for: no relevant retrieval (answer from general knowledge with explicit uncertainty + no numbers), API failure (cached data with staleness label), low-confidence classification (clarifying question), and guardrail block (safe educational redirect).
- End with a run-book: a repeatable weekly loop — sample N interactions → tag against taxonomy → prioritize by frequency × severity → apply one remediation → re-run the suite from workstream 1 → record metric deltas (intent accuracy, retrieval hit-rate, citation coverage, hallucination rate, deflection correctness).

**Output format:** one organized document with sections 1–3, tables where specified, code blocks for schemas/checks, and every few-shot example fully written out. Assume I will paste your artifacts directly into the repo — make them complete and runnable, not illustrative sketches. Ask me at most 3 clarifying questions first only if something materially changes the artifacts; otherwise proceed with stated assumptions listed at the top.
