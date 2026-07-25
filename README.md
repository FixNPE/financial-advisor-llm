---
title: NexWealth AI
emoji: 💬
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# NexWealth AI — Personalized Financial Advisor using LLM

**Product:** *FinAdvisor* — the AI-native wealth-management experience.
Plan, review, and approve investment decisions right in the chat panel.

Capstone project: a personalized financial guidance assistant built on an open-weights LLM (Llama-3.3-70B-Instruct via Hugging Face Inference Providers / Groq), grounded with retrieval-augmented generation over financial-education sources and Alpha Vantage market data.

> **New here?** Setup is `make setup && make index && make seed && make run`. Read
> [ARCHITECTURE.md](ARCHITECTURE.md) for the layered walkthrough, flow diagrams, and
> repo structure.

## What it does

Three pillars wrapped in one chat experience:

1. **Conversational Q&A (ReAct)** — the advisor runs a Reason → Act → Observe loop
   over 12 tools (Alpha Vantage market data + deterministic financial calculators)
   plus retrieved RAG snippets from SEC/IRS/FINRA materials. Answers carry inline
   `[Source: …]` and `[Tool: …]` citations and a per-turn tool-call audit trail.
2. **Investment & portfolio analysis** — pulls quotes, sector data, and news sentiment from Alpha Vantage; falls back to a versioned CSV price history when the daily quota is spent, so the app never renders empty.
3. **Goal-based planning with grounded rationale** — retirement / child education /
   home-purchase projections via deterministic calculators. A 7th agent
   (`recommendation_narrator`) layers a grounded LLM paragraph on top of the
   finished bundle explaining *why this model fits this customer*. Three investment
   options per plan with Approve / Reject / Override human-in-the-loop review.

A **docked chat panel** is available on every page (except the full-page FinAdvisor
itself). Click the "💬 Ask FinAdvisor" FAB (bottom-right, always visible) to open a
right-side panel that reads the current page's context (customer, journey, active
model, projected vs target, etc.) and can *drive changes*: say *"raise my monthly
contribution to $1,500"* or *"plan for buying a home instead"*, confirm the
proposed change on a card, and the app persists it, invalidates the cached plan, and
switches to the most relevant page so you see the effect land.

Every advisory response carries a disclaimer and source/tool citations.

## Architecture

```
User → Streamlit UI ── Docked chat panel (FAB) ─┐
       │                                        │
       │                            change_intent (regex)
       │                                        │
       │                            propose ─→ confirm ─→ upsert_customer
       │                                        │       ─→ pop cache
       │                                        │       ─→ st.switch_page
       │                                        │
       ▼                                        ▼
Input Guardrails → Intent Classifier
       ├─ Financial Q&A ─→ ReAct Advisor
       │                     Retrieve (Chroma + BM25, RRF-fused)
       │                     Loop up to 6× : Reason → Act (tool) → Observe
       │                     12 tools: Alpha Vantage × 6 + calculators × 6
       │
       └─ Planning journey ─→ Orchestrator
             Risk → Goal → Portfolio → Benchmark → Recommend → Narrate → Report
             ↑ 6 deterministic steps                        ↑ grounded LLM (fallback: template)
       → Output Guardrails → Response
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full mermaid diagrams (7-step pipeline,
change-intent flow, ReAct Q&A loop, recommendation narrator, HITL commit, price
fallback) and per-file component reference.

Price data has three tiers:

1. **Alpha Vantage** (live) — first attempt; fresh rows appended to the CSV history.
2. **`data/prices/history.csv`** — daily-appended source-of-truth floor. Diffable in git.
3. **PDF seed row** (`2026-07-13`) — absolute last resort so no ticker ever renders empty.

Benchmarking also pulls from Alpha Vantage: `TIME_SERIES_WEEKLY` for each risk
band's peer ETF (AOM / AOR / AOA), 24h-cached, from which the app computes a
5Y realized CAGR + realized volatility. Falls back to a hardcoded illustrative
long-run return when the series is unavailable — the Dashboard and Report
badge which path served the number.

## Prerequisites

- **Python 3.12+** (`python3 --version` — 3.12 or 3.13 both work).
- **macOS or Linux** (Windows via WSL). Streamlit and Chroma work fine on Apple Silicon.
- **API keys** — free tier is fine for both:
  - **Hugging Face** token — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Groq is enabled by default under HF Inference Providers, so a single HF token routes chat completions to Groq.
  - **Alpha Vantage** key — [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key). Free tier: 25 requests/day, 5/min.
- **~2 GB free disk** for the local Chroma index (`data/chroma`) and price cache.
- **Network** for the first `make index` (downloads the sentence-transformers embedder,
  ~500 MB) and for live prices at demo time. After that the app can run fully offline
  by setting `LLM_PROVIDER=none` in `.env`.

## First-time setup (one-time, ~5 minutes)

```bash
# 0. Clone and enter the repo
git clone <your-fork-or-remote-url> financial-advisor-llm
cd financial-advisor-llm

# 1. Install dependencies into a venv
make setup                # creates .venv/, installs requirements.txt
source .venv/bin/activate # activates it for this shell

# 2. Configure secrets
cp .env.example .env
# Open .env in your editor and set at minimum:
#   HF_TOKEN=hf_...                                       (from huggingface.co)
#   ALPHA_VANTAGE_KEY=...                                 (from alphavantage.co)
#   LLM_PROVIDER=groq                                     (default — or "none" for rule-based only)
#   LLM_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct        (or Llama-3.1-8B for a smaller/faster demo)

# 3. Sanity-check with the offline test suite (no keys required)
make test                 # 40/40 should pass in <10s

# 4. Build the RAG index from corpus/ (Chroma + sentence-transformers)
make index                # first run downloads the embedder — ~1-2 min

# 5. Load the 5 hero customers into data/profile.db (idempotent — safe to re-run)
make seed
```

After this, `data/chroma/`, `data/profile.db`, and `.env` are in place. You don't
repeat these steps unless the corpus, venv, or seed data changes.

**Verify the setup is healthy:**

```bash
# Should print a path ending in .venv/bin/python
which python

# Should list 5 customers
PYTHONPATH=src .venv/bin/python -c "from advisor.domain.data import list_customers; \
  [print(c.name) for c in list_customers()]"

# Should print >0 chunks in Chroma
PYTHONPATH=src .venv/bin/python -c "from advisor.rag.store import get_or_create_collection; \
  print(get_or_create_collection().count(), 'chunks in Chroma')"
```

## Pre-run scripts (before you launch the app)

Run these in order. All are idempotent — safe to re-run.

### 1. Refresh daily prices (`make prices`)

**When:** at the start of each demo day (or once a day in general).
**What:** appends today's row per tracked ticker to `data/prices/history.csv`, using Alpha Vantage. Paced at 13 seconds between calls to respect the 5/min throttle.
**Budget:** ~21 tickers × 1 GLOBAL_QUOTE call each = 21 calls (fits under the 25/day free-tier ceiling with headroom for interactive use).

```bash
make prices
# equivalent to: python scripts/refresh_prices.py
```

Options:

```bash
# Refresh a subset only:
python scripts/refresh_prices.py --tickers AAPL MSFT VTI

# Force overwrite even if today's row already exists:
python scripts/refresh_prices.py --force
```

If AV is rate-limited or unreachable, the script logs the failure per ticker and continues — the app still runs off the previous day's rows.

### 2. Pre-warm the AV request cache (`make prewarm`) — *optional, demo day only*

**When:** immediately before a mentored session or live demo.
**What:** primes the on-disk request cache at `data/raw/av_cache.sqlite` for `get_quote` / `get_company_overview` / `get_news_sentiment` on the hero tickers (AAPL / MSFT / VOO by default). Also paced at 13 s.

```bash
make prewarm
```

Cache TTL is 60 min, so run this within an hour of demo start.

### 3. (One-time) Rebuild the RAG index (`make index`)

**When:** only after adding or updating documents under `corpus/`.
**What:** ingests `corpus/{sec_education, irs_pubs, finra_investor, glossary.md}` into Chroma using the embedder set by `EMBED_MODEL_ID` (default `BAAI/bge-small-en-v1.5`).

## Running the app

```bash
make run
# opens http://localhost:8501 automatically
```

Or the equivalent long form:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Stop with `Ctrl-C`. Streamlit hot-reloads on file save; hard-refresh the browser
(`Cmd/Ctrl-Shift-R`) if the CSS or session-state schema changes.

The sidebar shows the **active customer** (picker across seeded hero customers) and
the **LLM provider label** (`groq · Llama-3.3-70B-Instruct` or `OFF · rule-based
fallback`).

## Running via Docker

Production shape — multi-stage build, non-root runtime user, healthcheck, named
volume for HITL state / Chroma / price cache.

```bash
# 1. Copy .env.example to .env and fill in HF_TOKEN + ALPHA_VANTAGE_KEY.
cp .env.example .env

# 2. Build + start
docker compose up --build -d

# 3. Open http://localhost:8501
```

The image copies `data/seed/` in at build time (so seed customers ship with the
image) but keeps mutable state — `data/profile.db`, `data/chroma/`,
`data/prices/`, `data/processed/`, `data/raw/` — on the `finadvisor-data`
named volume, so the HITL log and RAG index survive `docker compose down`
and re-`up`.

**First-run indexing.** The image does not run `build_rag_index.py` at build
time (needs the corpus wired through Chroma at runtime paths). After the first
`up`, run it once against the running container:

```bash
docker compose exec finadvisor python scripts/build_rag_index.py
docker compose exec finadvisor python scripts/seed_customers.py
```

Subsequent restarts skip both — the volume persists them.

**Env vars.** Never bake secrets into the image; `.env` is on `.dockerignore`
and is passed at run time via `env_file:` in `docker-compose.yml`. To rotate
`HF_TOKEN` / `ALPHA_VANTAGE_KEY`, edit `.env` and `docker compose restart`.

**Logs, health, shell.**

```bash
docker compose logs -f finadvisor          # tail streamlit + agent logs
docker compose ps                           # STATUS shows (healthy) once healthcheck passes
docker compose exec finadvisor bash         # shell inside the container as the `app` user
docker compose exec finadvisor pytest -q    # run the offline test suite in-container
```

**Plain-`docker` equivalent** (no compose):

```bash
docker build -t nexwealth/finadvisor:latest .
docker run --rm -d --name finadvisor \
    -p 8501:8501 \
    --env-file .env \
    -v finadvisor-data:/app/data \
    nexwealth/finadvisor:latest
```

### Page flow

| Page | What happens |
|---|---|
| **Home** | Select a seeded customer or start a new-customer onboarding; quick-start prompts route into the chat. |
| **FinAdvisor** | Chat-style prompt with quick-start pills → intent classifier. Planning intents route into Risk Profile → Recommendations. Financial Q&A runs the **ReAct advisor** — retrieve, then loop over 12 tools until the LLM has enough to answer. Every response shows citations + a tool-call audit trail. |
| **Dashboard** | Full picture for the active customer: risk band, goal projection line + funding-ratio + MC p10/p50/p90, allocation donut, current-vs-target bars, benchmark card (proxy ETF + live 5Y CAGR with "Why this proxy?" tooltip), holdings table with price freshness, HITL log. |
| **Risk Profile** | Arc gauge score + five-question tolerance/capacity questionnaire → risk band. Journey-specific goal inputs (retirement age, tuition target, home price, etc.) saved to the customer. |
| **Recommendations** | Three investment options centered on the AI-suggested band, with a grounded **LLM rationale** rendered above the option cards (source-badge caption indicates `llm` / `template` / `llm_error_fallback`). HITL: Approve / Reject / Override (with a different model or a custom asset-class allocation). |
| **Report** | Dashboard block + full markdown report (Risk Profile, Goal Plan, Portfolio, Benchmarking, Recommendation, HITL Decision Log). Downloadable `.md`. |

Every planning page (Dashboard / Recommendations / Report) also renders the docked
FinAdvisor chat panel — see [Floating chat](#floating-chat--drive-changes-from-anywhere) below.

### Journeys

Four intents; the first three run the planning pipeline, the last runs the grounded Q&A path:

- **Retirement Planning** — 25× annual desired income; inflated to retirement year (CPI 3%).
- **Child Education** — target college cost inflated at CPI + 2% education premium.
- **Buy Home** — down-payment target; expected return capped at 4.5% for horizons ≤ 5 years (capital preservation).
- **Financial Q&A** — ReAct-driven answer over the corpus **plus** live tools. The LLM
  can request quotes, sector performance, news sentiment, or run a retirement / debt
  / emergency-fund calculation, chaining calls up to 6 rounds before answering. Every
  answer carries inline `[Source: …]` and `[Tool: …]` citations. Falls back to a
  deterministic snippet-echo when the LLM is off.

**Sample-lens journey handling.** A customer's `primary_goal` is a *sample lens*, not
a hard attribute. If a customer's saved goal is `"Financial Q&A"` (Emily Nguyen), the
planning pages (Dashboard / Recommendations / Report) render a **sample Retirement
Planning** view with a caption inviting the user to switch via the chat. Say *"plan
for buying a home"* in the panel; the chat proposes the change, you confirm, and the
pages re-render under the new lens. See
[ARCHITECTURE.md §7](ARCHITECTURE.md#7-sample-lens-journey-handling) for details.

### Floating chat — drive changes from anywhere

The docked chat panel is available on every page except the full-page FinAdvisor
itself.

- **FAB** — bottom-right "💬 Ask FinAdvisor" button, always sticky. Click to open.
- **Panel** — right-side docked drawer (440px wide) with a close button in the top-right
  corner. The page content shifts left so nothing is hidden.
- **Grounding** — the panel gets a `page_context` dict from each page (journey, active
  model, projected vs target, monthly contribution on file, etc.). Expand *"What I can
  see on this page"* to see exactly what the chat is seeing.
- **Q&A** — anything that isn't a change intent goes to the ReAct advisor. Tool
  calls and citations render in the same chat message as the answer.
- **Change intents** — a fixed vocabulary of edits is recognized by regex parser
  ([src/advisor/agents/change_intent.py](src/advisor/agents/change_intent.py)):
  - `raise my monthly contribution to $1,500` → mutates `goal_inputs.monthly_contribution`
  - `retire at 60` → mutates `goal_inputs.target_retirement_age`
  - `target a $900k home` → mutates `goal_inputs.home_price`
  - `budget $200k for college` → mutates `goal_inputs.target_cost_today`
  - `plan for buying a home` / `switch to retirement planning` → mutates `primary_goal`
  - `use the Aggressive model` → overrides the recommendation model

  Each intent renders a **confirmation card** — nothing lands until you click Apply.
  On Apply the change is persisted (`upsert_customer`), the cached pipeline is
  invalidated, and the app switches to the most relevant page (Dashboard for
  goal-input edits, Risk Profile for journey switches, Recommendations for model
  overrides). See [ARCHITECTURE.md §5.2](ARCHITECTURE.md#52-floating-chat--change-intent-flow-hitl-for-edits).

### HITL (human-in-the-loop) decision log

Every pipeline run inserts one row into `hitl_log` (SQLite, `data/profile.db`). While
the user is on the Recommendations page, the row is *open* (`committed_at IS NULL`)
— the AI suggestion is captured, but the final choice is not. The row is **committed**
when the user hits Approve, Reject, or Override, at which point the final choice,
action, rationale, and optional custom allocation are stamped in. The Report page
reads the latest committed row for the active journey and renders it as a **HITL
Decision Log** section — that's the audit trail the mentor asked to see.

### LLM provider toggle

Switch `LLM_PROVIDER=none` in `.env` (or leave `HF_TOKEN` blank) and re-launch to
demo the **deterministic rule-based floor** with no LLM calls. Everything renders —
projections, drift, rebalancing actions, benchmarks, a **deterministic template
rationale** on the Recommendations page (badged as `template` in the caption), and
snippet-echo Q&A (no tool calls, no ReAct loop). The disclaimer and citations are
still applied by the guardrail layer. Use this to prove the system is not
LLM-dependent for correctness, and to run offline.

## Project layout

```
financial-advisor-llm/
├── .streamlit/config.toml  # NexWealth AI theme (coral primary, navy sidebar)
├── src/advisor/
│   ├── config.py           # single Settings source (.env)
│   ├── guardrails.py       # input screening + output disclaimer/scrub
│   ├── llm/                # HF Inference Providers client + prompts
│   ├── rag/                # ingest, retrieve, store (Chroma + BM25 fusion)
│   ├── tools/              # alpha_vantage + calculators + registry
│   │                       # (12-tool OpenAI catalog for the ReAct loop)
│   ├── domain/             # prices, models, risk, calculators, recommend,
│   │                       # benchmark, data (SQLite: customers, holdings,
│   │                       # agent_runs, hitl_log)
│   ├── agents/             # risk / goal / portfolio / benchmark / recommend /
│   │                       # recommendation_narrator / report agents +
│   │                       # intent classifier + orchestrator +
│   │                       # advisor (ReAct Q&A) + change_intent
│   │                       # (regex parser powering the docked chat)
│   └── eval/               # FPB sentiment eval
├── app/
│   ├── streamlit_app.py    # entry — applies NexWealth AI theme, bounces to Home
│   ├── components/         # theme, session helpers, Plotly charts, dashboard
│   │                       # block, floating_chat (FAB + docked panel)
│   └── pages/              # 0_Home, 1_FinAdvisor, 2_Dashboard,
│                           # 3_Risk_Profile, 4_Recommendations, 5_Report
├── corpus/                 # SEC / IRS / FINRA / glossary / client-portfolio-ref
├── data/
│   ├── seed/customers.json # hero customers loaded by scripts/seed_customers.py
│   ├── prices/history.csv  # daily-appended price history (three-tier floor)
│   ├── raw/av_cache.sqlite         # Alpha Vantage quote cache (60min TTL)
│   ├── raw/av_series_cache.sqlite  # Alpha Vantage weekly-series cache (24h TTL)
│   ├── chroma/             # RAG vector store (gitignored)
│   └── profile.db          # customers, holdings, agent_runs, hitl_log (gitignored)
├── scripts/                # refresh_prices, prewarm_cache, seed_customers,
│                           # build_rag_index, run_eval
├── tests/                  # pytest suite (37 tests, offline)
├── README.md               # this file
└── ARCHITECTURE.md         # layered walkthrough with flow diagrams
```

Full annotated tree lives in
[ARCHITECTURE.md §3](ARCHITECTURE.md#3-repository-structure).

## Common commands

| Command | Purpose |
|---|---|
| `make setup` | Create `.venv` and install deps |
| `make test` | Run the pytest suite |
| `make index` | Build / rebuild the Chroma vector store from `corpus/` |
| `make prices` | Append today's prices to `data/prices/history.csv` (run once daily) |
| `make prewarm` | Warm the AV request cache (run before a live demo) |
| `make seed` | Load hero customers from `data/seed/customers.json` (idempotent) |
| `make run` | Launch Streamlit at `http://localhost:8501` |
| `make eval` | Run the Financial PhraseBank eval (50 samples) |
| `make clean` | Remove `data/chroma`, `data/raw`, `data/processed`, `data/profile.db` |

## Typical demo-day sequence

```bash
source .venv/bin/activate
make prices        # 1. append today's row per ticker (~4 min, paced)
make prewarm       # 2. warm the AV cache for hero tickers
make seed          # 3. (re)load the hero customers into data/profile.db (idempotent)
make run           # 4. launch the app
```

The seed step loads 5 hero customers from `data/seed/customers.json`:

1. **Sarah Chen** — 32, retirement planning (VTI + CASH). Balanced growth story.
2. **Michael Rodriguez** — 45, retirement planning, 100% AAPL (concentrated position
   — great for drift visualisation and the "Override with custom allocation" HITL flow).
3. **Priya Patel** — 29, child education (VFIAX + AGG + CASH). Long horizon
   compounding.
4. **David Kim** — 38, buying a home in 2029 (BND + CASH). Short-horizon capital-preservation cap in play.
5. **Emily Nguyen** — 26, saved as Financial Q&A but with a starter portfolio
   (VTI + VXUS + CASH) and retirement-style goal inputs. The Dashboard /
   Recommendations / Report pages render a **sample Retirement Planning** view for
   Emily; use the docked chat to switch her journey to Buy Home or Child Education
   to see the pipeline re-run under a different lens.

## Configuration

Settings are read from `.env` (see `.env.example`):

| Key | Purpose | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face access token | *required for LLM* |
| `ALPHA_VANTAGE_KEY` | AV API key (free tier: 25/day, 5/min) | *required for live prices* |
| `LLM_PROVIDER` | `groq` (default), or `none` for rule-based only | `groq` |
| `LLM_MODEL_ID` | Model routed via HF Inference Providers | `meta-llama/Llama-3.3-70B-Instruct` |
| `CHROMA_DIR` | Vector store location | `./data/chroma` |

`.env` is gitignored. Never commit it. `.env.example` contains placeholders only.

## Troubleshooting

- **"AV rate-limited" during `make prices`** — you've hit 25/day. Wait until UTC
  midnight, or run with `--tickers AAPL MSFT` to refresh a subset. The app falls
  back to the last CSV row automatically.
- **App renders `seed (2026-07-13)` next to prices** — Alpha Vantage was unreachable
  and no CSV history was written yet. Run `make prices` and reload.
- **`ModuleNotFoundError: advisor`** — the venv isn't activated, or you're not at the
  repo root. Run `source .venv/bin/activate` from `financial-advisor-llm/`.
- **Chroma throws on `make run`** — the index doesn't exist yet. Run `make index` first.
- **FinAdvisor FAB missing for a specific customer** — the FAB is rendered by
  [render_floating_chat](app/components/floating_chat.py) on every page except
  `1_FinAdvisor`. If it doesn't appear on Dashboard / Recommendations / Report,
  make sure the page finished rendering (no `st.stop()` before the
  `render_floating_chat(...)` call at the bottom of the page).
- **`ValueError: Unknown journey: 'Financial Q&A'`** — you're on an older code path;
  the current version falls back to a sample Retirement Planning view for
  Financial Q&A customers (Emily). Pull latest.
- **Chat panel opens at the bottom instead of the right** — you're missing the CSS
  rules keyed off `.st-key-nw_chat_panel` in
  [app/components/theme.py](app/components/theme.py). This is Streamlit's official
  auto-generated class for keyed widgets; if it's not being applied, hard-refresh
  the browser to bust the CSS cache.
- **HITL row shows "Not yet decided"** — a pipeline was run but the user never
  clicked Approve / Reject / Override on the Recommendations page. That's a valid
  intermediate state (`committed_at IS NULL`); re-visit Recommendations and pick
  a decision to commit the row.

## Disclaimers

This software is for educational use only and does not constitute regulated financial advice. Outputs may be incomplete, outdated, or wrong. Always consult a licensed advisor before acting.

## References

- Yang, Liu, Wang. *FinGPT: Open-Source Financial Large Language Models.* arXiv:2306.06031 (2023).
- Alpha Vantage API: https://www.alphavantage.co/documentation/
- Hugging Face Inference Providers: https://huggingface.co/docs/inference-providers/
