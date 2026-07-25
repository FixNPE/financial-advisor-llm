"""Dashboard — the full picture for the active customer.

If no pipeline has been run yet, runs one on the fly (rule-based; safe).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from advisor.agents.orchestrator import run_pipeline  # noqa: E402

from app.components.dashboard import render  # noqa: E402
from app.components.floating_chat import render_floating_chat  # noqa: E402
from app.components.session import KEY_LAST_PIPELINE, valid_pipeline  # noqa: E402
from app.components.theme import BRAND_NAME, apply_theme  # noqa: E402

st.set_page_config(page_title=f"Dashboard · {BRAND_NAME}",
                    page_icon=":bar_chart:", layout="wide")

customer = apply_theme(page_key="Dashboard")

st.markdown('<div class="nw-hero-title">Dashboard</div>', unsafe_allow_html=True)

if not customer:
    st.info("Pick a customer from Home to see the dashboard.")
    st.stop()

# The journey is a sample lens on the customer, not a hard attribute.
# Financial Q&A is a chat-only journey with no plan to build, so we fall
# back to Retirement Planning as the default sample and let the user swap
# via the chat ("plan for buying a home instead").
_saved_journey = customer.primary_goal or "Retirement Planning"
if _saved_journey == "Financial Q&A":
    journey = "Retirement Planning"
    st.caption(
        f"Showing a **sample Retirement Planning** dashboard for "
        f"{customer.name}. Ask FinAdvisor to *plan for buying a home* or "
        f"*plan for child education* to switch the journey."
    )
else:
    journey = _saved_journey

prev = st.session_state.get(KEY_LAST_PIPELINE)
if not valid_pipeline(prev, customer.id, journey):
    with st.spinner(f"Running planning pipeline for {customer.name} ({journey})..."):
        result = run_pipeline(customer, journey, customer.goal_inputs, allow_live_prices=True)
    st.session_state[KEY_LAST_PIPELINE] = result
else:
    result = prev

render(customer, result)

st.divider()
st.caption("*Rule-based projections. Live prices via three-tier fallback: "
             "Alpha Vantage → CSV history → seed.*")

_gi = customer.goal_inputs or {}
_goal = result.goal
_bench = result.benchmark
render_floating_chat(
    page_key="Dashboard",
    page_context={
        "Journey": journey,
        "Goal inputs (use verbatim)": (
            ", ".join(f"{k}={v}" for k, v in _gi.items()) if _gi else None
        ),
        "Active model": result.recommendation.active_model,
        "AI-suggested model": result.recommendation.ai_suggested,
        "Risk band": f"{result.risk.risk_band} (score {result.risk.risk_score})",
        "Assumed annual return": f"{_goal.assumed_annual_return:.2%}",
        "Model expected return": f"{_bench.portfolio_expected_return:.2%}",
        "Horizon (years)": _goal.years,
        "Target amount (today's $)": f"${_goal.target_amount_today:,.0f}",
        "Target amount (future $)": f"${_goal.target_amount_future:,.0f}",
        "Projected amount at horizon": f"${_goal.projected_amount:,.0f}",
        "Funding ratio": f"{_goal.funding_ratio * 100:.0f}%",
        "Outlook": _goal.outlook,
        "Monte-Carlo p10 / p50 / p90": (
            f"${_goal.p10:,.0f} / ${_goal.p50:,.0f} / ${_goal.p90:,.0f}"
        ),
        "Funding gap": f"${_goal.funding_gap:,.0f}",
        "Required monthly SIP to close gap": f"${_goal.required_monthly_sip:,.0f}",
        "Success probability (illustrative)": f"{_goal.success_prob * 100:.1f}%",
    },
    customer=customer,
)
