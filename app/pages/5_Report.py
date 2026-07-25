"""Report — final markdown output, downloadable, embeds dashboard block."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from advisor.agents.orchestrator import run_pipeline  # noqa: E402
from advisor.agents.report_agent import build_markdown_report  # noqa: E402
from advisor.domain.data import latest_committed_for_journey  # noqa: E402

from app.components.dashboard import render  # noqa: E402
from app.components.floating_chat import render_floating_chat  # noqa: E402
from app.components.session import KEY_LAST_PIPELINE, valid_pipeline  # noqa: E402
from app.components.theme import BRAND_NAME, apply_theme  # noqa: E402

st.set_page_config(page_title=f"Report · {BRAND_NAME}",
                    page_icon=":page_facing_up:", layout="wide")

customer = apply_theme(page_key="Report")

st.markdown(f'<div class="nw-hero-title">{BRAND_NAME} Report</div>',
                unsafe_allow_html=True)

if not customer:
    st.info("Pick a customer from Home."); st.stop()

_saved_journey = customer.primary_goal or "Retirement Planning"
if _saved_journey == "Financial Q&A":
    journey = "Retirement Planning"
    st.caption(
        f"Showing a **sample Retirement Planning** report for "
        f"{customer.name}. Ask FinAdvisor to switch the journey."
    )
else:
    journey = _saved_journey

prev = st.session_state.get(KEY_LAST_PIPELINE)
if not valid_pipeline(prev, customer.id, journey):
    with st.spinner("Running pipeline..."):
        result = run_pipeline(customer, journey, customer.goal_inputs, allow_live_prices=True)
    st.session_state[KEY_LAST_PIPELINE] = result
else:
    result = prev

# Refresh the report markdown if a HITL decision was committed after the pipeline ran.
latest = latest_committed_for_journey(customer.id, journey)
if latest and latest != result.prior_hitl:
    result.report_markdown = build_markdown_report(
        customer, journey, result.goal, result.risk,
        result.recommendation.active_model,
        result.portfolio, result.benchmark, result.recommendation,
        hitl_decision=latest,
        rationale_markdown=(result.rationale.markdown if result.rationale else None),
        risk_rationale_markdown=(result.risk_rationale.markdown
                                    if result.risk_rationale else None),
    )

render(customer, result)

st.divider()

st.markdown("### Full report (markdown)")
st.markdown(result.report_markdown)

st.download_button(
    "Download report (.md)",
    data=result.report_markdown,
    file_name=f"nexwealth_{customer.external_id}_{journey.replace(' ', '_')}.md",
    mime="text/markdown",
    use_container_width=True,
)

_goal = result.goal
_gi = customer.goal_inputs or {}
render_floating_chat(
    page_key="Report",
    page_context={
        "Journey": journey,
        "Goal inputs (use verbatim)": (
            ", ".join(f"{k}={v}" for k, v in _gi.items()) if _gi else None
        ),
        "Active model": result.recommendation.active_model,
        "Risk band": f"{result.risk.risk_band} (score {result.risk.risk_score})",
        "Assumed annual return": f"{_goal.assumed_annual_return:.2%}",
        "Horizon (years)": _goal.years,
        "Target amount (future $)": f"${_goal.target_amount_future:,.0f}",
        "Projected amount at horizon": f"${_goal.projected_amount:,.0f}",
        "Funding ratio": f"{_goal.funding_ratio * 100:.0f}%",
        "Outlook": _goal.outlook,
        "Monte-Carlo p10 / p50 / p90": (
            f"${_goal.p10:,.0f} / ${_goal.p50:,.0f} / ${_goal.p90:,.0f}"
        ),
        "Funding gap": f"${_goal.funding_gap:,.0f}",
        "Success probability (illustrative)": f"{_goal.success_prob * 100:.1f}%",
        "Committed HITL decision": (latest.get("final_action") if latest else "none yet"),
        "Final choice": (latest.get("final_choice") if latest else None),
    },
    customer=customer,
)
