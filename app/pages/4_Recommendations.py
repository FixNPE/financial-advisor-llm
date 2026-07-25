"""Recommendations — 3 investment options + HITL Approve/Reject/Override.

Runs the full pipeline via advisor.agents.orchestrator.run_pipeline, which
opens a HITL row (committed_at NULL). The Approve/Reject/Override buttons
commit that row with the final choice, action, and rationale.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from advisor.agents.orchestrator import run_pipeline  # noqa: E402
from advisor.agents.recommend_agent import (  # noqa: E402
    apply_custom_allocation_override, apply_model_override,
)
from advisor.domain.data import commit_hitl_decision  # noqa: E402
from advisor.domain.models import ASSET_CLASSES  # noqa: E402

from app.components.floating_chat import render_floating_chat  # noqa: E402
from app.components.session import (  # noqa: E402
    KEY_HITL_PENDING, KEY_LAST_PIPELINE, valid_pipeline,
)
from app.components.theme import BRAND_NAME, apply_theme  # noqa: E402

st.set_page_config(page_title=f"Recommendations · {BRAND_NAME}",
                    page_icon=":dart:", layout="wide")

customer = apply_theme(page_key="Recommendations")

st.markdown('<div class="nw-hero-title">Investment Recommendations</div>',
                unsafe_allow_html=True)

if not customer:
    st.info("Pick a customer from Home."); st.stop()

_saved_journey = customer.primary_goal or "Retirement Planning"
if _saved_journey == "Financial Q&A":
    journey = "Retirement Planning"
    st.caption(
        f"Showing **sample Retirement Planning** recommendations for "
        f"{customer.name}. Ask FinAdvisor to switch the journey."
    )
else:
    journey = _saved_journey

prev = st.session_state.get(KEY_LAST_PIPELINE)
needs_recompute = (
    not valid_pipeline(prev, customer.id, journey)
    or st.button("Re-run pipeline", type="secondary")
)
if needs_recompute:
    with st.spinner("Running Risk → Goal → Portfolio → Benchmark → Recommend..."):
        result = run_pipeline(customer, journey, customer.goal_inputs, allow_live_prices=True)
    st.session_state[KEY_LAST_PIPELINE] = result
    st.session_state[KEY_HITL_PENDING] = {
        "hitl_id": result.hitl_id,
        "ai_suggested": result.recommendation.ai_suggested,
        "journey": journey,
    }
else:
    result = prev

rec = result.recommendation

st.markdown(f"### AI-suggested model: **{rec.ai_suggested}**")
if rec.is_overridden:
    st.warning(f"Overridden by user — active model is **{rec.active_model}**")

# Grounded LLM rationale (falls back to a deterministic template if the LLM is off).
_rationale = getattr(result, "rationale", None)
if _rationale and _rationale.markdown:
    with st.container(border=True):
        _src_hint = {
            "llm": "Grounded rationale from the LLM narrator.",
            "template": "Deterministic template rationale (LLM off).",
            "llm_error_fallback": "Deterministic template rationale (LLM unavailable — fell back).",
        }.get(_rationale.source, "Recommendation rationale")
        st.caption(_src_hint)
        st.markdown(_rationale.markdown)

_n_words = {1: "One", 2: "Two", 3: "Three"}.get(len(rec.options), str(len(rec.options)))
st.markdown(f"### {_n_words} investment option{'s' if len(rec.options) != 1 else ''}")
if len(rec.options) < 3:
    st.caption(
        f"Fewer than three shown — **{rec.ai_suggested}** sits at the "
        "edge of the risk-band scale (Moderate → Growth → Aggressive), "
        "so there's no neighbour on one side."
    )
cols = st.columns(len(rec.options))
for col, opt in zip(cols, rec.options):
    with col:
        badge = " (AI-suggested)" if opt.is_ai_suggested else ""
        st.markdown(f"#### {opt.model}{badge}")
        st.metric("Expected return", f"{opt.expected_return*100:.1f}%")
        st.metric("Volatility",       f"{opt.volatility*100:.1f}%")
        st.metric(
            "Fit vs current", f"{opt.fit_score:.0f}/100",
            help=(
                "How close this model's target allocation is to the customer's "
                "current holdings. Scale 0–100 — higher = fewer trades to get "
                "there. 100 means no rebalancing needed; 0 means every asset "
                "class has to move. Computed as 100 − ½·Σ|current% − target%|."
            ),
        )
        st.markdown("**Target allocation**")
        st.dataframe(
            [{"Asset class": ac, "%": v} for ac, v in opt.target_allocation_pct.items()],
            use_container_width=True, hide_index=True,
        )
        if opt.rebalancing_actions:
            st.markdown("**Rebalancing actions**")
            for a in opt.rebalancing_actions:
                arrow = "▲" if a["action"] == "add" else "▼"
                st.caption(f"{arrow} {a['action'].title()} {a['asset_class']} "
                             f"by {a['delta_pct']:.0f}pp")

st.divider()

st.markdown("### Human-in-the-Loop decision")
mode = st.radio(
    "Decision mode",
    ["Approve (accept AI suggestion)",
     "Override with a different model",
     "Override with a custom allocation",
     "Reject (do nothing yet)"],
    index=0,
)
rationale = st.text_area("Rationale (optional)", height=80,
                            placeholder="Why did you approve / override / reject?")

if mode.startswith("Override with a different"):
    chosen_model = st.selectbox("Pick model", [o.model for o in rec.options],
                                  index=[o.model for o in rec.options].index(rec.ai_suggested))
    if st.button("Commit override", type="primary", use_container_width=True):
        apply_model_override(rec, chosen_model, rationale)
        commit_hitl_decision(result.hitl_id, final_choice=chosen_model,
                                final_action="override", rationale=rationale)
        st.success(f"Committed override → {chosen_model}")
        st.switch_page("pages/5_Report.py")

elif mode.startswith("Override with a custom"):
    st.caption("Enter a custom asset-class allocation (must sum to 100).")
    active_target = next(
        (o.target_allocation_pct for o in rec.options if o.model == rec.active_model),
        {ac: 0.0 for ac in ASSET_CLASSES},
    )
    custom_pct = {}
    cols = st.columns(len(ASSET_CLASSES))
    for col, ac in zip(cols, ASSET_CLASSES):
        with col:
            custom_pct[ac] = st.number_input(
                ac, min_value=0.0, max_value=100.0,
                value=float(active_target.get(ac, 0.0)), step=1.0, key=f"cust_{ac}",
            )
    total = sum(custom_pct.values())
    st.caption(f"Total: **{total:.1f}%**")
    if st.button("Commit custom allocation", type="primary",
                  disabled=abs(total - 100) > 0.5, use_container_width=True):
        apply_custom_allocation_override(rec, custom_pct, rationale)
        commit_hitl_decision(result.hitl_id, final_choice="custom",
                                final_action="override", rationale=rationale,
                                override_allocation=custom_pct)
        st.success("Committed custom allocation")
        st.switch_page("pages/5_Report.py")

elif mode.startswith("Reject"):
    if st.button("Commit rejection", type="secondary", use_container_width=True):
        commit_hitl_decision(result.hitl_id, final_choice=rec.ai_suggested,
                                final_action="reject", rationale=rationale)
        st.success("Recorded rejection")
        st.switch_page("pages/5_Report.py")

else:  # Approve
    if st.button("Commit approval", type="primary", use_container_width=True):
        commit_hitl_decision(result.hitl_id, final_choice=rec.ai_suggested,
                                final_action="approve", rationale=rationale)
        st.success(f"Approved → {rec.ai_suggested}")
        st.switch_page("pages/5_Report.py")

_goal = result.goal
_option_summary = "; ".join(
    f"{o.model}: exp {o.expected_return*100:.1f}% / vol {o.volatility*100:.1f}% / "
    f"fit {o.fit_score:.0f}"
    for o in rec.options
)
_gi = customer.goal_inputs or {}
render_floating_chat(
    page_key="Recommendations",
    page_context={
        "Journey": journey,
        "Goal inputs (use verbatim)": (
            ", ".join(f"{k}={v}" for k, v in _gi.items()) if _gi else None
        ),
        "AI-suggested model": rec.ai_suggested,
        "Active model": rec.active_model,
        "Overridden": "yes" if rec.is_overridden else "no",
        "Risk band": f"{result.risk.risk_band} (score {result.risk.risk_score})",
        "Options shown": _option_summary,
        "Assumed annual return (active plan)": f"{_goal.assumed_annual_return:.2%}",
        "Horizon (years)": _goal.years,
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
