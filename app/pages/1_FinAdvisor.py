"""FinAdvisor — pure chat surface for the active customer.

Two-column layout: chat on the left (main), session context on the right.
Every user turn goes through:
  1. propose_change  → confirm-before-mutate card if it's a change intent
  2. classify_intent → refuse OUT_OF_SCOPE cleanly
  3. answer_question → ReAct loop grounded in profile + page_facts
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from advisor.agents.advisor import answer_question  # noqa: E402
from advisor.agents.change_intent import ProposedChange, propose_change  # noqa: E402
from advisor.agents.intent import (  # noqa: E402
    OUT_OF_SCOPE, OUT_OF_SCOPE_MESSAGE, classify_intent,
)
from advisor.agents.intent import PLANNING_JOURNEYS, QA_LABEL  # noqa: E402
from advisor.agents.orchestrator import run_pipeline  # noqa: E402
from advisor.agents.recommend_agent import apply_model_override  # noqa: E402
from advisor.domain.data import upsert_customer  # noqa: E402

from app.components import goal_setup  # noqa: E402
from app.components.chat_ui import (  # noqa: E402
    push_bot, push_user, render_context_panel, render_history,
)
from app.components.onboarding import missing_goal_input_fields  # noqa: E402
from app.components.session import (  # noqa: E402
    KEY_FA_HISTORY, KEY_LAST_PIPELINE,
    profile_dict_for_prompt, provider_label, valid_pipeline,
)
from app.components.theme import BRAND_NAME, apply_theme  # noqa: E402

st.set_page_config(page_title=f"FinAdvisor · {BRAND_NAME}",
                    page_icon=":speech_balloon:", layout="wide")

customer = apply_theme(page_key="FinAdvisor")

KEY_FA_PENDING_CHANGE = "finadvisor_pending_change"
KEY_FA_PENDING_SETUP = "finadvisor_pending_setup"   # str — journey awaiting user consent

_hero_journey = (customer.primary_goal if customer and customer.primary_goal
                    else "Financial planning")
st.markdown(
    f'<div class="nw-hero-title">{_hero_journey} — '
    f'{"existing user" if customer else "no customer loaded"}</div>',
    unsafe_allow_html=True,
)
if customer:
    st.markdown(
        f'<div class="nw-hero-sub">Chatting as <b>{customer.name}</b> · '
        f'{customer.age} · ${customer.annual_income:,.0f}/yr &nbsp;·&nbsp; '
        f'{provider_label()}</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Pick a customer from Home to enable planning.")
    st.stop()

st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

# Lazy pipeline run — the advisor's page_facts + PAGE FACTS block are
# populated from KEY_LAST_PIPELINE. Without this the first chat turn has
# no plan numbers to ground on and the ReAct loop falls back to generic RAG.
# Financial Q&A is chat-only — no plan to build, so skip the pipeline and
# let the ReAct advisor answer without a PAGE FACTS plan block.
_prev = st.session_state.get(KEY_LAST_PIPELINE)
_journey = customer.primary_goal
if (_journey and _journey != QA_LABEL and customer.goal_inputs
        and not valid_pipeline(_prev, customer.id, _journey)):
    with st.spinner(f"Loading {customer.name}'s {_journey.lower()} plan..."):
        st.session_state[KEY_LAST_PIPELINE] = run_pipeline(
            customer, _journey, customer.goal_inputs, allow_live_prices=True,
        )

# ---- Two-column layout ---------------------------------------------------
main_col, ctx_col = st.columns([3, 2], gap="large")

# ---- Change-commit helper (mirror of floating_chat._commit_change) ------
def _commit_change(change: ProposedChange) -> str:
    """Persist a proposed change, invalidate cached state, return redirect."""
    if change.kind == "goal_input":
        goal_inputs = dict(customer.goal_inputs or {})
        goal_inputs[change.field] = change.new_value
        customer.goal_inputs = goal_inputs
        upsert_customer(customer)
        st.session_state.pop(KEY_LAST_PIPELINE, None)
        return "pages/2_Dashboard.py"
    if change.kind == "journey":
        customer.primary_goal = change.new_value
        upsert_customer(customer)
        st.session_state.pop(KEY_LAST_PIPELINE, None)
        return "pages/3_Risk_Profile.py"
    if change.kind == "model":
        prev = st.session_state.get(KEY_LAST_PIPELINE)
        if prev and prev.customer_id == customer.id:
            apply_model_override(prev.recommendation, change.new_value,
                                    "Chat-driven override")
        return "pages/4_Recommendations.py"
    st.session_state.pop(KEY_LAST_PIPELINE, None)
    return "pages/2_Dashboard.py"


def _current_model() -> str:
    prev = st.session_state.get(KEY_LAST_PIPELINE)
    if prev and prev.customer_id == customer.id:
        return prev.recommendation.active_model
    return ""


def _current_page_facts() -> dict:
    """PAGE FACTS from the most-recent pipeline result, if any.

    Includes goal_inputs verbatim so the LLM can call plan_retirement /
    plan_education / plan_home with exact values instead of inventing them.
    """
    facts: dict = {"Page": "FinAdvisor",
                    "Customer": f"{customer.name} ({customer.age}, "
                                 f"${customer.annual_income:,.0f}/yr)"}
    if customer.primary_goal:
        facts["Journey"] = customer.primary_goal
    if customer.goal_inputs:
        # Verbatim so the LLM has exact numbers to plug into plan_* tools.
        facts["Goal inputs (use verbatim)"] = ", ".join(
            f"{k}={v}" for k, v in customer.goal_inputs.items()
        )
    prev = st.session_state.get(KEY_LAST_PIPELINE)
    # Only surface the plan block if the cached pipeline matches the current
    # customer AND has the current GoalPlan shape — stale objects from a
    # prior session lack funding_ratio/p10/etc. and would AttributeError.
    if (prev and prev.customer_id == customer.id
            and valid_pipeline(prev, customer.id, prev.journey)):
        _goal = prev.goal
        facts.update({
            "Risk band": f"{prev.risk.risk_band} (score {prev.risk.risk_score})",
            "Active model": prev.recommendation.active_model,
            "Assumed annual return": f"{_goal.assumed_annual_return:.2%}",
            "Horizon (years)": _goal.years,
            "Target amount (future $)": f"${_goal.target_amount_future:,.0f}",
            "Projected amount at horizon": f"${_goal.projected_amount:,.0f}",
            "Funding ratio": f"{_goal.funding_ratio * 100:.0f}%",
            "Outlook": _goal.outlook,
            "Monte-Carlo p10 / p50 / p90": (
                f"${_goal.p10:,.0f} / ${_goal.p50:,.0f} / ${_goal.p90:,.0f}"
            ),
            "Success probability (illustrative)": f"{_goal.success_prob * 100:.1f}%",
            "Required monthly SIP": f"${_goal.required_monthly_sip:,.0f}",
        })
    return facts


# ---- Main chat surface ---------------------------------------------------
with main_col:
    history: list[dict[str, str]] = st.session_state.setdefault(KEY_FA_HISTORY, [])

    if not history:
        push_bot(
            history,
            f"Hi {customer.name}! I can help with retirement, education, home-buying, "
            "or general financial questions. What would you like to look at?",
        )
        st.session_state[KEY_FA_HISTORY] = history

    render_history(history)

    # Pending-confirmation card ---------------------------------------
    pending: ProposedChange | None = st.session_state.get(KEY_FA_PENDING_CHANGE)
    if pending is not None:
        with st.chat_message("assistant"):
            st.markdown(
                f"**Proposed change:** {pending.label}\n\n*{pending.reason}*\n\n"
                "Apply this and re-run the plan?"
            )
            c1, c2 = st.columns(2)
            if c1.button("Apply", type="primary", use_container_width=True,
                            key="fa_apply"):
                target = _commit_change(pending)
                push_bot(history,
                            f"Applied — {pending.label}. Re-running the plan…")
                st.session_state[KEY_FA_HISTORY] = history
                st.session_state.pop(KEY_FA_PENDING_CHANGE, None)
                st.switch_page(target)
            if c2.button("Discard", use_container_width=True, key="fa_discard"):
                push_bot(history, f"Discarded — {pending.label}.")
                st.session_state[KEY_FA_HISTORY] = history
                st.session_state.pop(KEY_FA_PENDING_CHANGE, None)
                st.rerun()

    # Pending goal-setup card -----------------------------------------
    pending_setup: str | None = st.session_state.get(KEY_FA_PENDING_SETUP)
    if pending_setup is not None and not goal_setup.is_active():
        with st.chat_message("assistant"):
            st.markdown(
                f"You haven't set up a **{pending_setup}** plan yet. "
                "Want me to walk you through it?"
            )
            c1, c2 = st.columns(2)
            if c1.button(f"Set up {pending_setup}", type="primary",
                            use_container_width=True, key="fa_setup_start"):
                first = goal_setup.start(pending_setup)
                st.session_state.pop(KEY_FA_PENDING_SETUP, None)
                if first is not None:
                    push_bot(history, first)
                st.session_state[KEY_FA_HISTORY] = history
                st.rerun()
            if c2.button("Not now", use_container_width=True,
                            key="fa_setup_decline"):
                st.session_state.pop(KEY_FA_PENDING_SETUP, None)
                push_bot(history, "No problem — ask me anything else.")
                st.session_state[KEY_FA_HISTORY] = history
                st.rerun()

    # Input row --------------------------------------------------------
    prompt = st.chat_input("Type your answer or ask a question…", key="fa_input")
    if prompt:
        push_user(history, prompt)

        # Mini-onboarding takes priority over normal chat routing.
        if goal_setup.is_active():
            accepted, message, complete = goal_setup.submit_answer(prompt)
            push_bot(history, message)
            st.session_state[KEY_FA_HISTORY] = history
            if complete:
                goal_setup.commit(customer)
                st.switch_page("pages/2_Dashboard.py")
            st.rerun()

        change = propose_change(
            prompt,
            goal_inputs=customer.goal_inputs,
            current_journey=customer.primary_goal,
            current_model=_current_model(),
        )
        if change is not None:
            st.session_state[KEY_FA_PENDING_CHANGE] = change
            push_bot(history,
                        f"I can make that change: **{change.label}**.\n\n"
                        f"{change.reason} Confirm below to apply.")
            st.session_state[KEY_FA_HISTORY] = history
            st.rerun()

        intent = classify_intent(prompt)
        if intent.journey == OUT_OF_SCOPE:
            push_bot(history, OUT_OF_SCOPE_MESSAGE)
            st.session_state[KEY_FA_HISTORY] = history
            st.rerun()

        # Mismatch guard: user asked about a planning journey whose required
        # goal_inputs are missing (either because it's a different goal or
        # because the profile drifted). Offer mini-onboarding instead of
        # letting the ReAct advisor answer with wrong / defaulted fields.
        if (intent.journey in PLANNING_JOURNEYS
                and missing_goal_input_fields(intent.journey,
                                                customer.goal_inputs)):
            st.session_state[KEY_FA_PENDING_SETUP] = intent.journey
            push_bot(history,
                        f"You haven't set up a **{intent.journey}** plan yet — "
                        "want me to walk you through it? Confirm below.")
            st.session_state[KEY_FA_HISTORY] = history
            st.rerun()

        with st.spinner("Thinking…"):
            answer = answer_question(
                prompt,
                profile=profile_dict_for_prompt(customer),
                page_facts=_current_page_facts(),
            )
        push_bot(history, answer.answer_markdown)
        st.session_state[KEY_FA_HISTORY] = history
        st.rerun()

    if st.button("Clear chat", key="fa_clear"):
        st.session_state[KEY_FA_HISTORY] = []
        st.session_state.pop(KEY_FA_PENDING_CHANGE, None)
        st.rerun()

# ---- Session-context rail -----------------------------------------------
with ctx_col:
    prev = st.session_state.get(KEY_LAST_PIPELINE)
    parameters: list[tuple[str, str]] = [
        ("Name", customer.name),
        ("Age", str(customer.age)),
        ("Income", f"${customer.annual_income:,.0f}/yr"),
    ]
    intake_percent = 40
    intake_label = "Profile loaded"
    if prev and prev.customer_id == customer.id:
        parameters.extend([
            ("Risk", prev.risk.risk_band),
            ("Goal", customer.primary_goal or "Retirement"),
            ("Active model", prev.recommendation.active_model),
            ("Success prob", f"{prev.goal.success_prob * 100:.0f}%"),
        ])
        intake_percent = 100
        intake_label = "Plan complete"

    render_context_panel(
        title="SESSION CONTEXT",
        intake_label=intake_label,
        intake_percent=intake_percent,
        parameters=parameters,
    )
