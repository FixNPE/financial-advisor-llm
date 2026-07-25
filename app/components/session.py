"""Streamlit session helpers — active-customer, nav, provider label.

Only touches ``st.session_state``. Sidebar rendering lives in
``app.components.theme``.
"""
from __future__ import annotations

import streamlit as st

from advisor.config import settings
from advisor.domain.data import Customer, get_customer, list_customers

# Keys we own in st.session_state
KEY_ACTIVE_CUSTOMER = "active_customer_id"
KEY_LAST_PIPELINE = "last_pipeline_result"    # advisor.agents.orchestrator.PipelineResult
KEY_LAST_ADVISOR_TURN = "last_advisor_turn"   # advisor.agents.advisor.AdvisorAnswer
KEY_LAST_INTENT = "last_intent"
KEY_LAST_QUESTION = "last_question"
KEY_PENDING_QUESTION = "pending_question"     # question queued from Home → FinAdvisor
KEY_HITL_PENDING = "hitl_pending"             # {hitl_id, ai_suggested, journey}
KEY_ANSWERED = "risk_answers_temp"            # in-flight questionnaire answers
KEY_RISK_RATIONALE = "risk_rationale_cache"   # {(customer_id, answers_tuple): RiskRationale}

# Chat histories — declared here so `set_active_customer` can reset them
# without importing the chat modules (which would form an import cycle).
KEY_FA_HISTORY = "finadvisor_history"          # main FinAdvisor page chat
KEY_CHAT_HISTORY = "floating_chat_history"     # floating right-side panel


def customer_options() -> list[Customer]:
    return list_customers()


def active_customer() -> Customer | None:
    cid = st.session_state.get(KEY_ACTIVE_CUSTOMER)
    if cid is None:
        return None
    c = get_customer(cid)
    if c is None:
        st.session_state.pop(KEY_ACTIVE_CUSTOMER, None)
        return None
    return c


def set_active_customer(customer_id: int | None) -> None:
    if customer_id is None:
        st.session_state.pop(KEY_ACTIVE_CUSTOMER, None)
    else:
        st.session_state[KEY_ACTIVE_CUSTOMER] = customer_id
    for k in (KEY_LAST_PIPELINE, KEY_LAST_ADVISOR_TURN, KEY_HITL_PENDING,
                KEY_ANSWERED, KEY_RISK_RATIONALE,
                KEY_FA_HISTORY, KEY_CHAT_HISTORY):
        st.session_state.pop(k, None)


def provider_label() -> str:
    if settings.llm_provider == "none" or not settings.hf_token:
        return "LLM: rule-based fallback"
    return f"LLM: {settings.llm_provider} · {settings.llm_model_id.split('/')[-1]}"


# Fields added to GoalPlan in the funding-ratio / Monte-Carlo overhaul.
# A pipeline result cached in st.session_state before that change lacks
# them and would AttributeError when the UI reads them. Bumping this
# tuple whenever we add new required fields lets us invalidate stale
# cached results in one place instead of scattering ``hasattr`` checks
# across every page.
_GOAL_REQUIRED_ATTRS = ("funding_ratio", "p10", "p50", "p90", "outlook")


def valid_pipeline(prev, customer_id: int, journey: str) -> bool:
    """True if ``prev`` matches the active customer/journey AND has the
    current GoalPlan shape. Stale entries return False so the caller re-runs.
    """
    if prev is None:
        return False
    if prev.customer_id != customer_id or prev.journey != journey:
        return False
    goal = getattr(prev, "goal", None)
    if goal is None:
        return False
    return all(hasattr(goal, attr) for attr in _GOAL_REQUIRED_ATTRS)


def profile_dict_for_prompt(customer: Customer | None) -> dict | None:
    if not customer:
        return None
    holdings_summary = (
        ", ".join(f"{h.ticker}({h.units:.0f})" for h in customer.holdings)
        if customer.holdings else "none"
    )
    # Prefer the deterministic risk band from the last pipeline run; fall
    # back to "unspecified" so we never mis-label the goal as a tolerance.
    prev = st.session_state.get(KEY_LAST_PIPELINE)
    risk_tolerance = "unspecified"
    if prev and prev.customer_id == customer.id:
        risk_tolerance = prev.risk.risk_band.lower()
    return {
        "age": customer.age,
        "risk_tolerance": risk_tolerance,
        "income": customer.annual_income,
        "goals": [customer.primary_goal] if customer.primary_goal else [],
        "goal_inputs": customer.goal_inputs or {},
        "holdings": holdings_summary,
    }
