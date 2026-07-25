"""Mini-onboarding for an EXISTING customer adding a new goal.

When a loaded customer asks the chat a question about a goal that isn't set
up (e.g. their profile is Retirement, they ask about buying a home), the
chat can offer to run a short branch-only intake. This module holds the
state machine for that flow — a slimmer sibling of ``onboarding.py`` that
skips name/age/income/tail and only asks the goal-specific questions.

On completion, we:
  1. merge the new fields into ``customer.goal_inputs``
  2. set ``customer.primary_goal`` to the new journey
  3. persist the customer
  4. invalidate the cached pipeline so the next page-load reruns it
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from advisor.domain.data import Customer, upsert_customer

from app.components.onboarding import OnboardQ, branch_questions
from app.components.session import KEY_LAST_PIPELINE

KEY_GS_JOURNEY = "goalsetup_journey"      # str — journey being onboarded
KEY_GS_STATE = "goalsetup_state"          # dict[str, Any] — answers so far
KEY_GS_STEP = "goalsetup_step"            # int — index into branch_questions


def is_active() -> bool:
    return KEY_GS_JOURNEY in st.session_state


def active_journey() -> str | None:
    return st.session_state.get(KEY_GS_JOURNEY)


def start(journey: str) -> str | None:
    """Kick off mini-onboarding for ``journey``. Returns the first bot prompt."""
    st.session_state[KEY_GS_JOURNEY] = journey
    st.session_state[KEY_GS_STATE] = {}
    st.session_state[KEY_GS_STEP] = 0
    return _next_prompt()


def cancel() -> None:
    for k in (KEY_GS_JOURNEY, KEY_GS_STATE, KEY_GS_STEP):
        st.session_state.pop(k, None)


def _questions() -> list[OnboardQ]:
    journey = st.session_state.get(KEY_GS_JOURNEY, "")
    return branch_questions(journey)


def _next_prompt() -> str | None:
    step = st.session_state.get(KEY_GS_STEP, 0)
    qs = _questions()
    if step >= len(qs):
        return None
    return qs[step].prompt


def submit_answer(raw: str) -> tuple[bool, str, bool]:
    """Parse a user answer against the current branch question.

    Returns ``(accepted, message, complete)``. When ``complete`` is True,
    the caller should commit the collected fields via ``commit(customer)``.
    """
    step = st.session_state.get(KEY_GS_STEP, 0)
    qs = _questions()
    if step >= len(qs):
        return True, "", True
    q = qs[step]
    value = q.extract(raw)
    if value is None:
        return False, q.error, False
    st.session_state[KEY_GS_STATE][q.key] = value
    st.session_state[KEY_GS_STEP] = step + 1
    nxt = _next_prompt()
    if nxt is None:
        journey = st.session_state[KEY_GS_JOURNEY]
        return True, (
            f"Got it — I have what I need for your **{journey}** plan. "
            "Running the analysis now…"
        ), True
    return True, nxt, False


def commit(customer: Customer) -> None:
    """Merge the collected branch fields into the customer, persist, and
    invalidate the cached pipeline so the next render reruns it."""
    journey = st.session_state[KEY_GS_JOURNEY]
    collected: dict[str, Any] = dict(st.session_state.get(KEY_GS_STATE, {}))

    goal_inputs = dict(customer.goal_inputs or {})

    if journey == "Retirement Planning":
        monthly_income = customer.annual_income / 12
        goal_inputs.setdefault("current_savings", 50_000.0)
        goal_inputs.setdefault("monthly_contribution",
                                    max(monthly_income * 0.15, 1_000.0))
        goal_inputs.update({
            "target_retirement_age": int(collected["retire_age"]),
            "desired_monthly_income": float(collected["monthly_need"]),
        })
    elif journey == "Child Education":
        monthly_income = customer.annual_income / 12
        goal_inputs.setdefault("current_savings", 5_000.0)
        goal_inputs.setdefault("monthly_contribution",
                                    max(monthly_income * 0.10, 500.0))
        goal_inputs.update({
            "child_current_age": int(collected["child_current_age"]),
            "target_cost_today": float(collected["target_cost_today"]),
        })
    elif journey == "Buy Home":
        monthly_income = customer.annual_income / 12
        goal_inputs.setdefault("current_savings", 20_000.0)
        goal_inputs.setdefault("monthly_saving_capacity",
                                    max(monthly_income * 0.15, 1_000.0))
        goal_inputs.setdefault("current_year", 2026)
        goal_inputs.update({
            "home_price": float(collected["home_price"]),
            "down_payment_pct": float(collected["down_payment_pct"]),
            "target_purchase_year": int(collected["target_purchase_year"]),
        })

    customer.primary_goal = journey
    customer.goal_inputs = goal_inputs
    upsert_customer(customer)
    st.session_state.pop(KEY_LAST_PIPELINE, None)
    cancel()
