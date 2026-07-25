"""Docked FinAdvisor chat — a right-side panel on every page.

Every page calls ``render_floating_chat(page_key, page_context)`` at the
bottom. When the user clicks the FAB, the panel opens on the right side of
the screen; the page content shifts left so both are visible at once. There
is a close button in the panel header. The chat is grounded in the current
page (customer, pipeline, highlights) and *drives changes*: when the user
says "raise my monthly contribution to $1,500" it proposes a structured
edit, the user confirms, we mutate the customer's profile, invalidate the
cached pipeline, and switch to the page most relevant to the change so they
can see the effect land.

Fix-positioning trick: Streamlit's ``key="foo"`` on any widget or container
adds a ``st-key-foo`` class to that widget's DOM wrapper (see
theme._CSS). We use fixed key names — ``nw_fab``, ``nw_chat_panel``,
``nw_chat_close`` — so a single CSS rule pins each in place.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from advisor.agents.advisor import answer_question
from advisor.agents.change_intent import ProposedChange, propose_change
from advisor.agents.intent import (
    OUT_OF_SCOPE, OUT_OF_SCOPE_MESSAGE, PLANNING_JOURNEYS, classify_intent,
)
from advisor.agents.recommend_agent import apply_model_override
from advisor.domain.data import Customer, upsert_customer

from app.components import goal_setup
from app.components.chat_ui import push_bot, push_user, render_history
from app.components.onboarding import missing_goal_input_fields
from app.components.session import (
    KEY_CHAT_HISTORY, KEY_LAST_PIPELINE,
    active_customer, profile_dict_for_prompt,
)

KEY_CHAT_PROMPT_QUEUE = "floating_chat_pending"  # str, dispatched on next open
KEY_CHAT_PENDING_CHANGE = "floating_chat_pending_change"  # ProposedChange awaiting confirmation
KEY_CHAT_PENDING_SETUP = "floating_chat_pending_setup"    # str — journey awaiting user consent
KEY_CHAT_OPEN = "floating_chat_open"             # bool — panel visible?


def _context_lines(page_key: str, page_context: dict[str, Any] | None) -> list[str]:
    """Compact human-readable summary of the page the user is on."""
    lines = [f"User is on the **{page_key}** page."]
    if not page_context:
        return lines
    for label, value in page_context.items():
        if value in (None, "", [], {}):
            continue
        lines.append(f"- {label}: {value}")
    return lines


def _build_page_facts(page_key: str,
                        page_context: dict[str, Any] | None) -> dict[str, Any]:
    """Build the PAGE FACTS dict fed into the ReAct system prompt.

    Includes the page name + every non-empty key/value from ``page_context``.
    These are the *exact* numbers on the user's screen — the LLM is
    instructed to prefer them over its own recomputation.
    """
    facts: dict[str, Any] = {"Page": page_key}
    if page_context:
        for k, v in page_context.items():
            if v in (None, "", [], {}):
                continue
            facts[k] = v
    return facts


def _commit_change(customer: Customer, change: ProposedChange) -> str:
    """Persist the change + invalidate cached pipeline. Returns page target."""
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


def _current_model(customer: Customer | None) -> str:
    if not customer:
        return ""
    prev = st.session_state.get(KEY_LAST_PIPELINE)
    if prev and prev.customer_id == customer.id:
        return prev.recommendation.active_model
    return ""


def _render_panel_body(page_key: str, page_context: dict[str, Any] | None,
                            customer: Customer | None) -> None:
    """Render the docked chat panel's inner content."""
    # Header row — title on the left; the Close button is rendered separately
    # via a keyed container (st-key-nw_chat_close) so CSS can absolute-position
    # it in the top-right of the panel.
    st.markdown(
        '<div class="nw-chat-header">'
        '<div class="nw-chat-title"><span class="dot"></span>'
        f'FinAdvisor · {page_key}</div></div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Grounded in this page. Ask a question, or tell me to *change* "
        "something (contribution, retirement age, home price, journey, model) "
        "— I'll confirm before it lands."
    )

    with st.expander("What I can see on this page", expanded=False):
        for line in _context_lines(page_key, page_context):
            st.markdown(line)

    # ---- Chat history ----
    history: list[dict[str, str]] = st.session_state.setdefault(KEY_CHAT_HISTORY, [])
    if not history and customer is not None:
        push_bot(
            history,
            f"Hi {customer.name}! Ask me anything about your plan, or tell me "
            "to change something (contribution, retirement age, journey, model).",
        )
        st.session_state[KEY_CHAT_HISTORY] = history
    render_history(history, scroll_key="nw_chat_scroll_fc")

    # ---- Pending-confirmation card ----
    pending_change: ProposedChange | None = st.session_state.get(KEY_CHAT_PENDING_CHANGE)
    if pending_change and customer is not None:
        with st.chat_message("assistant"):
            st.markdown(
                f"**Proposed change:** {pending_change.label}\n\n"
                f"*{pending_change.reason}*\n\n"
                "Apply this and re-run the plan?"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Apply", type="primary",
                                use_container_width=True, key="fc_apply"):
                    target = _commit_change(customer, pending_change)
                    push_bot(history,
                                f"Applied — {pending_change.label}. Re-running the plan…")
                    st.session_state[KEY_CHAT_HISTORY] = history
                    st.session_state.pop(KEY_CHAT_PENDING_CHANGE, None)
                    st.switch_page(target)
            with c2:
                if st.button("Discard", use_container_width=True,
                                key="fc_discard"):
                    push_bot(history, f"Discarded — {pending_change.label}.")
                    st.session_state[KEY_CHAT_HISTORY] = history
                    st.session_state.pop(KEY_CHAT_PENDING_CHANGE, None)
                    st.rerun()

    # ---- Pending goal-setup card ----
    # The floating chat can't host the multi-turn mini-onboarding itself
    # (limited surface + closes on page switch), so we redirect the user to
    # the FinAdvisor page with the target journey pre-loaded via
    # KEY_FA_PENDING_SETUP — the full-page chat then owns the flow.
    pending_setup_journey: str | None = st.session_state.get(KEY_CHAT_PENDING_SETUP)
    if pending_setup_journey and customer is not None:
        with st.chat_message("assistant"):
            st.markdown(
                f"You haven't set up a **{pending_setup_journey}** plan yet. "
                "Want me to walk you through it?"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"Set up {pending_setup_journey}", type="primary",
                                use_container_width=True, key="fc_setup_start"):
                    st.session_state["finadvisor_pending_setup"] = pending_setup_journey
                    st.session_state.pop(KEY_CHAT_PENDING_SETUP, None)
                    st.session_state[KEY_CHAT_OPEN] = False
                    st.switch_page("pages/1_FinAdvisor.py")
            with c2:
                if st.button("Not now", use_container_width=True,
                                key="fc_setup_decline"):
                    st.session_state.pop(KEY_CHAT_PENDING_SETUP, None)
                    push_bot(history, "No problem — ask me anything else.")
                    st.session_state[KEY_CHAT_HISTORY] = history
                    st.rerun()

    # ---- Input row ----
    pending = st.session_state.pop(KEY_CHAT_PROMPT_QUEUE, None)
    prompt = st.chat_input("Type your answer or ask a question…",
                            key="fc_chat_input") or pending

    if prompt:
        push_user(history, prompt)

        change: ProposedChange | None = None
        if customer is not None:
            change = propose_change(
                prompt,
                goal_inputs=customer.goal_inputs,
                current_journey=customer.primary_goal,
                current_model=_current_model(customer),
            )

        if change is not None:
            st.session_state[KEY_CHAT_PENDING_CHANGE] = change
            push_bot(history,
                        f"I can make that change: **{change.label}**.\n\n"
                        f"{change.reason} Confirm below to apply.")
            st.session_state[KEY_CHAT_HISTORY] = history
            st.rerun()

        intent = classify_intent(prompt)
        if intent.journey == OUT_OF_SCOPE:
            push_bot(history, OUT_OF_SCOPE_MESSAGE)
            st.session_state[KEY_CHAT_HISTORY] = history
            st.rerun()

        # Mismatch guard: user asked about a planning journey whose required
        # goal_inputs are missing. Redirect to the FinAdvisor page for
        # mini-onboarding.
        if (customer is not None
                and intent.journey in PLANNING_JOURNEYS
                and missing_goal_input_fields(intent.journey,
                                                customer.goal_inputs)):
            st.session_state[KEY_CHAT_PENDING_SETUP] = intent.journey
            push_bot(history,
                        f"You haven't set up a **{intent.journey}** plan yet — "
                        "want me to walk you through it? Confirm below.")
            st.session_state[KEY_CHAT_HISTORY] = history
            st.rerun()

        profile = profile_dict_for_prompt(customer)
        page_facts = _build_page_facts(page_key, page_context)
        with st.spinner("Thinking…"):
            answer = answer_question(prompt, profile=profile,
                                        page_facts=page_facts)
        push_bot(history, answer.answer_markdown)
        st.session_state[KEY_CHAT_HISTORY] = history
        st.rerun()

    # ---- Footer actions ----
    if st.button("Clear chat", use_container_width=True, key="fc_clear"):
        st.session_state[KEY_CHAT_HISTORY] = []
        st.session_state.pop(KEY_CHAT_PENDING_CHANGE, None)
        st.rerun()


def render_floating_chat(page_key: str,
                            page_context: dict[str, Any] | None = None,
                            customer: Customer | None = None) -> None:
    """FAB launcher + docked right-side chat panel.

    No-op on the FinAdvisor page (that page IS the chat). Otherwise:
    - closed state: renders a keyed button (``key="nw_fab"``) that CSS
      fix-positions bottom-right;
    - open state: renders a keyed container (``key="nw_chat_panel"``) that
      CSS fix-positions to the right side of the viewport, with a keyed
      close button (``key="nw_chat_close"``) in its top-right corner.
    """
    if page_key == "FinAdvisor":
        return
    if customer is None:
        customer = active_customer()

    is_open = st.session_state.get(KEY_CHAT_OPEN, False)

    if not is_open:
        # Keyed container so `.st-key-nw_fab` in CSS fixes it to bottom-right.
        with st.container(key="nw_fab"):
            if st.button("💬  Ask FinAdvisor", key="nw_fab_btn",
                            help=f"Open FinAdvisor with the current {page_key} "
                                    "context loaded."):
                st.session_state[KEY_CHAT_OPEN] = True
                st.rerun()
        return

    # Open state — keyed bordered container for the docked panel.
    with st.container(key="nw_chat_panel", border=True):
        # Close button in its own keyed container so CSS can absolute-position
        # it in the panel's top-right corner.
        with st.container(key="nw_chat_close"):
            if st.button("Close ✕", key="nw_chat_close_btn"):
                st.session_state[KEY_CHAT_OPEN] = False
                st.rerun()
        _render_panel_body(page_key, page_context, customer)
