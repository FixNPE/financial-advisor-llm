"""Home — pick an existing user or chat-onboard a new one.

Two modes toggle by ``KEY_USER_MODE`` in session state (``existing`` / ``new``):

  * ``existing``: search + selectbox of seeded customers, then Load.
  * ``new``: chat-driven onboarding — 8 sequential questions handled by
    ``app.components.onboarding``. On completion, we build a Customer,
    run the 8-step pipeline, and switch to the Dashboard.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from advisor.agents.orchestrator import run_pipeline  # noqa: E402
from advisor.domain.data import get_customer, list_customers  # noqa: E402

from app.components.chat_ui import (  # noqa: E402
    push_bot, push_user, render_context_panel, render_history,
)
from app.components.onboarding import (  # noqa: E402
    KEY_ONBOARD_HISTORY, commit_customer_and_run, context_lines,
    next_bot_prompt, onboarding_complete, onboarding_progress,
    onboarding_state, reset_onboarding, submit_answer,
)
from app.components.session import (  # noqa: E402
    KEY_LAST_PIPELINE, set_active_customer,
)
from app.components.theme import BRAND_NAME, apply_theme  # noqa: E402

st.set_page_config(page_title=f"Home · {BRAND_NAME}",
                    page_icon=":diamonds:", layout="wide")

customer = apply_theme(page_key="Home")

KEY_USER_MODE = "home_user_mode"  # "existing" | "new"
KEY_ONBOARD_COMMITTED_ID = "onboard_committed_customer_id"

if KEY_USER_MODE not in st.session_state:
    st.session_state[KEY_USER_MODE] = "existing"

# --- Header ---------------------------------------------------------
_mode = st.session_state[KEY_USER_MODE]
# Once the user picks a goal in onboarding, reflect it in the hero title.
# Before that, keep the header goal-agnostic so we don't imply a bias.
_onboard_state = st.session_state.get("onboard_state", {}) or {}
_chosen_goal = _onboard_state.get("goal_type") if _mode == "new" else None
_hero_journey = _chosen_goal or "Financial planning"
_hero_sub = (
    "FinAdvisor will ask you a few questions to build your plan"
    if _mode == "new" else
    "Pick a seeded customer to jump into their plan"
)
st.markdown(
    f'<div class="nw-hero-title">{_hero_journey} — '
    f'{"new user onboarding" if _mode == "new" else "existing user"}</div>',
    unsafe_allow_html=True,
)
st.markdown(f'<div class="nw-hero-sub">{_hero_sub}</div>',
                unsafe_allow_html=True)

# --- Mode toggle ---------------------------------------------------
st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
mode_cols = st.columns([1, 1, 6])
with mode_cols[0]:
    if st.button("New user", use_container_width=True,
                    type=("primary" if _mode == "new" else "secondary")):
        st.session_state[KEY_USER_MODE] = "new"
        st.rerun()
with mode_cols[1]:
    if st.button("Existing", use_container_width=True,
                    type=("primary" if _mode == "existing" else "secondary")):
        st.session_state[KEY_USER_MODE] = "existing"
        st.rerun()

st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

# --- Body -----------------------------------------------------------
main_col, ctx_col = st.columns([3, 2], gap="large")

# ==================================================================
# EXISTING user mode
# ==================================================================
if _mode == "existing":
    with main_col:
        customers = list_customers()
        if not customers:
            st.info("No customers loaded yet. Run `make seed` to load the hero customers.")
        else:
            label_by_id = {
                c.id: f"#{c.external_id} — {c.name} ({c.age}, "
                        f"${c.annual_income:,.0f}/yr"
                        + (f", {c.primary_goal}" if c.primary_goal else "")
                        + ")"
                for c in customers
            }
            query = st.text_input(
                "Search customers", placeholder="Search by name or ID…",
                label_visibility="collapsed",
            )
            filtered_ids = [
                cid for cid, lbl in label_by_id.items()
                if query.lower() in lbl.lower()
            ] or list(label_by_id.keys())
            default_index = (
                filtered_ids.index(customer.id)
                if customer and customer.id in filtered_ids else 0
            )
            chosen = st.selectbox(
                "Customer", options=filtered_ids,
                format_func=lambda i: label_by_id[i], index=default_index,
                label_visibility="collapsed",
            )
            if st.button("Load Customer", type="primary", use_container_width=True):
                set_active_customer(chosen)
                st.switch_page("pages/1_FinAdvisor.py")

    with ctx_col:
        loaded = get_customer(customer.id) if customer else None
        parameters: list[tuple[str, str]] = []
        if loaded:
            parameters = [
                ("Name", loaded.name),
                ("Age", str(loaded.age)),
                ("Income", f"${loaded.annual_income:,.0f}/yr"),
                ("Goal", loaded.primary_goal or "not set"),
            ]
        render_context_panel(
            title="SESSION CONTEXT",
            intake_label="Profile loaded" if loaded else "No profile loaded yet",
            intake_percent=100 if loaded else 0,
            parameters=parameters,
        )

# ==================================================================
# NEW user mode — chat-driven onboarding
# ==================================================================
else:
    history: list[dict[str, str]] = st.session_state.setdefault(
        KEY_ONBOARD_HISTORY, []
    )

    # Kick off with the first bot prompt if the history is empty.
    if not history:
        first = next_bot_prompt()
        if first is not None:
            push_bot(history, first)
            st.session_state[KEY_ONBOARD_HISTORY] = history

    with main_col:
        render_history(history)

        # ---- Completion path -----------------------------------
        if onboarding_complete():
            state = onboarding_state()
            journey = state.get("goal_type", "Retirement Planning")

            # Persist the new customer + run the pipeline exactly ONCE per
            # completed onboarding. commit_customer_and_run() is NOT idempotent
            # — every call inserts a new customer row — so we track the
            # committed id in its own session key. A stale KEY_LAST_PIPELINE
            # from a prior session must NOT short-circuit this: without the
            # commit, the completion card would render another customer's
            # numbers and journey.
            committed_id = st.session_state.get(KEY_ONBOARD_COMMITTED_ID)
            loaded = None
            if committed_id is None:
                with st.spinner(f"Running your {journey.lower()} analysis..."):
                    new_id = commit_customer_and_run()
                    st.session_state[KEY_ONBOARD_COMMITTED_ID] = new_id
                    loaded = get_customer(new_id)
                    if loaded:
                        result = run_pipeline(
                            loaded, journey,
                            loaded.goal_inputs, allow_live_prices=True,
                        )
                        st.session_state[KEY_LAST_PIPELINE] = result

            result = st.session_state.get(KEY_LAST_PIPELINE)
            if result is not None:
                if loaded is None:
                    loaded = get_customer(result.customer_id)
                goal = result.goal
                risk = result.risk
                journey_label = result.journey
                with st.chat_message("assistant"):
                    st.markdown(
                        f"Here is your **{risk.risk_band.lower()}-risk "
                        f"{journey_label.lower()} plan** based on your profile:"
                    )
                    m1, m2 = st.columns(2)
                    m1.metric("Corpus / target needed",
                                f"${goal.target_amount_future:,.0f}",
                                help="Inflation-adjusted target")
                    m2.metric("Required monthly SIP",
                                f"${goal.required_monthly_sip:,.0f}")
                    m3, m4 = st.columns(2)
                    m3.metric("Funding ratio",
                                f"{goal.funding_ratio * 100:.0f}%",
                                goal.outlook,
                                help="Projected ÷ Target — how funded the "
                                        "plan is. Outlook: On track / "
                                        "Uncertain / At risk.")
                    m4.metric("MC p10 → p90",
                                f"${goal.p10:,.0f} → ${goal.p90:,.0f}",
                                help="10th–90th percentile terminal wealth "
                                        f"from a 2000-path Monte-Carlo "
                                        f"simulation under {risk.risk_band} "
                                        "assumptions")

                    st.markdown("**Top recommendations for your portfolio:**")
                    for i, opt in enumerate(result.recommendation.options, 1):
                        st.markdown(
                            f"- Priority {i} — **{opt.model}** portfolio: "
                            f"expected return {opt.expected_return * 100:.1f}%, "
                            f"volatility {opt.volatility * 100:.1f}%, "
                            f"fit vs current {opt.fit_score:.0f}/100"
                        )

                    if st.button("Open the full plan on the Dashboard →",
                                    type="primary", use_container_width=True,
                                    key="onboard_go_dashboard"):
                        st.switch_page("pages/2_Dashboard.py")
                    if st.button("Start over",
                                    key="onboard_restart_after_done"):
                        reset_onboarding()
                        st.session_state.pop(KEY_LAST_PIPELINE, None)
                        st.session_state.pop(KEY_ONBOARD_COMMITTED_ID, None)
                        st.rerun()

        # ---- Active prompt -------------------------------------
        else:
            prompt = st.chat_input("Type your answer or ask a question…",
                                    key="onboard_input")
            if prompt:
                push_user(history, prompt)
                accepted, reply = submit_answer(prompt)
                push_bot(history, reply)
                st.session_state[KEY_ONBOARD_HISTORY] = history
                st.rerun()

            _r1, _r2 = st.columns([1, 3])
            with _r1:
                if st.button("Start over", key="onboard_restart"):
                    reset_onboarding()
                    st.session_state.pop(KEY_ONBOARD_COMMITTED_ID, None)
                    st.rerun()

    with ctx_col:
        step, total = onboarding_progress()
        pct = int((step - 1) / total * 100) if not onboarding_complete() else 100
        intake_label = (
            "Plan complete" if onboarding_complete()
            else f"Question {step} of {total}"
        )
        parameters = context_lines()
        render_context_panel(
            title="SESSION CONTEXT",
            intake_label=intake_label,
            intake_percent=pct,
            parameters=parameters,
        )
