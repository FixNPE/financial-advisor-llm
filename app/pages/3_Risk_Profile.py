"""Risk Profile — read-only band summary + on-demand LLM rationale.

The 5-question questionnaire is gone (risk is now inferred from a single
onboarding question, mapped to answer points in
``app.components.onboarding``). Goal inputs are gone too — they're captured
during onboarding and edited via the FinAdvisor chat afterwards.

This page just displays the deterministic risk band that was already
computed for the active customer and lets them ask for a grounded LLM
explanation of it.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from advisor.agents.risk_narrator import narrate_risk  # noqa: E402
from advisor.domain.risk import QUESTIONNAIRE, compute_risk  # noqa: E402

from app.components.charts import risk_gauge  # noqa: E402
from app.components.floating_chat import render_floating_chat  # noqa: E402
from app.components.session import KEY_RISK_RATIONALE  # noqa: E402
from app.components.theme import BRAND_NAME, apply_theme  # noqa: E402

st.set_page_config(page_title=f"Risk Profile · {BRAND_NAME}",
                    page_icon=":scales:", layout="wide")

customer = apply_theme(page_key="Risk Profile")

st.markdown('<div class="nw-hero-title">Risk Profile</div>', unsafe_allow_html=True)

if not customer:
    st.info("Pick a customer from Home to see the risk profile.")
    st.stop()

journey = customer.primary_goal or "Retirement Planning"

# Compute from the answers stored during onboarding (or use a neutral
# default if the customer was seeded without any).
answers = (
    list(customer.risk_answers)
    if customer.risk_answers else [1] * len(QUESTIONNAIRE)
)
result = compute_risk(answers, customer.age, customer.annual_income,
                        customer.dependents)

gauge_col, summary_col = st.columns([2, 3])
with gauge_col:
    st.plotly_chart(risk_gauge(result.risk_score, result.risk_band),
                    use_container_width=True)
with summary_col:
    st.markdown(
        f"""
        <div class="nw-card">
          <div style="font-size:13px; color:#6B7280; letter-spacing:.06em;
                        text-transform:uppercase;">On file for</div>
          <div style="font-size:20px; font-weight:700; padding:2px 0 8px 0;">
            {customer.name}
          </div>
          <div style="font-size:15px;">
            Risk Score <b>{result.risk_score}/100</b> → <b>{result.risk_band}</b>
          </div>
          <div style="color:#4A5468; font-size:13px; padding-top:6px;">
            {result.description}
          </div>
          <div style="color:#6B7280; font-size:12px; padding-top:10px;">
            Active journey: <b>{journey}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='padding-top:12px; color:#4A5468;'>"
        f"Tolerance <b>{result.tolerance:.0f}/100</b> · "
        f"Capacity <b>{result.capacity:.0f}/100</b></div>",
        unsafe_allow_html=True,
    )

st.divider()

# --- Risk rationale (grounded LLM explanation of the band) ---------
_rationale_cache: dict = st.session_state.setdefault(KEY_RISK_RATIONALE, {})
_answers_key = (customer.id, tuple(answers))
_rationale = _rationale_cache.get(_answers_key)

with st.container(border=True):
    st.markdown("### Why this band")
    if _rationale is None:
        st.caption(
            f"Explain why **{customer.name}** landed in the **{result.risk_band}** band. "
            "The score is deterministic — this only adds context."
        )
        if st.button("Explain this band", use_container_width=True,
                        key="explain_band_btn"):
            with st.spinner("Grounded explanation from the narrator..."):
                _rationale = narrate_risk(customer, result, answers)
            _rationale_cache[_answers_key] = _rationale
            st.rerun()
    else:
        _src_hint = {
            "llm": "Grounded rationale from the LLM narrator.",
            "template": "Deterministic template rationale (LLM off).",
            "llm_error_fallback": "Deterministic template rationale (LLM unavailable — fell back).",
        }.get(_rationale.source, "Risk-band rationale")
        st.caption(_src_hint)
        st.markdown(_rationale.markdown)
        if st.button("Regenerate", key="regen_band_btn"):
            _rationale_cache.pop(_answers_key, None)
            st.rerun()

st.caption(
    "To change your risk answers or goal inputs, use the **FinAdvisor** chat — "
    "e.g. \"lower my target retirement age to 60\" or \"I'm okay with more volatility\"."
)

render_floating_chat(
    page_key="Risk Profile",
    page_context={
        "Journey": journey,
        "Risk band": f"{result.risk_band} (score {result.risk_score})",
        "Tolerance": f"{result.tolerance:.0f}/100",
        "Capacity": f"{result.capacity:.0f}/100",
    },
    customer=customer,
)
