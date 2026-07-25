"""Reusable dashboard block — used by 2_Dashboard.py and embedded in Report.

Renders: customer chips, risk band, goal projection, allocation donut,
current-vs-target bars, benchmark stats, HITL log.
"""
from __future__ import annotations

import streamlit as st

from advisor.agents.orchestrator import PipelineResult
from advisor.domain.data import Customer, get_hitl_log

from app.components.charts import (
    allocation_donut, current_vs_target_bars, project_series, projection_line,
)


def _fmt_usd(x: float) -> str: return f"${x:,.0f}"
def _fmt_pct(x: float) -> str: return f"{x:.1f}%"


def render(customer: Customer, result: PipelineResult) -> None:
    """Render the full dashboard block. Idempotent — safe to call from any page."""
    st.subheader(f"{customer.name} · {result.journey}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", customer.age)
    c2.metric("Income", _fmt_usd(customer.annual_income),
                help="Annual pre-tax income on file for this customer.")
    c3.metric(
        "Risk band", result.risk.risk_band, f"score {result.risk.risk_score}",
        help=(
            "AI-assigned risk band derived from a blend of **tolerance** (five-question "
            "questionnaire) and **capacity** (age, income, dependents). Bands: "
            "0–34 Moderate · 35–74 Growth · 75–100 Aggressive."
        ),
    )
    c4.metric(
        "Model", result.recommendation.active_model,
        "overridden" if result.recommendation.is_overridden else None,
        help=(
            "The active model portfolio driving the target allocation and rebalancing "
            "actions. Defaults to the AI-suggested band; changes to *overridden* once "
            "the reviewer picks a different model on the Recommendations page."
        ),
    )

    # Warnings from the pipeline
    if result.warnings:
        with st.expander("Pipeline warnings", expanded=False):
            for w in result.warnings:
                st.warning(w)

    st.markdown("### Goal projection")
    goal = result.goal
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(
        "Target (today $)", _fmt_usd(goal.target_amount_today),
        help="Goal amount stated in *today's* dollars — before applying inflation.",
    )
    g2.metric(
        f"Target (yr {int(goal.years)})",
        _fmt_usd(goal.target_amount_future),
        help=(
            "Same goal after inflation compounding to the target year. Retirement "
            "and Buy-Home use CPI ≈ 3%; Child Education uses CPI + 2% education premium."
        ),
    )
    g3.metric(
        "Projected", _fmt_usd(goal.projected_amount),
        _fmt_usd(goal.projected_amount - goal.target_amount_future),
        help=(
            "Deterministic year-end projection of the customer's savings — current "
            "balance compounded at the assumed annual return plus monthly contributions. "
            "Delta shown is *Projected − Target (yr N)*: positive means on track, negative "
            "means a funding gap."
        ),
    )
    g4.metric(
        "Funding ratio", _fmt_pct(goal.funding_ratio * 100),
        goal.outlook,
        help=(
            "Projected ÷ Target (year N). Distribution-free 'how funded is the "
            "plan?' number. Outlook band: On track (≥100% funded or ≥70% success "
            "probability), Uncertain (75–99% funded or ≥35% prob), At risk otherwise."
        ),
    )

    # Second row: honest range from Monte-Carlo simulation + the older
    # single-point success probability (kept for continuity but relabelled
    # 'illustrative').
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "MC p10 (downside)", _fmt_usd(goal.p10),
        help=(
            "10th percentile of terminal wealth from a 2000-path Monte-Carlo "
            "simulation. Roughly 'in 1 out of 10 scenarios you finish with less "
            "than this'."
        ),
    )
    m2.metric(
        "MC p50 (median)", _fmt_usd(goal.p50),
        help="Median terminal wealth across the Monte-Carlo paths.",
    )
    m3.metric(
        "MC p90 (upside)", _fmt_usd(goal.p90),
        help="90th percentile — best-case-ish; the *upside* to plan around.",
    )
    m4.metric(
        "Success prob (illustrative)", _fmt_pct(goal.success_prob * 100),
        help=(
            "P(terminal wealth ≥ target) under a log-normal approximation. "
            "Tends to swing hard around the target because it collapses the "
            "full distribution to one number — read alongside the funding "
            "ratio and MC band above."
        ),
    )

    xs, ys = project_series(
        current_savings=goal.current_savings,
        monthly_contribution=goal.planned_monthly_contribution,
        annual_return=goal.assumed_annual_return,
        years=int(goal.years),
    )
    st.plotly_chart(projection_line(xs, ys, goal.target_amount_future),
                    use_container_width=True)

    st.markdown("### Portfolio")
    port = result.portfolio
    if port.has_holdings:
        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Market value", _fmt_usd(port.total_market_value),
            help=(
                "Σ *units × current price* across all holdings. CASH is priced at "
                "$1.00 per unit. Prices come from live Alpha Vantage → cached CSV "
                "history → seed row, in that order."
            ),
        )
        p2.metric(
            "Gain/Loss", _fmt_usd(port.total_gain_loss),
            _fmt_pct(port.total_gain_loss_pct),
            help=(
                "Unrealised P/L = market value − cost basis. Cost basis is "
                "*units × buy price* per holding as recorded in the seed / onboarding data."
            ),
        )
        p3.metric(
            "# Holdings", len(port.by_holding),
            help="Number of distinct tickers on the customer's book.",
        )

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(allocation_donut(port.allocation_pct), use_container_width=True)
        with col_b:
            active_target = _active_target(result)
            st.plotly_chart(
                current_vs_target_bars(port.allocation_pct, active_target),
                use_container_width=True,
            )

        with st.expander("Holdings (with price freshness)", expanded=False):
            st.dataframe(
                [{
                    "Ticker": h.ticker, "Category": h.category,
                    "Units": h.units, "Buy $": h.buy_price, "Now $": h.current_price,
                    "Freshness": h.price_freshness,
                    "Market Value": h.market_value, "P/L": h.gain_loss,
                    "Asset Class": h.asset_class,
                } for h in port.by_holding],
                use_container_width=True,
            )
    else:
        st.info("No holdings on record. Recommendation uses model target allocation.")

    st.markdown("### Benchmarking")
    st.caption(
        "How the customer's target allocation stacks up against a public reference "
        "portfolio at the same risk band. Higher **excess return** means the target "
        "allocation is expected to outrun the reference on a like-for-like basis; "
        "negative means it lags."
    )
    bm = result.benchmark
    b1, b2 = st.columns([2, 3])
    with b1:
        # Real Streamlit container so we can nest a native clickable help
        # tooltip inside — st.caption(help=...) doesn't work inside raw HTML.
        with st.container(border=True):
            st.markdown(
                f'<div style="font-size:12px; color:#6B7280; letter-spacing:.06em; '
                f'text-transform:uppercase;">Proxy benchmark</div>'
                f'<div style="font-size:18px; font-weight:700; padding:4px 0 2px 0;">'
                f'{bm.proxy_ticker}</div>'
                f'<div style="font-size:14px; color:#3F4A63; padding-bottom:8px;">'
                f'{bm.proxy_name}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Why this proxy?", help=bm.rationale)
    with b2:
        r1, r2 = st.columns(2)
        r1.metric(
            "Portfolio return",
            _fmt_pct(bm.portfolio_expected_return * 100),
            help=(
                "Expected annual return of the customer's *target* allocation, "
                "computed as the weighted average of asset-class expected returns "
                "(Moderate/Growth/Aggressive model assumptions). Forward-looking, "
                "not realised."
            ),
        )
        _bench_help_src = (
            f"Realized CAGR of {bm.proxy_ticker} from Alpha Vantage weekly closes "
            f"(~{bm.benchmark_series_weeks} weekly observations over ~5Y, 24h cached)."
            if bm.benchmark_source == "live_5y"
            else f"Illustrative long-run return for {bm.proxy_ticker} — live series "
                 "unavailable, using the model's reference constant."
        )
        r2.metric(
            "Benchmark return",
            _fmt_pct(bm.benchmark_expected_return * 100),
            _fmt_pct(bm.excess_return * 100),
            help=(
                f"{_bench_help_src} Delta shown below is **excess return** = "
                "portfolio − benchmark. Positive = portfolio expected to beat "
                "the peer; negative = lagging."
            ),
        )
        # Caption on r2 so it sits directly under the Benchmark return card,
        # not spanning both columns.
        _src_caption = (
            f"*Live CAGR from Alpha Vantage · {bm.benchmark_series_weeks} weekly obs (~5Y)*"
            if bm.benchmark_source == "live_5y"
            else "*Illustrative reference return — live series unavailable*"
        )
        r2.caption(_src_caption)
    st.caption(f"**Underlying blend:** {bm.blend_description}")

    st.markdown("### HITL decision log")
    log = get_hitl_log(customer.id, only_committed=True)
    if not log:
        st.caption("No committed HITL decisions yet.")
    else:
        st.dataframe(
            [{
                "When": r["committed_at"], "Journey": r["journey"],
                "AI suggested": r["ai_suggested"], "Final": r["final_choice"],
                "Action": r["final_action"], "Rationale": r["rationale"] or "",
            } for r in log],
            use_container_width=True,
        )


def _active_target(result: PipelineResult) -> dict[str, float]:
    if result.recommendation.custom_allocation:
        return result.recommendation.custom_allocation
    for o in result.recommendation.options:
        if o.model == result.recommendation.active_model:
            return o.target_allocation_pct
    return {}
