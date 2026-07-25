"""Report / Output Agent — markdown report + dashboard summary.

Deterministic. LLM narration can be layered later; the report is legible
without it.

Guardrails.screen_output is applied to the emitted markdown.
"""
from __future__ import annotations

from datetime import datetime

from advisor.agents.benchmark_agent import BenchmarkResult
from advisor.agents.goal_agent import GoalPlan
from advisor.agents.portfolio_agent import PortfolioAnalysis
from advisor.agents.recommend_agent import RecommendationBundle
from advisor.domain.data import Customer
from advisor.domain.risk import RiskResult
from advisor.guardrails import screen_output


def _fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def compose_summary(customer: Customer, journey: str, goal: GoalPlan,
                     risk: RiskResult, model_name: str, analysis: PortfolioAnalysis,
                     benchmark: BenchmarkResult, rec: RecommendationBundle) -> dict:
    return {
        "customer": {"name": customer.name, "age": customer.age,
                       "income": customer.annual_income},
        "journey": journey,
        "risk": {"score": risk.risk_score, "band": risk.risk_band,
                   "tolerance": risk.tolerance, "capacity": risk.capacity},
        "goal": {
            "target_today": goal.target_amount_today,
            "target_future": goal.target_amount_future,
            "years": goal.years,
            "projected": goal.projected_amount,
            "gap": goal.funding_gap,
            "required_sip": goal.required_monthly_sip,
            "success_prob": goal.success_prob,
            "funding_ratio": goal.funding_ratio,
            "outlook": goal.outlook,
            "p10": goal.p10,
            "p50": goal.p50,
            "p90": goal.p90,
        },
        "portfolio": {
            "market_value": analysis.total_market_value,
            "gain_loss": analysis.total_gain_loss,
            "gain_loss_pct": analysis.total_gain_loss_pct,
            "allocation_pct": analysis.allocation_pct,
            "has_holdings": analysis.has_holdings,
        },
        "benchmark": {
            "proxy": benchmark.proxy_ticker,
            "portfolio_return": benchmark.portfolio_expected_return,
            "benchmark_return": benchmark.benchmark_expected_return,
            "excess": benchmark.excess_return,
            "source": benchmark.benchmark_source,
            "series_weeks": benchmark.benchmark_series_weeks,
        },
        "recommendation": {
            "ai_suggested": rec.ai_suggested,
            "active_model": rec.active_model,
            "is_overridden": rec.is_overridden,
            "override_note": rec.override_note,
        },
    }


def build_markdown_report(customer: Customer, journey: str, goal: GoalPlan,
                            risk: RiskResult, model_name: str,
                            analysis: PortfolioAnalysis, benchmark: BenchmarkResult,
                            rec: RecommendationBundle,
                            hitl_decision: dict | None = None,
                            rationale_markdown: str | None = None,
                            risk_rationale_markdown: str | None = None) -> str:
    """Emit the full report as markdown. Applies output guardrails."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append(f"# FinAdvisor Report — {customer.name}")
    lines.append(f"*Generated {now} · Journey: **{journey}** · "
                    f"Customer #{customer.id or '—'}*")
    lines.append("")

    lines.append("## Risk Profile")
    lines.append(f"- **Band:** {risk.risk_band}  ")
    lines.append(f"- **Score:** {risk.risk_score}  (tolerance {risk.tolerance}, capacity {risk.capacity})")
    lines.append(f"- {risk.description}")
    lines.append("")
    if risk_rationale_markdown:
        lines.append("### Why this band")
        lines.append(risk_rationale_markdown.strip())
        lines.append("")

    lines.append("## Goal Plan")
    lines.append(f"- **Target (today's dollars):** {_fmt_usd(goal.target_amount_today)}")
    lines.append(f"- **Target (inflated to year {goal.years:.0f}):** {_fmt_usd(goal.target_amount_future)}")
    lines.append(f"- **Projected corpus at horizon:** {_fmt_usd(goal.projected_amount)}")
    lines.append(f"- **Funding gap:** {_fmt_usd(goal.funding_gap)}")
    lines.append(f"- **Funding ratio:** {_fmt_pct(goal.funding_ratio * 100)} · **Outlook:** {goal.outlook}")
    lines.append(f"- **Monte-Carlo terminal wealth (p10 / p50 / p90):** "
                    f"{_fmt_usd(goal.p10)} / {_fmt_usd(goal.p50)} / {_fmt_usd(goal.p90)}")
    lines.append(f"- **Required monthly SIP to close gap:** {_fmt_usd(goal.required_monthly_sip)}")
    lines.append(f"- **Success probability (illustrative):** {_fmt_pct(goal.success_prob * 100)}")
    lines.append("")

    lines.append("## Current Portfolio")
    if analysis.has_holdings:
        lines.append(f"- **Market value:** {_fmt_usd(analysis.total_market_value)}  "
                        f"(cost basis {_fmt_usd(analysis.total_cost_basis)}, "
                        f"P/L {_fmt_usd(analysis.total_gain_loss)} / "
                        f"{_fmt_pct(analysis.total_gain_loss_pct)})")
        lines.append("- **Allocation vs. target:**")
        lines.append("")
        lines.append("| Asset Class | Current % | Target % |")
        lines.append("|---|---:|---:|")
        target = rec.custom_allocation or {}
        active_target = _target_lookup(rec)
        for ac, v in analysis.allocation_pct.items():
            t = active_target.get(ac, 0.0)
            lines.append(f"| {ac} | {v:.1f} | {t:.1f} |")
        lines.append("")
        lines.append("- **Holdings:**")
        lines.append("")
        lines.append("| Ticker | Units | Buy | Now | Freshness | Market Value | P/L |")
        lines.append("|---|---:|---:|---:|---|---:|---:|")
        for h in analysis.by_holding:
            lines.append(
                f"| {h.ticker} | {h.units:.2f} | ${h.buy_price:.2f} | ${h.current_price:.2f} "
                f"| {h.price_freshness} | {_fmt_usd(h.market_value)} | {_fmt_usd(h.gain_loss)} |"
            )
    else:
        lines.append("- No holdings on record. Plan uses model-target allocation as the "
                        "starting point.")
    lines.append("")

    lines.append("## Benchmarking")
    lines.append(f"- **Proxy:** {benchmark.proxy_name} (`{benchmark.proxy_ticker}`)")
    lines.append(f"- *Why this proxy:* {benchmark.rationale}")
    if benchmark.benchmark_source == "live_5y":
        _bench_src = (f"live 5Y CAGR from Alpha Vantage weekly closes "
                        f"({benchmark.benchmark_series_weeks} observations)")
    else:
        _bench_src = "illustrative long-run return (live series unavailable)"
    lines.append(f"- Portfolio expected return: {_fmt_pct(benchmark.portfolio_expected_return * 100)} · "
                    f"Benchmark: {_fmt_pct(benchmark.benchmark_expected_return * 100)} "
                    f"({_bench_src}) · "
                    f"Excess: {_fmt_pct(benchmark.excess_return * 100)}")
    lines.append(f"- Underlying blend: {benchmark.blend_description}")
    if benchmark.benchmark_source == "live_5y":
        lines.append("- *Note:* live benchmark return is close-only (excludes "
                     "dividend reinvestment), so total return is ~2pp higher in practice. "
                     "Directional comparison to the portfolio's expected return remains valid.")
    lines.append("")

    lines.append("## Recommendation")
    lines.append(f"- **AI-suggested band:** {rec.ai_suggested}")
    lines.append(f"- **Active model:** {rec.active_model}"
                    f"{'  *(overridden by user)*' if rec.is_overridden else ''}")
    if rec.override_note:
        lines.append(f"- **User rationale:** {rec.override_note}")
    lines.append("")
    lines.append("### Investment options presented")
    for o in rec.options:
        marker = "  ← AI-suggested" if o.is_ai_suggested else ""
        lines.append(f"- **{o.model}**{marker}  ·  expected {_fmt_pct(o.expected_return * 100)} "
                        f"return / {_fmt_pct(o.volatility * 100)} vol  ·  "
                        f"fit {o.fit_score:.0f}/100")
        if o.rebalancing_actions:
            actions = "; ".join(
                f"{a['action']} {a['asset_class']} by {a['delta_pct']:.0f}pp"
                for a in o.rebalancing_actions
            )
            lines.append(f"    - rebalancing: {actions}")
    lines.append("")

    if rationale_markdown:
        lines.append("### Recommendation rationale")
        lines.append(rationale_markdown.strip())
        lines.append("")

    lines.append("## Human-in-the-Loop Decision Log")
    if hitl_decision:
        lines.append(f"- **AI suggested:** {hitl_decision.get('ai_suggested')}")
        lines.append(f"- **Final choice:** {hitl_decision.get('final_choice')} "
                        f"({hitl_decision.get('final_action')})")
        if hitl_decision.get("rationale"):
            lines.append(f"- **Rationale:** {hitl_decision['rationale']}")
        if hitl_decision.get("override_json"):
            lines.append(f"- **Custom allocation:** `{hitl_decision['override_json']}`")
        lines.append(f"- **Committed:** {hitl_decision.get('committed_at')}")
    else:
        lines.append("- *(no committed HITL decision yet — decision recorded on Approve/Reject/Override)*")
    lines.append("")

    lines.append("---")
    return screen_output("\n".join(lines))


def _target_lookup(rec: RecommendationBundle) -> dict[str, float]:
    """Which target allocation should the Current-vs-Target table show?

    - If user overrode with a custom allocation, use it.
    - Else, use the active model's target allocation.
    """
    if rec.custom_allocation:
        return rec.custom_allocation
    for o in rec.options:
        if o.model == rec.active_model:
            return o.target_allocation_pct
    return {}
