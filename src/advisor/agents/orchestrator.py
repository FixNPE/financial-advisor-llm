"""Orchestrator — chains the planning pipeline for a customer + journey.

Sequence:
  1. Risk           risk_agent.run_risk_profiling(answers, age, income, dependents)
  2. Risk-narrate   risk_narrator.narrate_risk(customer, risk, answers)
  3. Goal           goal_agent.plan_<journey>(...)
  4. Portfolio      portfolio_agent.analyze_portfolio(holdings, allow_live=...)
  5. Benchmark      benchmark_agent.run_benchmarking(model_name)
  6. Recommend      recommend_agent.run_recommendation(model_name, allocation_pct)
  7. Rec-narrate    recommendation_narrator.narrate_recommendation(...)
  8. Report         report_agent.build_markdown_report(...)

Persists via ``log_agent_run`` (with summary JSON) and opens a Shape-B HITL
review row in ``open_hitl_review``. The UI ``commit_hitl_decision`` when
the user hits Approve / Reject / Override on the Recommendations page.

Rule-based fallback: all deterministic steps are pure math — no LLM required.
Two narrator steps layer grounded LLM calls on top of finished deterministic
outputs — one explains the risk band, one explains the recommendation. Both
fall back to deterministic templates when the LLM is off. The Report agent
applies output guardrails to the emitted markdown.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from advisor.agents.benchmark_agent import BenchmarkResult, run_benchmarking
from advisor.agents.goal_agent import (
    GoalPlan, plan_buy_home, plan_child_education, plan_retirement,
)
from advisor.agents.portfolio_agent import PortfolioAnalysis, analyze_portfolio
from advisor.agents.recommend_agent import (
    RecommendationBundle, run_recommendation,
)
from advisor.agents.recommendation_narrator import (
    RecommendationRationale, narrate_recommendation,
)
from advisor.agents.report_agent import build_markdown_report, compose_summary
from advisor.agents.risk_agent import run_risk_profiling
from advisor.agents.risk_narrator import RiskRationale, narrate_risk
from advisor.domain.data import (
    Customer, latest_committed_for_journey, log_agent_run, open_hitl_review,
)
from advisor.domain.risk import RiskResult
from advisor.llm.client import last_error as llm_last_error


AGENTS_RUN = ["risk", "risk_narrate", "goal", "portfolio", "benchmark",
                "recommend", "narrate", "report"]


@dataclass
class PipelineResult:
    customer_id: int
    journey: str
    risk: RiskResult
    risk_rationale: RiskRationale
    goal: GoalPlan
    portfolio: PortfolioAnalysis
    benchmark: BenchmarkResult
    recommendation: RecommendationBundle
    rationale: RecommendationRationale
    report_markdown: str
    summary: dict
    agent_run_id: int
    hitl_id: int
    prior_hitl: dict | None = None
    warnings: list[str] = field(default_factory=list)


def _run_goal(journey: str, customer: Customer, goal_inputs: dict, risk_band: str) -> GoalPlan:
    """Dispatch to the right journey-specific planner.

    goal_inputs shape (per journey):
      Retirement Planning:
        target_retirement_age (int), desired_monthly_income (float),
        current_savings (float), monthly_contribution (float)
      Child Education:
        child_current_age (int), target_cost_today (float),
        current_savings (float), monthly_contribution (float),
        start_college_age (int, default 18)
      Buy Home:
        home_price (float), down_payment_pct (float),
        target_purchase_year (int), current_year (int),
        current_savings (float), monthly_saving_capacity (float)
    """
    if journey == "Retirement Planning":
        return plan_retirement(
            current_age=customer.age,
            target_retirement_age=goal_inputs.get("target_retirement_age", 65),
            desired_monthly_income=goal_inputs.get(
                "desired_monthly_income", customer.annual_income / 12 * 0.8),
            current_savings=goal_inputs.get("current_savings", 0.0),
            monthly_contribution=goal_inputs.get("monthly_contribution", 500.0),
            risk_band=risk_band,
        )
    if journey == "Child Education":
        return plan_child_education(
            child_current_age=goal_inputs.get("child_current_age", 5),
            target_cost_today=goal_inputs.get("target_cost_today", 120_000.0),
            current_savings=goal_inputs.get("current_savings", 0.0),
            monthly_contribution=goal_inputs.get("monthly_contribution", 400.0),
            risk_band=risk_band,
            start_college_age=goal_inputs.get("start_college_age", 18),
        )
    if journey == "Buy Home":
        return plan_buy_home(
            home_price=goal_inputs.get("home_price", 500_000.0),
            down_payment_pct=goal_inputs.get("down_payment_pct", 20.0),
            target_purchase_year=goal_inputs.get("target_purchase_year", 2029),
            current_year=goal_inputs.get("current_year", 2026),
            current_savings=goal_inputs.get("current_savings", 0.0),
            monthly_saving_capacity=goal_inputs.get("monthly_saving_capacity", 1_000.0),
            risk_band=risk_band,
        )
    raise ValueError(f"Unknown journey: {journey!r}")


def run_pipeline(customer: Customer, journey: str, goal_inputs: dict,
                    *, allow_live_prices: bool = True) -> PipelineResult:
    """Full planning pipeline. Never raises for missing pieces — logs warnings.

    HITL: opens a row (committed_at NULL). The Recommendations page commits it
    on Approve/Reject/Override. If the user never commits, the row stays open
    and won't appear in ``latest_committed_for_journey``.
    """
    if customer.id is None:
        raise ValueError("Customer must be persisted (id required) before running pipeline")

    warnings: list[str] = []

    # 1. Risk
    answer_points_used = customer.risk_answers or [1, 1, 1, 1, 1]  # neutral fallback
    risk, model_name = run_risk_profiling(
        answer_points=answer_points_used,
        age=customer.age, annual_income=customer.annual_income,
        dependents=customer.dependents,
    )
    if not customer.risk_answers:
        warnings.append("No risk-questionnaire answers on file — using neutral defaults.")

    # 2. Risk-narrate — grounded LLM explanation of the deterministic band.
    # Never raises: falls back to a deterministic template if the LLM is off.
    risk_rationale = narrate_risk(customer, risk, answer_points_used)
    if risk_rationale.source != "llm":
        detail = f"source={risk_rationale.source}"
        if risk_rationale.source == "llm_error_fallback":
            reason = llm_last_error()
            if reason:
                detail += f", reason={reason}"
        warnings.append(
            f"Risk-band rationale used the deterministic template ({detail})."
        )

    # 3. Goal
    goal = _run_goal(journey, customer, goal_inputs, risk.risk_band)

    # 4. Portfolio
    portfolio = analyze_portfolio(customer.holdings, allow_live=allow_live_prices)
    if not portfolio.has_holdings:
        warnings.append("No holdings on file — recommendation uses model targets as the "
                        "starting point instead of drift-from-current.")
    # Any prices missing entirely (source=missing) get called out.
    missing = [t for t, src in portfolio.price_sources.items() if src == "missing"]
    if missing:
        warnings.append(f"No price data for: {', '.join(missing)} — using buy price as proxy.")

    # 5. Benchmark
    benchmark = run_benchmarking(model_name, allow_live=allow_live_prices)
    if benchmark.benchmark_source != "live_5y":
        warnings.append(
            f"Benchmark {benchmark.proxy_ticker} used illustrative long-run "
            f"return — live 5Y series unavailable."
        )

    # 6. Recommend
    current_pct = portfolio.allocation_pct or {}
    recommendation = run_recommendation(model_name, current_pct)

    # 7. Rec-narrate — grounded LLM rationale over the finished bundle.
    # Never raises: falls back to a deterministic template if the LLM is off.
    rationale = narrate_recommendation(customer, journey, risk, goal, portfolio,
                                          benchmark, recommendation)
    if rationale.source != "llm":
        detail = f"source={rationale.source}"
        if rationale.source == "llm_error_fallback":
            reason = llm_last_error()
            if reason:
                detail += f", reason={reason}"
        warnings.append(
            f"Recommendation rationale used the deterministic template ({detail})."
        )

    # 8. Summary + persist
    summary = compose_summary(customer, journey, goal, risk, model_name,
                                portfolio, benchmark, recommendation)
    summary["rationale"] = {"source": rationale.source, "provider": rationale.provider}
    summary["risk_rationale"] = {"source": risk_rationale.source,
                                    "provider": risk_rationale.provider}
    agent_run_id = log_agent_run(customer.id, journey, AGENTS_RUN, summary)
    hitl_id = open_hitl_review(customer.id, agent_run_id, journey, ai_suggested=model_name)

    # 9. Prior committed decision (informational — shown in Report if present)
    prior = latest_committed_for_journey(customer.id, journey)

    # 10. Report
    report_md = build_markdown_report(customer, journey, goal, risk, model_name,
                                      portfolio, benchmark, recommendation,
                                      hitl_decision=prior,
                                      rationale_markdown=rationale.markdown,
                                      risk_rationale_markdown=risk_rationale.markdown)

    return PipelineResult(
        customer_id=customer.id, journey=journey,
        risk=risk, risk_rationale=risk_rationale,
        goal=goal, portfolio=portfolio, benchmark=benchmark,
        recommendation=recommendation, rationale=rationale,
        report_markdown=report_md,
        summary=summary, agent_run_id=agent_run_id, hitl_id=hitl_id,
        prior_hitl=prior, warnings=warnings,
    )


__all__ = ["PipelineResult", "run_pipeline", "AGENTS_RUN"]
