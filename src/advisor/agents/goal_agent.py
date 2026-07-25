"""Goal Planning Agent — three journey-specific planners.

Each returns a GoalPlan dataclass with projected value, required SIP,
funding gap, and success probability. Pure math; no LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

from advisor.domain.calculators import (
    EDUCATION_INFLATION_PREMIUM, US_INFLATION, funding_ratio, inflate,
    monte_carlo_terminal_wealth, outlook_band, project_portfolio,
    required_monthly_sip, success_probability,
)
from advisor.domain.models import MODEL_ASSUMPTIONS


@dataclass(frozen=True)
class GoalPlan:
    journey: str
    target_amount_today: float
    target_amount_future: float
    years: float
    current_savings: float
    planned_monthly_contribution: float
    assumed_annual_return: float
    projected_amount: float
    required_monthly_sip: float
    funding_gap: float
    success_prob: float
    # Newer, more honest metrics — used alongside success_prob in the UI:
    #  * funding_ratio: projected / target — distribution-free
    #  * p10/p50/p90: Monte-Carlo deciles of terminal wealth
    #  * outlook: "On track" / "Uncertain" / "At risk" (band)
    funding_ratio: float = 0.0
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    outlook: str = "Uncertain"


def _enrich(target_future: float, projected: float, current_savings: float,
             monthly_contribution: float, r: float, vol: float,
             years: float, success_prob: float) -> dict:
    """Compute funding ratio, MC deciles, and outlook band for a GoalPlan."""
    fr = funding_ratio(projected, target_future)
    if years > 0 and (current_savings > 0 or monthly_contribution > 0):
        p10, p50, p90 = monte_carlo_terminal_wealth(
            current_savings, monthly_contribution, r, vol, years,
        )
    else:
        p10 = p50 = p90 = projected
    return {
        "funding_ratio": round(fr, 4),
        "p10": round(p10, 2),
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "outlook": outlook_band(fr, success_prob),
    }


def _assumptions(risk_band: str) -> tuple[float, float]:
    a = MODEL_ASSUMPTIONS.get(risk_band, MODEL_ASSUMPTIONS["Moderate"])
    return a["expected_return"], a["volatility"]


def plan_retirement(current_age: int, target_retirement_age: int,
                     desired_monthly_income: float, current_savings: float,
                     monthly_contribution: float, risk_band: str) -> GoalPlan:
    """25x rule: target corpus = 25 × annual desired income (inflated to retirement)."""
    years = max(target_retirement_age - current_age, 0)
    annual_income_today = desired_monthly_income * 12
    target_today = annual_income_today * 25
    target_future = inflate(target_today, years, US_INFLATION)
    r, vol = _assumptions(risk_band)
    projected = project_portfolio(current_savings, monthly_contribution, r, years)
    req_sip = required_monthly_sip(target_future, current_savings, r, years)
    gap = max(target_future - projected, 0.0)
    prob = success_probability(projected, target_future, vol, years)
    extras = _enrich(target_future, projected, current_savings,
                      monthly_contribution, r, vol, years, prob)
    return GoalPlan(
        journey="Retirement Planning",
        target_amount_today=round(target_today, 2),
        target_amount_future=round(target_future, 2),
        years=years, current_savings=current_savings,
        planned_monthly_contribution=monthly_contribution,
        assumed_annual_return=r,
        projected_amount=round(projected, 2),
        required_monthly_sip=round(req_sip, 2),
        funding_gap=round(gap, 2),
        success_prob=round(prob, 3),
        **extras,
    )


def plan_child_education(child_current_age: int, target_cost_today: float,
                          current_savings: float, monthly_contribution: float,
                          risk_band: str, start_college_age: int = 18) -> GoalPlan:
    years = max(start_college_age - child_current_age, 0)
    edu_inflation = US_INFLATION + EDUCATION_INFLATION_PREMIUM
    target_future = inflate(target_cost_today, years, edu_inflation)
    r, vol = _assumptions(risk_band)
    projected = project_portfolio(current_savings, monthly_contribution, r, years)
    req_sip = required_monthly_sip(target_future, current_savings, r, years)
    gap = max(target_future - projected, 0.0)
    prob = success_probability(projected, target_future, vol, years)
    extras = _enrich(target_future, projected, current_savings,
                      monthly_contribution, r, vol, years, prob)
    return GoalPlan(
        journey="Child Education",
        target_amount_today=round(target_cost_today, 2),
        target_amount_future=round(target_future, 2),
        years=years, current_savings=current_savings,
        planned_monthly_contribution=monthly_contribution,
        assumed_annual_return=r,
        projected_amount=round(projected, 2),
        required_monthly_sip=round(req_sip, 2),
        funding_gap=round(gap, 2),
        success_prob=round(prob, 3),
        **extras,
    )


def plan_buy_home(home_price: float, down_payment_pct: float,
                    target_purchase_year: int, current_year: int,
                    current_savings: float, monthly_saving_capacity: float,
                    risk_band: str) -> GoalPlan:
    """Short-horizon goals cap the expected return at 4.5% regardless of risk band.

    Rationale: capital preservation matters more than growth when the horizon
    is under 5 years. We ignore risk_band in favour of a conservative return.
    """
    years = max(target_purchase_year - current_year, 0)
    down_payment_today = home_price * (down_payment_pct / 100)
    target_future = inflate(down_payment_today, years, US_INFLATION)
    r_ceiling = 0.045
    r_full, vol = _assumptions(risk_band)
    r = min(r_full, r_ceiling) if years <= 5 else min(r_full, 0.07)
    projected = project_portfolio(current_savings, monthly_saving_capacity, r, years)
    req_sip = required_monthly_sip(target_future, current_savings, r, years)
    gap = max(target_future - projected, 0.0)
    prob = success_probability(projected, target_future, vol, years)
    extras = _enrich(target_future, projected, current_savings,
                      monthly_saving_capacity, r, vol, years, prob)
    return GoalPlan(
        journey="Buy Home",
        target_amount_today=round(down_payment_today, 2),
        target_amount_future=round(target_future, 2),
        years=years, current_savings=current_savings,
        planned_monthly_contribution=monthly_saving_capacity,
        assumed_annual_return=r,
        projected_amount=round(projected, 2),
        required_monthly_sip=round(req_sip, 2),
        funding_gap=round(gap, 2),
        success_prob=round(prob, 3),
        **extras,
    )
