"""Deterministic financial calculators. Pure functions, easy to test."""
from __future__ import annotations

import math


def retirement_projection(
    current_age: int,
    retire_age: int,
    current_savings: float,
    monthly_contribution: float,
    annual_return: float = 0.07,
) -> dict:
    """Project retirement portfolio value with monthly compounding."""
    if retire_age <= current_age:
        return {"error": "retire_age must exceed current_age"}
    months = (retire_age - current_age) * 12
    r = annual_return / 12
    fv = current_savings * (1 + r) ** months
    if r > 0:
        fv += monthly_contribution * (((1 + r) ** months - 1) / r)
    else:
        fv += monthly_contribution * months
    return {
        "future_value": round(fv, 2),
        "years": retire_age - current_age,
        "monthly_contribution": monthly_contribution,
        "assumed_return": annual_return,
    }


def _plan_to_dict(plan, risk_band: str) -> dict:
    """Shared serialiser for plan_* results."""
    from advisor.domain.models import MODEL_ASSUMPTIONS
    a = MODEL_ASSUMPTIONS[risk_band]
    return {
        "journey": plan.journey,
        "risk_band": risk_band,
        "expected_return": a["expected_return"],
        "volatility": a["volatility"],
        "years": plan.years,
        "target_amount_today": plan.target_amount_today,
        "target_amount_future": plan.target_amount_future,
        "projected_amount": plan.projected_amount,
        "funding_gap": plan.funding_gap,
        "funding_ratio": plan.funding_ratio,
        "required_monthly_sip": plan.required_monthly_sip,
        "success_prob": plan.success_prob,
        "p10": plan.p10,
        "p50": plan.p50,
        "p90": plan.p90,
        "outlook": plan.outlook,
        "assumed_annual_return": plan.assumed_annual_return,
    }


def _validate_band(risk_band: str) -> str | None:
    from advisor.domain.models import MODEL_ASSUMPTIONS
    if risk_band not in MODEL_ASSUMPTIONS:
        return f"risk_band must be one of {list(MODEL_ASSUMPTIONS)}"
    return None


def plan_retirement(
    current_age: int,
    target_retirement_age: int,
    desired_monthly_income: float,
    current_savings: float,
    monthly_contribution: float,
    risk_band: str,
) -> dict:
    """Recompute the retirement plan (target, projection, SIP, funding ratio)."""
    from advisor.agents.goal_agent import plan_retirement as _plan
    err = _validate_band(risk_band)
    if err:
        return {"error": err}
    plan = _plan(
        current_age=current_age,
        target_retirement_age=target_retirement_age,
        desired_monthly_income=desired_monthly_income,
        current_savings=current_savings,
        monthly_contribution=monthly_contribution,
        risk_band=risk_band,
    )
    return _plan_to_dict(plan, risk_band)


def plan_education(
    child_current_age: int,
    target_cost_today: float,
    current_savings: float,
    monthly_contribution: float,
    risk_band: str,
    start_college_age: int = 18,
) -> dict:
    """Recompute the child-education plan (target, projection, SIP, funding ratio)."""
    from advisor.agents.goal_agent import plan_child_education as _plan
    err = _validate_band(risk_band)
    if err:
        return {"error": err}
    plan = _plan(
        child_current_age=child_current_age,
        target_cost_today=target_cost_today,
        current_savings=current_savings,
        monthly_contribution=monthly_contribution,
        risk_band=risk_band,
        start_college_age=start_college_age,
    )
    return _plan_to_dict(plan, risk_band)


def plan_home(
    home_price: float,
    down_payment_pct: float,
    target_purchase_year: int,
    current_year: int,
    current_savings: float,
    monthly_saving_capacity: float,
    risk_band: str,
) -> dict:
    """Recompute the home-purchase plan (target, projection, SIP, funding ratio)."""
    from advisor.agents.goal_agent import plan_buy_home as _plan
    err = _validate_band(risk_band)
    if err:
        return {"error": err}
    plan = _plan(
        home_price=home_price,
        down_payment_pct=down_payment_pct,
        target_purchase_year=target_purchase_year,
        current_year=current_year,
        current_savings=current_savings,
        monthly_saving_capacity=monthly_saving_capacity,
        risk_band=risk_band,
    )
    return _plan_to_dict(plan, risk_band)


def savings_goal(target: float, years: int, annual_return: float = 0.05) -> dict:
    """Required monthly contribution to hit a target value."""
    if years <= 0:
        return {"error": "years must be positive"}
    months = years * 12
    r = annual_return / 12
    if r == 0:
        monthly = target / months
    else:
        monthly = target * r / ((1 + r) ** months - 1)
    return {
        "monthly_contribution": round(monthly, 2),
        "target": target,
        "years": years,
        "assumed_return": annual_return,
    }


def asset_allocation(age: int, risk_tolerance: str) -> dict:
    """Glide-path style allocation (stocks/bonds/cash). risk_tolerance in {low,moderate,high}."""
    if risk_tolerance not in ("low", "moderate", "high"):
        return {"error": "risk_tolerance must be low|moderate|high"}
    base_equity = max(20, 110 - age)
    adj = {"low": -15, "moderate": 0, "high": 10}[risk_tolerance]
    eq = max(20, min(95, base_equity + adj))
    bonds = 100 - eq - 5
    return {"stocks_pct": eq, "bonds_pct": bonds, "cash_pct": 5,
            "rationale": f"Glide path = 110 - age ({110 - age}%) adjusted for {risk_tolerance} risk."}


def debt_payoff(balance: float, apr: float, monthly_payment: float) -> dict:
    """Months to pay off and total interest paid (fixed monthly payment)."""
    r = apr / 12
    if monthly_payment <= balance * r:
        return {"error": "Payment does not cover interest — debt grows."}
    n = math.log(monthly_payment / (monthly_payment - balance * r)) / math.log(1 + r)
    return {
        "months_to_payoff": round(n, 1),
        "years_to_payoff": round(n / 12, 1),
        "total_paid": round(monthly_payment * n, 2),
        "total_interest": round(monthly_payment * n - balance, 2),
    }


def emergency_fund(monthly_expenses: float, months_target: int = 6) -> dict:
    return {
        "target_amount": round(monthly_expenses * months_target, 2),
        "months_target": months_target,
    }
