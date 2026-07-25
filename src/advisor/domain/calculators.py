"""Financial math: FV, SIP, drift, success probability.

All functions are pure and deterministic — no LLM, no I/O. The Goal Planning
and Recommendation agents call these; the LLM only narrates the results.
"""
from __future__ import annotations

import math

US_INFLATION = 0.03                    # long-run US CPI assumption
EDUCATION_INFLATION_PREMIUM = 0.02     # education inflation ≈ CPI + 2%


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def future_value(present_value: float, annual_rate: float, years: float) -> float:
    """FV of a lump sum compounded annually."""
    if years <= 0:
        return present_value
    return present_value * ((1 + annual_rate) ** years)


def future_value_annuity(monthly_contribution: float, annual_rate: float, years: float) -> float:
    """FV of a monthly SIP compounded monthly."""
    if years <= 0 or monthly_contribution <= 0:
        return 0.0
    n = years * 12
    r = annual_rate / 12
    if r == 0:
        return monthly_contribution * n
    return monthly_contribution * (((1 + r) ** n - 1) / r)


def project_portfolio(current_savings: float, monthly_contribution: float,
                       annual_rate: float, years: float) -> float:
    """Combined projection: lump-sum FV + monthly-SIP FV."""
    return (future_value(current_savings, annual_rate, years)
            + future_value_annuity(monthly_contribution, annual_rate, years))


def required_monthly_sip(target: float, current_savings: float,
                          annual_rate: float, years: float) -> float:
    """Solve for the monthly contribution that reaches ``target`` in ``years``.

    Returns 0 if the lump-sum future value alone already meets or exceeds
    ``target`` — no additional SIP required.
    """
    if years <= 0:
        return max(target - current_savings, 0.0)
    lump_fv = future_value(current_savings, annual_rate, years)
    remaining = max(target - lump_fv, 0.0)
    if remaining <= 0:
        return 0.0
    n = years * 12
    r = annual_rate / 12
    if r == 0:
        return remaining / n
    return remaining * r / ((1 + r) ** n - 1)


def inflate(present_amount: float, years: float, inflation: float = US_INFLATION) -> float:
    """Inflate a present-day amount to a future year."""
    return present_amount * ((1 + inflation) ** years)


def success_probability(projected: float, target: float, volatility: float,
                          years: float = 1.0) -> float:
    """Probability that the portfolio meets ``target`` at horizon.

    Models the terminal wealth as log-normal with expected value ``projected``
    and log-std that scales with ``sqrt(years)`` — the correct dimensional
    behaviour for a compounded return. The previous version used the annual
    volatility as the *terminal* std (no time scaling), which made every
    plan with the smallest funding gap collapse to <1%.

    ``volatility`` is the annualised portfolio std-dev (from
    MODEL_ASSUMPTIONS); ``years`` is the plan horizon. If projected == 0 or
    target == 0 we short-circuit.
    """
    if projected <= 0:
        return 0.0
    if target <= 0:
        return 1.0
    if years <= 0:
        return 1.0 if projected >= target else 0.0
    sigma = max(volatility * math.sqrt(years), 1e-6)
    # log-normal: median = projected * exp(-sigma^2/2); we set the mean to
    # ``projected`` and back out mu so E[X] = projected.
    mu = math.log(projected) - 0.5 * sigma ** 2
    # P(X >= target) = 1 - Phi((ln target - mu)/sigma)
    z = (math.log(target) - mu) / sigma
    return clamp(0.5 * (1 - math.erf(z / math.sqrt(2))), 0.0, 1.0)


def funding_ratio(projected: float, target: float) -> float:
    """``projected / target``, clamped at 0 and 5.

    A distribution-free "how funded is this plan" number that reads
    intuitively (79% funded, 120% funded). Free of the volatility-scaling
    trap that plagues success_probability.
    """
    if target <= 0:
        return 1.0 if projected >= 0 else 0.0
    return clamp(projected / target, 0.0, 5.0)


def outlook_band(funding_ratio_value: float, success_prob_value: float) -> str:
    """Human label for the plan's overall health.

    Combines funding ratio and success probability so we don't flag a
    plan as "at risk" only because its Monte-Carlo tail is thin. Bands
    are calibrated to keep the *action guidance* sensible: a 95% funded
    plan is Uncertain, not At risk, even if the probability of clearing
    the target on the nose is 40%.
    """
    if funding_ratio_value >= 1.0 or success_prob_value >= 0.70:
        return "On track"
    if funding_ratio_value >= 0.75 or success_prob_value >= 0.35:
        return "Uncertain"
    return "At risk"


def monte_carlo_terminal_wealth(current_savings: float,
                                  monthly_contribution: float,
                                  annual_return: float,
                                  volatility: float,
                                  years: float,
                                  n_paths: int = 2000,
                                  seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap p10/p50/p90 of terminal wealth via a monthly-return simulation.

    Draws monthly log-returns from Normal(mu_m, sigma_m) where
    ``mu_m = ln(1+r)/12`` and ``sigma_m = vol/sqrt(12)``; compounds a lump-sum
    with a level monthly contribution over ``years * 12`` steps. Returns
    (p10, p50, p90) terminal-wealth percentiles.

    Deterministic under a fixed ``seed`` so pipeline runs stay reproducible.
    2000 paths is enough for stable deciles at demo scale (<10ms).
    """
    if years <= 0 or (current_savings <= 0 and monthly_contribution <= 0):
        v = max(current_savings, 0.0)
        return v, v, v
    import random
    rng = random.Random(seed)
    n_months = int(round(years * 12))
    mu_m = math.log(1 + annual_return) / 12
    sigma_m = volatility / math.sqrt(12)
    finals: list[float] = []
    for _ in range(n_paths):
        wealth = current_savings
        for _m in range(n_months):
            # Box-Muller for a standard normal (avoid numpy dependency).
            u1 = 1.0 - rng.random()  # (0,1]
            u2 = rng.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
            r = math.exp(mu_m + sigma_m * z) - 1.0
            wealth = wealth * (1 + r) + monthly_contribution
        finals.append(wealth)
    finals.sort()
    def pct(p: float) -> float:
        idx = clamp(int(p * len(finals)), 0, len(finals) - 1)
        return finals[int(idx)]
    return pct(0.10), pct(0.50), pct(0.90)


def drift(current_pct: dict[str, float], target_pct: dict[str, float]) -> dict[str, float]:
    """Per-asset-class drift = current% - target%. Positive = overweight."""
    classes = set(current_pct) | set(target_pct)
    return {ac: round(current_pct.get(ac, 0.0) - target_pct.get(ac, 0.0), 2)
            for ac in classes}


def max_abs_drift(drift_map: dict[str, float]) -> float:
    if not drift_map:
        return 0.0
    return max(abs(v) for v in drift_map.values())


def rebalancing_actions(current_pct: dict[str, float], target_pct: dict[str, float],
                         min_drift_pct: float = 5.0) -> list[dict]:
    """Generate rebalancing actions when |drift| >= min_drift_pct on any class.

    Returns a list of dicts:
        {"asset_class": ..., "action": "trim"|"add", "delta_pct": ...}
    Empty list when no class exceeds the threshold.
    """
    d = drift(current_pct, target_pct)
    if max_abs_drift(d) < min_drift_pct:
        return []
    actions = []
    for ac, delta in sorted(d.items(), key=lambda kv: -abs(kv[1])):
        if abs(delta) < min_drift_pct:
            continue
        actions.append({
            "asset_class": ac,
            "action": "trim" if delta > 0 else "add",
            "delta_pct": round(abs(delta), 2),
        })
    return actions
