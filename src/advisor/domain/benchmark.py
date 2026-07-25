"""Benchmarking — turnkey ETF proxies per model portfolio.

Approach: one benchmark proxy per portfolio. AOM / AOR / AOA are iShares Core
Allocation ETFs whose glide-path mirrors the Morningstar Target Risk indexes.
This keeps benchmarking to a single quote per portfolio (cheap under AV free
tier), while still giving a legitimate peer comparison.

Live path: pull ~5Y of weekly closes for the proxy ETF (24h cached), then
compute annualized total return + realized volatility. We use weekly because
AV's free tier no longer serves the full daily series. Falls back to the
illustrative constants below when the fetch fails or the series is too short
to be meaningful. Custom-blend construction is documented in
corpus/client_portfolio_reference.md but not evaluated live.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from advisor.domain.models import MODEL_ASSUMPTIONS
from advisor.tools.alpha_vantage import AlphaVantageError, get_weekly_series

log = logging.getLogger(__name__)

# Minimum number of weekly observations before we trust the live series. Below
# this we fall back to illustrative — a handful of weeks would give a wildly
# noisy annualization and read as more confident than it is.
_MIN_SERIES_WEEKS = 52   # ~1 year of Friday closes
_WEEKS_PER_YEAR = 52
_LOOKBACK_YEARS = 5


@dataclass(frozen=True)
class BenchmarkProxy:
    model: str
    proxy_ticker: str
    proxy_name: str
    alt_ticker: str | None
    blend_description: str
    illustrative_return: float   # long-run expected annual return, illustrative
    rationale: str               # why THIS proxy for THIS risk band — shown as tooltip


BENCHMARK_PROXIES: dict[str, BenchmarkProxy] = {
    "Moderate": BenchmarkProxy(
        model="Moderate",
        proxy_ticker="AOM",
        proxy_name="iShares Core Moderate Allocation ETF",
        alt_ticker="VBIAX",
        blend_description=(
            "60% equity (MSCI ACWI or CRSP US Total Market) + 40% fixed income "
            "(Bloomberg US Aggregate Bond Index)"
        ),
        illustrative_return=0.065,
        rationale=(
            "AOM is iShares' Core Moderate Allocation ETF, holding ~40% equity / "
            "~60% bonds via underlying iShares index ETFs. Its Morningstar Target "
            "Risk category (Moderate) matches the risk band, so it's a like-for-like "
            "peer for a balanced portfolio — expense ratio 0.15%, daily-priced."
        ),
    ),
    "Growth": BenchmarkProxy(
        model="Growth",
        proxy_ticker="AOR",
        proxy_name="iShares Core Growth Allocation ETF",
        alt_ticker=None,
        blend_description=(
            "60% CRSP US Total Market + 20% FTSE Global All-Cap ex-US + 20% "
            "Bloomberg US Aggregate Bond Index"
        ),
        illustrative_return=0.085,
        rationale=(
            "AOR is iShares' Core Growth Allocation ETF, ~60% equity / ~40% bonds "
            "via underlying iShares index ETFs. Its Morningstar Target Risk "
            "category (Growth) matches the risk band, so it's a like-for-like peer "
            "for a tilted-to-equity portfolio — expense ratio 0.15%, daily-priced."
        ),
    ),
    "Aggressive": BenchmarkProxy(
        model="Aggressive",
        proxy_ticker="AOA",
        proxy_name="iShares Core Aggressive Allocation ETF",
        alt_ticker=None,
        blend_description=(
            "S&P 500 + Nasdaq-100 (via QQQ) + Russell 2000 (small-caps) with "
            "10% Bloomberg US Aggregate Bond Index"
        ),
        illustrative_return=0.100,
        rationale=(
            "AOA is iShares' Core Aggressive Allocation ETF, ~80% equity / ~20% "
            "bonds via underlying iShares index ETFs. Its Morningstar Target Risk "
            "category (Aggressive) matches the risk band, so it's a like-for-like "
            "peer for an equity-heavy portfolio — expense ratio 0.15%, daily-priced."
        ),
    ),
}


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    proxy_ticker: str
    proxy_name: str
    portfolio_expected_return: float
    benchmark_expected_return: float
    excess_return: float            # portfolio - benchmark
    tracking_error_est: float       # realized vol of proxy when live; else illustrative
    blend_description: str
    rationale: str
    benchmark_source: str           # 'live_5y' | 'illustrative_fallback'
    benchmark_series_weeks: int     # 0 for illustrative fallback


def _annualized_stats(series: list[tuple[str, float]]) -> tuple[float, float, int]:
    """CAGR + realized annualized volatility from a weekly-close series.

    Uses log-returns so the volatility is comparable to what an academic peer
    computes. CAGR is close-only (AV's ``TIME_SERIES_WEEKLY`` doesn't adjust for
    dividends) so this understates ETF total return slightly, but the delta
    versus portfolio expected return is directionally correct.
    """
    prices = [p for _, p in series if p > 0]
    n = len(prices)
    if n < _MIN_SERIES_WEEKS:
        return (0.0, 0.0, n)
    # Slice to the trailing 5Y window if the series is longer.
    max_bars = _LOOKBACK_YEARS * _WEEKS_PER_YEAR + 4
    if n > max_bars:
        prices = prices[-max_bars:]
        n = len(prices)
    start_p, end_p = prices[0], prices[-1]
    years = n / _WEEKS_PER_YEAR
    cagr = (end_p / start_p) ** (1 / years) - 1
    log_rets = [
        math.log(prices[i] / prices[i - 1])
        for i in range(1, n)
    ]
    if not log_rets:
        return (round(cagr, 4), 0.0, n)
    mean = sum(log_rets) / len(log_rets)
    var = sum((r - mean) ** 2 for r in log_rets) / max(len(log_rets) - 1, 1)
    weekly_vol = math.sqrt(var)
    ann_vol = weekly_vol * math.sqrt(_WEEKS_PER_YEAR)
    return (round(cagr, 4), round(ann_vol, 4), n)


def run_benchmarking(model_name: str, *, allow_live: bool = True) -> BenchmarkResult:
    if model_name not in BENCHMARK_PROXIES:
        raise ValueError(f"Unknown model for benchmarking: {model_name}")
    proxy = BENCHMARK_PROXIES[model_name]
    portfolio_ret = MODEL_ASSUMPTIONS[model_name]["expected_return"]
    portfolio_vol = MODEL_ASSUMPTIONS[model_name]["volatility"]

    live_return: float | None = None
    live_vol: float | None = None
    series_weeks = 0
    if allow_live:
        try:
            series = get_weekly_series(proxy.proxy_ticker)
            live_return, live_vol, series_weeks = _annualized_stats(series)
            if series_weeks < _MIN_SERIES_WEEKS:
                log.info(
                    "Benchmark series for %s too short (%d < %d weeks) — using illustrative",
                    proxy.proxy_ticker, series_weeks, _MIN_SERIES_WEEKS,
                )
                live_return = None
                live_vol = None
        except AlphaVantageError as exc:
            log.info("Benchmark live fetch failed for %s: %s", proxy.proxy_ticker, exc)
        except Exception as exc:  # noqa: BLE001 — never break the pipeline on benchmarking
            log.warning("Unexpected benchmark error for %s: %s", proxy.proxy_ticker, exc)

    if live_return is not None:
        benchmark_return = live_return
        tracking_error = live_vol if live_vol is not None else round(portfolio_vol * 0.15, 4)
        source = "live_5y"
    else:
        benchmark_return = proxy.illustrative_return
        # Illustrative tracking error: ~15% of portfolio vol. Not a live measurement.
        tracking_error = round(portfolio_vol * 0.15, 4)
        source = "illustrative_fallback"
        series_weeks = 0

    excess = portfolio_ret - benchmark_return
    return BenchmarkResult(
        model=model_name,
        proxy_ticker=proxy.proxy_ticker,
        proxy_name=proxy.proxy_name,
        portfolio_expected_return=round(portfolio_ret, 4),
        benchmark_expected_return=round(benchmark_return, 4),
        excess_return=round(excess, 4),
        tracking_error_est=tracking_error,
        blend_description=proxy.blend_description,
        rationale=proxy.rationale,
        benchmark_source=source,
        benchmark_series_weeks=series_weeks,
    )
