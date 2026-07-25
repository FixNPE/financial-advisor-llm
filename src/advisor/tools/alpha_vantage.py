"""Alpha Vantage API wrappers with on-disk caching.

Free tier is 25 requests/day and 5/min, so caching is essential during dev.
"""
from __future__ import annotations

from pathlib import Path

import requests_cache

from advisor.config import settings

_CACHE_DIR = Path("data/raw")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_session = requests_cache.CachedSession(
    str(_CACHE_DIR / "av_cache"),
    expire_after=60 * 60,  # 60 min — widened for mentored-session demo buffer
    allowable_methods=("GET",),
)
# Daily-series cache is separate from the quote cache: series responses are
# large (100+ KB each), pull rarely (once a day, per proxy ETF), and would
# otherwise flush live quote hits when the shared cache trims. 24-hour TTL
# is enough that benchmarking stays under the 25-req/day free-tier ceiling.
_series_session = requests_cache.CachedSession(
    str(_CACHE_DIR / "av_series_cache"),
    expire_after=60 * 60 * 24,
    allowable_methods=("GET",),
)
_BASE = "https://www.alphavantage.co/query"


class AlphaVantageError(RuntimeError):
    pass


def _call(params: dict, session=None) -> dict:
    params = {**params, "apikey": settings.alpha_vantage_key}
    sess = session if session is not None else _session
    r = sess.get(_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    info_str = str(data.get("Information", ""))
    info_lower = info_str.lower()
    rate_limited = "Note" in data or (
        "Information" in data and "rate limit" in info_lower
    )
    premium_gated = "Information" in data and (
        "premium" in info_lower or "subscribe" in info_lower
    )
    if rate_limited or premium_gated or "Error Message" in data:
        # AV returns HTTP 200 for rate-limit/error responses, so requests-cache would
        # otherwise store them and poison the cache until TTL. Evict before raising.
        if getattr(r, "from_cache", False) is False:
            try:
                sess.cache.delete(r.cache_key)
            except Exception:  # noqa: BLE001
                pass
        if rate_limited:
            msg = data.get("Note") or data.get("Information")
            raise AlphaVantageError(f"AV rate-limited: {msg}")
        if premium_gated:
            raise AlphaVantageError(f"AV premium-gated: {info_str[:200]}")
        raise AlphaVantageError(f"AV error: {data['Error Message']}")
    return data


def get_quote(symbol: str) -> dict:
    """Latest price + change for a symbol."""
    data = _call({"function": "GLOBAL_QUOTE", "symbol": symbol.upper()})
    q = data.get("Global Quote", {})
    return {
        "symbol": q.get("01. symbol"),
        "price": float(q["05. price"]) if q.get("05. price") else None,
        "change": float(q["09. change"]) if q.get("09. change") else None,
        "change_percent": q.get("10. change percent"),
        "volume": int(q["06. volume"]) if q.get("06. volume") else None,
        "latest_trading_day": q.get("07. latest trading day"),
    }


def get_weekly_series(symbol: str) -> list[tuple[str, float]]:
    """Weekly-close series (Friday closes), oldest → newest.

    We use weekly rather than daily because Alpha Vantage now gates
    ``TIME_SERIES_DAILY?outputsize=full`` behind a premium tier, while
    ``TIME_SERIES_WEEKLY`` still returns the full ~20-year history on the free
    plan. ~260 weekly observations over 5Y is plenty for CAGR + annualized
    volatility (σ_weekly · √52). Cached 24h. Empty list on parse failure —
    caller decides whether to fall back.
    """
    data = _call({
        "function": "TIME_SERIES_WEEKLY",
        "symbol": symbol.upper(),
    }, session=_series_session)
    ts = data.get("Weekly Time Series") or {}
    if not ts:
        return []
    rows: list[tuple[str, float]] = []
    for date_str, bar in ts.items():
        close = bar.get("4. close")
        if close is None:
            continue
        try:
            rows.append((date_str, float(close)))
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda r: r[0])
    return rows


def get_company_overview(symbol: str) -> dict:
    """Fundamentals for a stock."""
    data = _call({"function": "OVERVIEW", "symbol": symbol.upper()})
    keep = [
        "Symbol", "Name", "Sector", "Industry", "MarketCapitalization",
        "PERatio", "PEGRatio", "DividendYield", "EPS", "Beta",
        "52WeekHigh", "52WeekLow", "Description",
    ]
    return {k: data.get(k) for k in keep}


def get_news_sentiment(tickers: str | None = None, topics: str | None = None,
                       limit: int = 20) -> dict:
    """Recent news with sentiment scores. tickers='AAPL,MSFT' or topics='technology,ipo'."""
    p: dict = {"function": "NEWS_SENTIMENT", "limit": limit}
    if tickers:
        p["tickers"] = tickers.upper()
    if topics:
        p["topics"] = topics
    raw = _call(p)
    feed = raw.get("feed", [])[:limit]
    return {
        "items": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "time": item.get("time_published"),
                "summary": item.get("summary"),
                "overall_sentiment_label": item.get("overall_sentiment_label"),
                "overall_sentiment_score": item.get("overall_sentiment_score"),
                "tickers": [t.get("ticker") for t in item.get("ticker_sentiment", [])],
            }
            for item in feed
        ],
        "count": len(feed),
    }


def get_technical(symbol: str, indicator: str = "RSI", interval: str = "daily",
                  time_period: int = 14, series_type: str = "close") -> dict:
    """Technical indicator (RSI, SMA, EMA, MACD, ...) for a symbol."""
    return _call({
        "function": indicator.upper(),
        "symbol": symbol.upper(),
        "interval": interval,
        "time_period": time_period,
        "series_type": series_type,
    })


def get_sector_performance() -> dict:
    """Realtime + lagged sector performance."""
    return _call({"function": "SECTOR"})


def get_fx_rate(from_currency: str, to_currency: str) -> dict:
    data = _call({
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_currency.upper(),
        "to_currency": to_currency.upper(),
    })
    rate = data.get("Realtime Currency Exchange Rate", {})
    return {
        "from": rate.get("1. From_Currency Code"),
        "to": rate.get("3. To_Currency Code"),
        "rate": float(rate["5. Exchange Rate"]) if rate.get("5. Exchange Rate") else None,
        "as_of": rate.get("6. Last Refreshed"),
    }
