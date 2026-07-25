"""OpenAI-compatible tool catalog + dispatch table for the ReAct agent."""
from __future__ import annotations

from advisor.tools import alpha_vantage as av
from advisor.tools import calculators as calc

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Latest price, change, and volume for a stock symbol.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_overview",
            "description": "Fundamentals for a stock: sector, P/E, market cap, dividend yield, etc.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": "Recent news headlines and sentiment scores. Provide tickers (comma-separated) or topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "string", "description": "Comma-separated tickers, e.g. AAPL,MSFT"},
                    "topics": {"type": "string", "description": "AV topics, e.g. technology,ipo,economy_macro"},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicator",
            "description": "Compute a technical indicator (RSI, SMA, EMA, MACD) for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "indicator": {"type": "string", "enum": ["RSI", "SMA", "EMA", "MACD"]},
                    "interval": {"type": "string", "default": "daily"},
                    "time_period": {"type": "integer", "default": 14},
                },
                "required": ["symbol", "indicator"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_performance",
            "description": "Real-time and trailing performance for the major US sectors.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fx_rate",
            "description": "Realtime FX rate between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retirement_projection",
            "description": "Project portfolio future value at retirement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_age": {"type": "integer"},
                    "retire_age": {"type": "integer"},
                    "current_savings": {"type": "number"},
                    "monthly_contribution": {"type": "number"},
                    "annual_return": {"type": "number", "default": 0.07},
                },
                "required": ["current_age", "retire_age", "current_savings", "monthly_contribution"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "savings_goal",
            "description": "Compute required monthly contribution to hit a target value over N years.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "number"},
                    "years": {"type": "integer"},
                    "annual_return": {"type": "number", "default": 0.05},
                },
                "required": ["target", "years"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "asset_allocation",
            "description": "Suggested stocks/bonds/cash split given age and risk tolerance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "risk_tolerance": {"type": "string", "enum": ["low", "moderate", "high"]},
                },
                "required": ["age", "risk_tolerance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "debt_payoff",
            "description": "Months to pay off a debt and total interest paid given balance, APR, monthly payment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "balance": {"type": "number"},
                    "apr": {"type": "number", "description": "Annual rate as decimal, e.g. 0.18 for 18%"},
                    "monthly_payment": {"type": "number"},
                },
                "required": ["balance", "apr", "monthly_payment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_retirement",
            "description": (
                "Recompute the retirement plan (target corpus, projection, "
                "required SIP, funding ratio, Monte-Carlo p10/p50/p90). Use "
                "the goal_inputs values from PAGE FACTS verbatim — do not "
                "invent or round them. Only change values the user explicitly "
                "asked to change (e.g. risk_band for a 'what-if aggressive' "
                "question)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_age": {"type": "integer"},
                    "target_retirement_age": {"type": "integer"},
                    "desired_monthly_income": {"type": "number", "description": "In today's dollars"},
                    "current_savings": {"type": "number"},
                    "monthly_contribution": {"type": "number"},
                    "risk_band": {
                        "type": "string",
                        "enum": ["Moderate", "Growth", "Aggressive"],
                        "description": "Sets expected_return/volatility from MODEL_ASSUMPTIONS (Moderate 7.5%/10%, Growth 9.5%/13%, Aggressive 11.5%/17%).",
                    },
                },
                "required": [
                    "current_age", "target_retirement_age",
                    "desired_monthly_income", "current_savings",
                    "monthly_contribution", "risk_band",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_education",
            "description": (
                "Recompute the child-education plan (target, projection, "
                "required SIP, funding ratio, Monte-Carlo p10/p50/p90). Use "
                "the goal_inputs from PAGE FACTS verbatim — do not invent "
                "target_cost_today or bump monthly_contribution unless the "
                "user explicitly asked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "child_current_age": {"type": "integer"},
                    "target_cost_today": {"type": "number", "description": "Total education cost in today's dollars"},
                    "current_savings": {"type": "number"},
                    "monthly_contribution": {"type": "number"},
                    "risk_band": {
                        "type": "string",
                        "enum": ["Moderate", "Growth", "Aggressive"],
                    },
                    "start_college_age": {"type": "integer", "default": 18},
                },
                "required": [
                    "child_current_age", "target_cost_today",
                    "current_savings", "monthly_contribution", "risk_band",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_home",
            "description": (
                "Recompute the home-purchase plan (down-payment target, "
                "projection, required SIP, funding ratio, Monte-Carlo "
                "p10/p50/p90). Use the goal_inputs from PAGE FACTS verbatim. "
                "Short-horizon goals (≤5y) cap expected return at 4.5% "
                "regardless of risk_band — this is by design."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "home_price": {"type": "number"},
                    "down_payment_pct": {"type": "number", "description": "As a percentage, e.g. 20 for 20%"},
                    "target_purchase_year": {"type": "integer"},
                    "current_year": {"type": "integer"},
                    "current_savings": {"type": "number"},
                    "monthly_saving_capacity": {"type": "number"},
                    "risk_band": {
                        "type": "string",
                        "enum": ["Moderate", "Growth", "Aggressive"],
                    },
                },
                "required": [
                    "home_price", "down_payment_pct", "target_purchase_year",
                    "current_year", "current_savings",
                    "monthly_saving_capacity", "risk_band",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emergency_fund",
            "description": "Recommended emergency fund target given monthly expenses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_expenses": {"type": "number"},
                    "months_target": {"type": "integer", "default": 6},
                },
                "required": ["monthly_expenses"],
            },
        },
    },
]

DISPATCH: dict = {
    "get_stock_quote": lambda symbol: av.get_quote(symbol),
    "get_company_overview": lambda symbol: av.get_company_overview(symbol),
    "get_news_sentiment": lambda **kw: av.get_news_sentiment(**kw),
    "get_technical_indicator": lambda symbol, indicator, interval="daily", time_period=14: av.get_technical(
        symbol=symbol, indicator=indicator, interval=interval, time_period=time_period
    ),
    "get_sector_performance": lambda: av.get_sector_performance(),
    "get_fx_rate": lambda from_currency, to_currency: av.get_fx_rate(from_currency, to_currency),
    "retirement_projection": calc.retirement_projection,
    "plan_retirement": calc.plan_retirement,
    "plan_education": calc.plan_education,
    "plan_home": calc.plan_home,
    "savings_goal": calc.savings_goal,
    "asset_allocation": calc.asset_allocation,
    "debt_payoff": calc.debt_payoff,
    "emergency_fund": calc.emergency_fund,
}
