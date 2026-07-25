"""New-user onboarding — chat-driven, goal-aware intake.

Three phases:

  1. **Intro** (4 Qs) — name, age, monthly income, goal_type.
     ``goal_type`` classifies into one of the three journeys:
       - ``Retirement Planning``
       - ``Child Education``
       - ``Buy Home``
  2. **Branch** — goal-specific follow-up Qs (different per journey).
  3. **Tail** (3 Qs) — savings, health, risk_confirm.

The composed question list is computed dynamically from
``st.session_state[KEY_ONBOARD_STATE]["goal_type"]`` — before the goal is
chosen, only the intro is visible; once chosen, the branch + tail slot in.
That lets ``step`` stay a simple int index into the list while the total
grows/changes as the user commits to a goal.

Answer extraction is entirely regex — no LLM required for onboarding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from advisor.domain.data import Customer, list_customers, upsert_customer

from app.components.session import KEY_LAST_PIPELINE, set_active_customer

KEY_ONBOARD_STATE = "onboard_state"           # dict[str, Any] — collected answers
KEY_ONBOARD_HISTORY = "onboard_history"       # list[{role, content}]
KEY_ONBOARD_STEP = "onboard_step"             # int — index into current questions

CURRENT_YEAR = 2026


# ---- Value extractors -----------------------------------------------------

_NAME_PREFIX_RE = re.compile(
    r"^\s*(?:my\s+name\s+is|i(?:'m|\s*am)|this\s+is|it'?s|call\s+me|"
    r"they\s+call\s+me|name'?s|hi[,!\s]+i(?:'m|\s*am))\s+",
    re.IGNORECASE,
)

# Only keep capitalised words (proper nouns) at the start — stops the moment
# we hit any non-name token: comma, period, digit, lowercase word, etc.
_NAME_TOKEN_RE = re.compile(r"^([A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*){0,3})")


def _extract_name(raw: str) -> str | None:
    """Pull just the name out of natural phrasing.

    Handles the common lead-ins ("I am Peter Parker", "my name is Peter",
    "It's Peter Parker, 35 year old male") — strips the lead-in, then keeps
    only the leading 1-4 capitalised words. Any trailing clause (age,
    occupation, extra commentary) is dropped so we don't store the whole
    sentence as the name.
    """
    if not raw:
        return None
    cleaned = _NAME_PREFIX_RE.sub("", raw).strip()
    # Title-case single lowercase names ("peter parker") so the token regex
    # still catches them.
    tokens = cleaned.split()
    if tokens and tokens[0].islower():
        cleaned = " ".join(t.capitalize() if t.isalpha() else t for t in tokens)
    m = _NAME_TOKEN_RE.match(cleaned)
    if not m:
        return None
    name = m.group(1).strip()
    if not name or len(name) > 80:
        return None
    return name


def _extract_int(raw: str, lo: int, hi: int) -> int | None:
    m = re.search(r"\d+", raw.replace(",", ""))
    if not m:
        return None
    val = int(m.group())
    if lo <= val <= hi:
        return val
    return None


def _extract_money(raw: str) -> float | None:
    """Parse '80000', '80,000', '$5,000', '5k', '1.2m' — USD only.

    The app targets the US market: no INR-scale units (lakh/crore) are
    recognised. Suffixes 'k' → ×1_000, 'm' → ×1_000_000 are accepted for
    convenience.
    """
    text = raw.lower().replace(",", "").replace("$", "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    value = float(m.group(1))
    if "m" in text and "mo" not in text:  # 'm' but not 'month/monthly'
        return value * 1_000_000
    if "k" in text:
        return value * 1_000
    return value


def _extract_money_from_freetext(raw: str) -> float | None:
    """Look for a money-like number inside a longer freetext answer.

    Returns the first plausible dollar amount if one is present, else None.
    Used to pull a numeric ``current_savings`` out of the tail ``savings``
    question when the user answered with a sentence like "I have $100,000
    in a brokerage account".
    """
    if not raw:
        return None
    text = raw.lower()
    m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(k|m)?", text)
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    suffix = m.group(2)
    if suffix == "m":
        value *= 1_000_000
    elif suffix == "k":
        value *= 1_000
    return value if value > 0 else None


def _extract_pct(raw: str) -> float | None:
    text = raw.replace("%", "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    val = float(m.group(1))
    if 0 <= val <= 100:
        return val
    return None


def _extract_year(raw: str) -> int | None:
    """Accept absolute years ('2029') or relative ('in 4 years' → 2030)."""
    text = raw.lower()
    m_abs = re.search(r"(20\d{2})", text)
    if m_abs:
        year = int(m_abs.group(1))
        if CURRENT_YEAR <= year <= 2100:
            return year
    m_rel = re.search(r"(\d+)\s*year", text)
    if m_rel:
        return CURRENT_YEAR + int(m_rel.group(1))
    m_num = re.search(r"\d+", text)
    if m_num:
        n = int(m_num.group())
        if 2 <= n <= 50:  # "in 4" — treat as relative
            return CURRENT_YEAR + n
    return None


def _extract_goal(raw: str) -> str | None:
    text = raw.strip().lower()
    if not text:
        return None
    if text in {"1", "(1)"} or any(k in text for k in
                                    ("retire", "retirement", "pension")):
        return "Retirement Planning"
    if text in {"2", "(2)"} or any(k in text for k in
                                    ("child", "education", "college",
                                     "school", "kid", "daughter", "son",
                                     "tuition")):
        return "Child Education"
    if text in {"3", "(3)"} or any(k in text for k in
                                    ("home", "house", "buy", "property",
                                     "flat", "apartment", "down payment",
                                     "downpayment")):
        return "Buy Home"
    return None


def _extract_freetext(raw: str) -> str:
    return raw.strip() or "none"


# ---- Displays -------------------------------------------------------------

def _display_money(v: float) -> str:
    return f"${v:,.0f}"


def _display_pct(v: float) -> str:
    return f"{v:.0f}%"


# ---- Question script ------------------------------------------------------

@dataclass
class OnboardQ:
    key: str
    prompt: str
    extract: Callable[[str], Any]
    display: Callable[[Any], str]
    error: str


_INTRO: list[OnboardQ] = [
    OnboardQ("name",
                "Hi! I'm FinAdvisor. What's your name?",
                _extract_name, lambda v: v,
                "Please share a name so I can save your profile."),
    OnboardQ("age",
                "Nice to meet you, {name}! How old are you?",
                lambda r: _extract_int(r, 18, 100), lambda v: str(v),
                "I need an age between 18 and 100."),
    OnboardQ("income",
                "What's your monthly take-home income? (e.g. $5,000 or $8k)",
                _extract_money, _display_money,
                "I couldn't read that amount — try a number like 5000 or $5,000."),
    OnboardQ("goal_type",
                "Great. What would you like to plan for today?\n\n"
                "  **(1)** Retirement\n"
                "  **(2)** Child's education\n"
                "  **(3)** Buying a home",
                _extract_goal, lambda v: v,
                "Please pick one — reply with 1/2/3 or the option name."),
]


_BRANCH: dict[str, list[OnboardQ]] = {
    "Retirement Planning": [
        OnboardQ("retire_age",
                    "At what age would you like to retire?",
                    lambda r: _extract_int(r, 40, 90), lambda v: str(v),
                    "Retirement age should be between 40 and 90."),
        OnboardQ("monthly_need",
                    "What monthly income would you need after retirement "
                    "(in today's dollars)?",
                    _extract_money, _display_money,
                    "I couldn't read that amount — try 6000 or $6,000."),
    ],
    "Child Education": [
        OnboardQ("child_current_age",
                    "How old is your child today?",
                    lambda r: _extract_int(r, 0, 17), lambda v: str(v),
                    "Child's age should be between 0 and 17."),
        OnboardQ("target_cost_today",
                    "What's your target for their 4-year college cost "
                    "(in today's dollars)?",
                    _extract_money, _display_money,
                    "I couldn't read that amount — try 120000 or $120k."),
    ],
    "Buy Home": [
        OnboardQ("home_price",
                    "What's your target home price? (in USD)",
                    _extract_money, _display_money,
                    "I couldn't read that amount — try 500000 or $500k."),
        OnboardQ("down_payment_pct",
                    "How much down payment percentage are you planning? "
                    "(e.g. 20%)",
                    _extract_pct, _display_pct,
                    "Down payment percentage should be between 0 and 100."),
        OnboardQ("target_purchase_year",
                    "By what year do you want to buy? (e.g. 2029 or "
                    "'in 4 years')",
                    _extract_year, lambda v: str(v),
                    "I need a year between "
                    f"{CURRENT_YEAR} and 2100."),
    ],
}


REQUIRED_GOAL_INPUTS: dict[str, tuple[str, ...]] = {
    "Retirement Planning": ("target_retirement_age", "desired_monthly_income"),
    "Child Education":     ("child_current_age", "target_cost_today"),
    "Buy Home":            ("home_price", "down_payment_pct",
                              "target_purchase_year"),
}


def missing_goal_input_fields(journey: str, goal_inputs: dict | None) -> list[str]:
    """Return the required goal_input keys that are missing for ``journey``.

    Used by the chat to decide whether a "you haven't set up X yet" prompt
    should fire instead of falling through to the ReAct advisor.
    """
    required = REQUIRED_GOAL_INPUTS.get(journey, ())
    gi = goal_inputs or {}
    return [k for k in required if k not in gi]


def branch_questions(journey: str) -> list[OnboardQ]:
    """Return the branch-only question list for ``journey``.

    Exposed so the mini-onboarding flow (invoked when an existing user asks
    about a goal that isn't set up yet) can reuse exactly the same prompts
    and extractors as the full onboarding.
    """
    return list(_BRANCH.get(journey, []))


_TAIL: list[OnboardQ] = [
    OnboardQ("savings",
                "Do you have any existing investments — 401(k), IRA, "
                "brokerage, or savings? If yes, roughly how much in USD?",
                _extract_freetext, lambda v: v, ""),
    OnboardQ("health",
                'Any critical health conditions I should factor in for '
                'insurance planning? (e.g. diabetes, heart condition, or "none")',
                _extract_freetext, lambda v: v, ""),
    OnboardQ("risk_confirm",
                "Almost done! One last thing — if your portfolio dropped 25% "
                "tomorrow, would you (a) hold steady, (b) partially shift to "
                "safer funds, (c) move everything to a high-yield savings "
                "account or Treasury bills, or (d) buy more while prices are low?",
                _extract_freetext, lambda v: v, ""),
]


def _current_questions() -> list[OnboardQ]:
    """Compose the active question list based on the chosen goal.

    Before ``goal_type`` is picked, only the intro is visible so the
    total-questions count doesn't jump around. Once picked, the branch +
    tail slot in immediately.
    """
    goal = st.session_state[KEY_ONBOARD_STATE].get("goal_type")
    if goal is None or goal not in _BRANCH:
        return list(_INTRO)
    return list(_INTRO) + list(_BRANCH[goal]) + list(_TAIL)


# ---- State helpers --------------------------------------------------------

def _init_state() -> None:
    if KEY_ONBOARD_STATE not in st.session_state:
        st.session_state[KEY_ONBOARD_STATE] = {}
    if KEY_ONBOARD_HISTORY not in st.session_state:
        st.session_state[KEY_ONBOARD_HISTORY] = []
    if KEY_ONBOARD_STEP not in st.session_state:
        st.session_state[KEY_ONBOARD_STEP] = 0


def reset_onboarding() -> None:
    for k in (KEY_ONBOARD_STATE, KEY_ONBOARD_HISTORY, KEY_ONBOARD_STEP):
        st.session_state.pop(k, None)


def onboarding_state() -> dict[str, Any]:
    _init_state()
    return st.session_state[KEY_ONBOARD_STATE]


def onboarding_progress() -> tuple[int, int]:
    """(current_question_1_indexed, total). At completion returns (total, total)."""
    _init_state()
    total = len(_current_questions())
    step = st.session_state[KEY_ONBOARD_STEP]
    return (min(step + 1, total), total)


def onboarding_complete() -> bool:
    _init_state()
    return st.session_state[KEY_ONBOARD_STEP] >= len(_current_questions())


# ---- Risk-confirm → risk_answers mapping ---------------------------------

_RISK_ANSWER_MAP: dict[str, list[int]] = {
    "buy":     [3, 3, 3, 3, 3],
    "hold":    [2, 2, 2, 2, 2],
    "partial": [1, 1, 1, 1, 1],
    "move":    [0, 0, 0, 0, 0],
}


def _risk_confirm_label(raw: str) -> str:
    """Human-readable label for the risk_confirm answer (for the context panel)."""
    text = raw.lower().strip()
    if "buy" in text or "add" in text or text in {"d", "(d)"}:
        return "Buy more"
    if "hold" in text or text in {"a", "(a)"}:
        return "Hold steady"
    if "partial" in text or "some" in text or text in {"b", "(b)"}:
        return "Partial shift"
    if ("move" in text or "treasury" in text or "hysa" in text
            or "high-yield" in text or "high yield" in text
            or text in {"c", "(c)"}):
        return "Move to safe assets"
    return raw.strip() or "Partial shift"


def _classify_risk_confirm(raw: str) -> list[int]:
    text = raw.lower()
    if "buy" in text or "add" in text or text.strip() in {"d", "(d)"}:
        return _RISK_ANSWER_MAP["buy"]
    if "hold" in text or text.strip() in {"a", "(a)"}:
        return _RISK_ANSWER_MAP["hold"]
    if "partial" in text or "some" in text or text.strip() in {"b", "(b)"}:
        return _RISK_ANSWER_MAP["partial"]
    if ("move" in text or "treasury" in text or "hysa" in text
            or "high-yield" in text or "high yield" in text
            or text.strip() in {"c", "(c)"}):
        return _RISK_ANSWER_MAP["move"]
    return _RISK_ANSWER_MAP["partial"]


# ---- Bot-side message assembly -------------------------------------------

def next_bot_prompt() -> str | None:
    _init_state()
    step = st.session_state[KEY_ONBOARD_STEP]
    questions = _current_questions()
    if step >= len(questions):
        return None
    q = questions[step]
    try:
        return q.prompt.format(**st.session_state[KEY_ONBOARD_STATE])
    except KeyError:
        return q.prompt


_AGE_HINT_RE = re.compile(
    r"(\d{1,3})\s*(?:y(?:ea)?rs?(?:\s*old)?|yo\b|y/o\b|-?year-?old)",
    re.IGNORECASE,
)
_MONEY_TOKEN_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)(k|m)?|"          # $5,000 / $5k / $1.2m
    r"(?<![a-zA-Z])([\d,]+(?:\.\d+)?)(k|m)\b", # 5k / 1.2m (no $)
    re.IGNORECASE,
)
_PCT_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_YEAR_TOKEN_RE = re.compile(r"\b(20\d{2})\b")


def _money_from(raw: str) -> float | None:
    """First dollar-marked or k/m-suffixed money value in ``raw``.

    Refuses bare integers on purpose — an opportunistic sweep can't tell
    "35" apart from "$35" or "35 years", so we require the user to have
    marked the number as money.
    """
    m = _MONEY_TOKEN_RE.search(raw)
    if not m:
        return None
    num = (m.group(1) or m.group(3)).replace(",", "")
    suffix = (m.group(2) or m.group(4) or "").lower()
    try:
        v = float(num)
    except ValueError:
        return None
    if suffix == "m":
        v *= 1_000_000
    elif suffix == "k":
        v *= 1_000
    return v if v > 0 else None


def _age_from(raw: str, lo: int, hi: int) -> int | None:
    """Age only when the number carries an age-context word."""
    m = _AGE_HINT_RE.search(raw)
    if not m:
        return None
    try:
        age = int(m.group(1))
    except ValueError:
        return None
    return age if lo <= age <= hi else None


# Per-field opportunistic sniffers — each returns a value only when the raw
# text contains a *contextual* signal for that field. Bare numbers alone
# never fill a field (they're ambiguous — age vs. year vs. dollars vs. pct).
# Freetext fields (savings/health/risk_confirm) are intentionally absent:
# they must be answered by their own turn.
def _sniff_field(key: str, raw: str) -> Any:
    text = raw.lower()

    if key == "age":
        return _age_from(raw, 18, 100)

    if key == "income":
        v = _money_from(raw)
        if v is None:
            return None
        # Require a monthly-income context word so we don't grab e.g. a
        # college-cost figure the user volunteered.
        if any(w in text for w in ("month", "/mo", "per mo", "salary",
                                     "income", "earn", "make", "take-home",
                                     "take home", "paycheck")):
            return v
        return None

    if key == "goal_type":
        return _extract_goal(raw)

    if key == "retire_age":
        # Age-shaped number near a retirement keyword. Two shapes: bare
        # ("retire at 60") and full ("retire when I'm 60 years old").
        if not any(w in text for w in ("retire", "retirement", "pension")):
            return None
        full = _age_from(raw, 40, 90)
        if full is not None:
            return full
        m = re.search(r"retir\w*\s+(?:at|by|when\s+i(?:'m|\s*am)?)\s+(\d{2})",
                       text)
        if m:
            n = int(m.group(1))
            if 40 <= n <= 90:
                return n
        return None

    if key == "monthly_need":
        # Require a retirement-context word so a mention of current
        # take-home income doesn't get mis-labelled as retirement need.
        if not any(w in text for w in ("retire", "retirement", "pension",
                                          "after retirement", "post-retire")):
            return None
        # Prefer the money value that appears *after* the retirement word.
        idx = min((text.find(w) for w in
                    ("retire", "retirement", "pension")
                    if text.find(w) >= 0), default=-1)
        if idx < 0:
            return None
        tail = raw[idx:]
        return _money_from(tail)

    if key == "child_current_age":
        # Small integer near a child keyword.
        if not any(w in text for w in ("child", "kid", "son", "daughter",
                                          "baby", "little one")):
            return None
        m = re.search(r"\b(\d{1,2})\b", text)
        if m:
            n = int(m.group(1))
            if 0 <= n <= 17:
                return n
        return None

    if key == "target_cost_today":
        v = _money_from(raw)
        if v is None:
            return None
        if any(w in text for w in ("college", "tuition", "education",
                                     "school", "university")):
            return v
        return None

    if key == "home_price":
        v = _money_from(raw)
        if v is None:
            return None
        if any(w in text for w in ("home", "house", "property", "flat",
                                     "apartment", "condo")):
            return v
        return None

    if key == "down_payment_pct":
        # Percentage explicit; require "down" context to disambiguate from
        # e.g. tax-rate mentions.
        if "down" not in text and "downpayment" not in text:
            return None
        m = _PCT_TOKEN_RE.search(raw)
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                return None
            if 0 <= v <= 100:
                return v
        return None

    if key == "target_purchase_year":
        m = _YEAR_TOKEN_RE.search(raw)
        if m:
            y = int(m.group(1))
            if CURRENT_YEAR <= y <= 2100:
                return y
        m_rel = re.search(r"\bin\s+(\d+)\s*year", text)
        if m_rel:
            return CURRENT_YEAR + int(m_rel.group(1))
        return None

    return None


def _opportunistic_sweep(current_key: str, raw: str,
                            state: dict[str, Any]) -> None:
    """Fill any unfilled onboarding field this message strongly signals.

    Runs after the primary extractor has already consumed ``raw`` for
    ``current_key``. For every other question in the currently-composed
    question list (intro + branch + tail) that doesn't yet have a value,
    call ``_sniff_field``. Non-None hits are written into ``state`` and
    the caller's ``_advance_past_prefilled`` will skip past them on the
    next turn.

    Two ordering notes:
      * If we just answered ``goal_type``, the branch questions are now
        part of ``_current_questions()`` — so branch fields like
        ``home_price`` become sniffable in the same message.
      * ``current_key`` itself is skipped since it's already stored.
    """
    for q in _current_questions():
        if q.key == current_key or q.key in state:
            continue
        value = _sniff_field(q.key, raw)
        if value is not None:
            state[q.key] = value


def _advance_past_prefilled(questions: list["OnboardQ"], state: dict[str, Any]) -> None:
    """Skip forward through questions whose keys already have values.

    Runs after ``submit_answer`` so an opportunistically-captured field
    (e.g. age pulled from the name message) doesn't cause the bot to ask
    the same question again on the next turn.
    """
    step = st.session_state[KEY_ONBOARD_STEP]
    while step < len(questions) and questions[step].key in state:
        step += 1
    st.session_state[KEY_ONBOARD_STEP] = step


def submit_answer(raw: str) -> tuple[bool, str]:
    """Parse a user answer against the current question.

    Returns (accepted, message). ``message`` is either the next bot prompt
    or the extractor's error message. When onboarding completes, ``message``
    is the wrap-up line.

    Two ways an answer is accepted:
      1. The primary extractor (``q.extract``) pulls a value from ``raw``
         — the strict path. Handles clean answers like "35" or "$500k".
      2. If the strict extractor fails, ``_sniff_field`` retries with the
         contextual sniffers used by ``_opportunistic_sweep``. That
         forgives noise around a well-marked answer (e.g. the user
         replies to "how old is your child?" with "she's 5 by the way,
         starting kindergarten next year" — strict fails, contextual
         succeeds since "child"/"she" + "5" match).

    After the primary answer is stored, we run the opportunistic sweep
    across every remaining unfilled field so a single message can fill
    several turns at once ("I'm Peter, 35, want to retire at 60 on $6k/mo").
    """
    _init_state()
    questions = _current_questions()
    step = st.session_state[KEY_ONBOARD_STEP]
    if step >= len(questions):
        return True, ""
    q = questions[step]
    value = q.extract(raw)
    if value is None:
        # Second chance — sniffer tolerates surrounding text.
        value = _sniff_field(q.key, raw)
    if value is None:
        return False, q.error
    state = st.session_state[KEY_ONBOARD_STATE]
    state[q.key] = value
    st.session_state[KEY_ONBOARD_STEP] = step + 1
    # Recompute the question list before the sweep — storing goal_type
    # expands it with the branch questions, which then become sniffable
    # in the same message.
    _opportunistic_sweep(q.key, raw, state)
    _advance_past_prefilled(_current_questions(), state)
    nxt = next_bot_prompt()
    if nxt is None:
        goal = state.get("goal_type", "your plan")
        return True, (
            f"Great, {state.get('name', 'there')}! I have everything I need. "
            f"Let me run your **{goal}** analysis..."
        )
    return True, nxt


# ---- Session-context panel (right rail) ---------------------------------

def context_lines() -> list[tuple[str, str]]:
    """Rows for the SESSION CONTEXT panel (label, value).

    Only lists keys that are relevant to the current (possibly partial)
    journey — questions from an unchosen branch don't leak in.
    """
    _init_state()
    state = st.session_state[KEY_ONBOARD_STATE]
    rows: list[tuple[str, str]] = []
    for q in _current_questions():
        if q.key not in state:
            continue
        raw = state[q.key]
        # Prettify the two freetext tail questions that are otherwise
        # rendered verbatim (risk_confirm 'a'/'b'/'c' and savings sentences).
        if q.key == "savings":
            parsed = _extract_money_from_freetext(str(raw))
            rows.append(("savings", _display_money(parsed) if parsed is not None
                            else "not specified"))
            continue
        if q.key == "risk_confirm":
            label = _risk_confirm_label(str(raw))
            rows.append(("risk_confirm", label))
            continue
        rows.append((q.key, q.display(raw)))
    # ---- Derived rows (small conveniences) --------------------------
    goal = state.get("goal_type")
    if goal == "Retirement Planning" and "age" in state and "retire_age" in state:
        rows.insert(
            _index_after(rows, "retire_age"),
            ("years_left", str(int(state["retire_age"]) - int(state["age"]))),
        )
    elif goal == "Child Education" and "child_current_age" in state:
        rows.insert(
            _index_after(rows, "child_current_age"),
            ("years_to_college", str(18 - int(state["child_current_age"]))),
        )
    elif goal == "Buy Home" and "target_purchase_year" in state:
        rows.insert(
            _index_after(rows, "target_purchase_year"),
            ("years_to_buy",
                str(int(state["target_purchase_year"]) - CURRENT_YEAR)),
        )
    return rows


def _index_after(rows: list[tuple[str, str]], key: str) -> int:
    for i, (k, _v) in enumerate(rows):
        if k == key:
            return i + 1
    return len(rows)


# ---- Commit — build customer + kick pipeline ----------------------------

def commit_customer_and_run() -> int:
    """Persist a Customer from the collected answers and return its id.

    Branches on ``goal_type`` to build the right ``goal_inputs`` shape for
    the downstream planner. ``risk_answers`` is synthesised from the single
    ``risk_confirm`` question so the deterministic risk agent still runs.
    """
    _init_state()
    state = st.session_state[KEY_ONBOARD_STATE]

    monthly_income = float(state["income"])
    annual_income = monthly_income * 12
    risk_answers = _classify_risk_confirm(str(state.get("risk_confirm", "partial")))
    goal = state.get("goal_type", "Retirement Planning")

    # Pull a numeric current_savings out of the tail 'savings' freetext when
    # one is present (e.g. "I have $100,000 in a brokerage"). Falls back to
    # per-journey defaults if we can't parse a number.
    savings_raw = str(state.get("savings", ""))
    parsed_savings = _extract_money_from_freetext(savings_raw)

    if goal == "Retirement Planning":
        savings_default = 50_000.0
    elif goal == "Child Education":
        savings_default = 5_000.0
    else:
        savings_default = 20_000.0
    current_savings = parsed_savings if parsed_savings is not None else savings_default

    goal_inputs: dict[str, Any] = {"current_savings": current_savings}

    if goal == "Retirement Planning":
        goal_inputs.update({
            "target_retirement_age": int(state["retire_age"]),
            "desired_monthly_income": float(state["monthly_need"]),
            "monthly_contribution": max(monthly_income * 0.15, 1_000.0),
        })
    elif goal == "Child Education":
        goal_inputs.update({
            "child_current_age": int(state["child_current_age"]),
            "target_cost_today": float(state["target_cost_today"]),
            "monthly_contribution": max(monthly_income * 0.10, 500.0),
        })
    elif goal == "Buy Home":
        goal_inputs.update({
            "home_price": float(state["home_price"]),
            "down_payment_pct": float(state["down_payment_pct"]),
            "target_purchase_year": int(state["target_purchase_year"]),
            "current_year": CURRENT_YEAR,
            "monthly_saving_capacity": max(monthly_income * 0.15, 1_000.0),
        })

    existing_ids = {c.external_id for c in list_customers()}
    base = "N" + str(len(existing_ids) + 1).zfill(3)
    external_id = base
    i = 1
    while external_id in existing_ids:
        i += 1
        external_id = f"{base}-{i}"

    customer = Customer(
        id=None,
        external_id=external_id,
        name=str(state["name"]),
        age=int(state["age"]),
        annual_income=annual_income,
        dependents=0,
        risk_answers=risk_answers,
        primary_goal=goal,
        goal_inputs=goal_inputs,
    )
    new_id = upsert_customer(customer)
    set_active_customer(new_id)
    st.session_state.pop(KEY_LAST_PIPELINE, None)
    return new_id
