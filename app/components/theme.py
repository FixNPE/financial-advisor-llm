"""NexWealth AI theme — global CSS + shared sidebar renderer.

The theme is dark-navy sidebar + coral accent + light neutral body,
matching the reference product. All pages call ``apply_theme()`` at the
top; that both injects the CSS and renders the shared sidebar (brand,
customer chip, in-app nav, agent pipeline, LLM status footer).
"""
from __future__ import annotations

import streamlit as st

from advisor.config import settings
from advisor.domain.data import Customer

from app.components.session import (
    KEY_ACTIVE_CUSTOMER, active_customer, customer_options,
    provider_label, set_active_customer,
)

BRAND_NAME = "NexWealth AI"
PRODUCT_NAME = "FinAdvisor"
PRODUCT_TAGLINE = "The AI-Native Wealth Management Experience"
PRODUCT_SUBTITLE = "Plan, review, and approve investment decisions right in the chat panel."

NAV_ITEMS: list[tuple[str, str]] = [
    ("Home",                    "pages/0_Home.py"),
    ("FinAdvisor",              "pages/1_FinAdvisor.py"),
    ("Dashboard",               "pages/2_Dashboard.py"),
    ("Risk Profile",            "pages/3_Risk_Profile.py"),
    ("Recommendations",         "pages/4_Recommendations.py"),
    ("Report",                  "pages/5_Report.py"),
    ("RAG Explorer",            "pages/6_RAG_Explorer.py"),
]

_PIPELINE_STEPS = [
    "Intent Classification",
    "Safety Guardrails",
    "Agent Orchestrator",
    "Goal Planning",
    "Risk Profiling",
    "Portfolio Analysis",
    "Benchmarking",
    "Recommendation",
    "Report / Output",
]

_CSS = """
<style>
:root {
    --nw-navy: #0F1A2C;
    --nw-navy-2: #182742;
    --nw-navy-3: #223357;
    --nw-blue: #2E5EFF;
    --nw-blue-hover: #1E48CC;
    --nw-blue-soft: #EEF3FF;
    --nw-blue-border: #C7D5FA;
    --nw-canvas: #F3F6FC;
    --nw-card: #FFFFFF;
    --nw-card-border: #D9E2F5;
    --nw-ink: #0F1A2C;
    --nw-muted: #6B7280;
}

/* Body canvas — soft blue behind everything on the right */
html, body, .stApp,
section[data-testid="stAppViewContainer"],
section[data-testid="stMain"] > div,
div[data-testid="stMainBlockContainer"] {
    background-color: var(--nw-canvas) !important;
}
header[data-testid="stHeader"] {
    background-color: var(--nw-canvas) !important;
}

/* Elevate common containers into white cards so they read against the blue canvas.
   Scope to main content — sidebar overrides these below. */
section[data-testid="stMain"] div[data-testid="stForm"],
section[data-testid="stMain"] div[data-testid="stExpander"],
section[data-testid="stMain"] div[data-testid="stDataFrame"],
section[data-testid="stMain"] div[data-testid="stTable"],
section[data-testid="stMain"] div[data-testid="stAlert"],
section[data-testid="stMain"] div[data-testid="stMetric"],
section[data-testid="stMain"] div[data-testid="stTextArea"] textarea,
section[data-testid="stMain"] div[data-testid="stTextInput"] input,
section[data-testid="stMain"] div[data-testid="stNumberInput"] input,
section[data-testid="stMain"] div[data-baseweb="select"] > div {
    background-color: var(--nw-card) !important;
    border-radius: 10px !important;
}
section[data-testid="stMain"] div[data-testid="stForm"],
section[data-testid="stMain"] div[data-testid="stExpander"],
section[data-testid="stMain"] div[data-testid="stMetric"] {
    border: 1px solid var(--nw-card-border) !important;
    padding: 6px 12px !important;
    box-shadow: 0 1px 2px rgba(15,26,44,0.04);
}
section[data-testid="stMain"] div[data-testid="stForm"] { padding: 14px 18px !important; }
section[data-testid="stMain"] div[data-testid="stMetric"] { padding: 12px 14px !important; }

/* Plotly chart panels look part of the card family */
section[data-testid="stMain"] div[data-testid="stPlotlyChart"] {
    background-color: var(--nw-card) !important;
    border: 1px solid var(--nw-card-border) !important;
    border-radius: 10px !important;
    padding: 6px !important;
    box-shadow: 0 1px 2px rgba(15,26,44,0.04);
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: var(--nw-navy) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div {
    background-color: var(--nw-navy) !important;
}
section[data-testid="stSidebar"] * {
    color: #EAEEF7 !important;
}
section[data-testid="stSidebar"] a { color: #EAEEF7 !important; }

/* Sidebar customer picker (selectbox) — dark navy field with white text.
   Without this, the field inherits the sidebar's forced light color and reads as
   pale text on a near-white background. */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #0A1220 !important;
    border: 1px solid #223357 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    min-height: 36px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div * {
    color: #ffffff !important;
    background-color: transparent !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #C2CBDB !important;
    color: #C2CBDB !important;
}
/* Popover options escape stSidebar in the DOM, so also give them readable ink. */
ul[data-baseweb="menu"], ul[data-baseweb="menu"] li {
    color: #0F1A2C !important;
    background-color: #FFFFFF !important;
}

/* Hide the default Streamlit multipage nav — we render our own */
section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
    display: none;
}

/* Brand block */
.nw-brand-row {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 4px 6px 4px;
}
.nw-brand-logo {
    width: 30px; height: 30px; border-radius: 7px;
    background: linear-gradient(135deg, #6EA8FE 0%, #2E5EFF 100%);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 700; font-size: 14px;
}
.nw-brand-name {
    font-size: 20px; font-weight: 700;
    color: #ffffff !important; letter-spacing: 0.2px;
}
.nw-brand-tag {
    color: #C2CBDB !important; font-size: 12px;
    padding: 2px 4px 0 4px; line-height: 1.35;
}
.nw-brand-product {
    color: #ffffff !important; font-size: 12.5px; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 4px 4px 2px 4px;
}
.nw-brand-sub {
    color: #93A1BC !important; font-size: 11.5px;
    padding: 4px 4px 12px 4px; line-height: 1.4;
}

/* Section headings inside the dark sidebar */
.nw-section-label {
    color: #93A1BC !important;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 14px 4px 6px 4px;
}

/* Active-customer chip */
.nw-chip {
    background: #0A1220; border: 1px solid #223357; border-radius: 8px;
    padding: 10px 12px; margin: 4px 0 8px 0;
}
.nw-chip .lbl {
    color: #93A1BC !important; font-size: 11px;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.nw-chip .name { color: #ffffff !important; font-weight: 600; font-size: 14px; padding-top: 2px; }
.nw-chip .meta { color: #C2CBDB !important; font-size: 12px; padding-top: 2px; }

/* Nav rows — tighten vertical spacing between items. Streamlit's stVerticalBlock
   uses a CSS var (--gap) that defaults to 1rem; shrinking the surrounding wrappers
   pulls the nav items close together. */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0.15rem !important;
}
section[data-testid="stSidebar"] div.nw-nav-active,
section[data-testid="stSidebar"] div.nw-nav-idle {
    margin: 0 !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stElementContainer"] {
    margin: 0 !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] {
    margin: 0 !important;
    padding: 0 !important;
}
/* Streamlit wraps button content in <div><p>, so we left-align at every level. */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: transparent !important;
    border: none !important;
    color: #EAEEF7 !important;
    padding: 4px 10px !important;
    margin: 1px 0 !important;
    width: 100%;
    min-height: 0 !important;
    line-height: 1.35 !important;
    display: flex !important;
    justify-content: flex-start !important;
    text-align: left !important;
    font-weight: 400 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button > div {
    padding: 0 !important; margin: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button > div > p {
    margin: 0 !important; padding: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button > div,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button > div > p {
    text-align: left !important;
    width: 100% !important;
    color: #EAEEF7 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: #182742 !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover > div > p {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] div.nw-nav-active div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div.nw-nav-active div[data-testid="stButton"] > button > div > p {
    background: #182742 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Sidebar expander — used for the collapsible "Agent pipeline" panel */
section[data-testid="stSidebar"] div[data-testid="stExpander"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin-top: 10px !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] > details {
    background: transparent !important;
    border: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
    background: #0A1220 !important;
    border: 1px solid #223357 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    color: #EAEEF7 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    list-style: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover {
    background: #182742 !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary svg {
    fill: #C2CBDB !important;
    color: #C2CBDB !important;
}

/* Pipeline block */
.nw-pipeline {
    background: #0A1220; border: 1px solid #223357; border-radius: 8px;
    padding: 10px 12px; margin-top: 6px;
}
.nw-pipeline .title {
    color: #93A1BC !important; font-size: 11px;
    letter-spacing: 0.06em; text-transform: uppercase; padding-bottom: 6px;
}
.nw-pipeline .step {
    color: #C2CBDB !important; font-size: 12px; padding: 2px 0;
    display: flex; align-items: center; gap: 6px;
}
.nw-pipeline .step::before {
    content: "›"; color: var(--nw-blue); font-weight: 700;
}

/* LLM status + disclaimer */
.nw-footer-status {
    color: #C2CBDB !important; font-size: 11.5px;
    padding: 12px 4px 2px 4px; border-top: 1px solid #223357; margin-top: 14px;
}
.nw-footer-disclaimer {
    color: #7A8AA9 !important; font-size: 11px;
    padding: 4px 4px 12px 4px; font-style: italic;
}

/* Body — primary CTA is a blue pill */
button[kind="primary"], .stButton > button[kind="primary"] {
    background-color: var(--nw-blue) !important;
    color: #fff !important; border: none !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover {
    background-color: var(--nw-blue-hover) !important;
}

/* Chat quick-start pills on FinAdvisor / Home */
.nw-quickstart-row div[data-testid="stButton"] > button {
    background: #ffffff !important;
    color: var(--nw-ink) !important;
    border: 1px solid #D4DEEF !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    text-align: center !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 2px rgba(15,26,44,0.04);
}
.nw-quickstart-row div[data-testid="stButton"] > button:hover {
    border-color: var(--nw-blue) !important;
    color: var(--nw-blue) !important;
}

/* Persona / AI assistant intro card — blue theme */
.nw-persona {
    background: var(--nw-blue-soft);
    border: 1px solid var(--nw-blue-border);
    border-radius: 12px;
    padding: 14px 16px;
    display: flex; align-items: flex-start; gap: 12px;
    margin: 12px 0 8px 0;
}
.nw-persona .avatar {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #6EA8FE 0%, #2E5EFF 100%);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 18px; flex-shrink: 0;
}
.nw-persona .who { font-weight: 600; color: var(--nw-ink); font-size: 14px; }
.nw-persona .msg { color: #3F4A63; font-size: 13.5px; line-height: 1.5; padding-top: 2px; }

/* Section header used on Home / FinAdvisor */
.nw-hero-title {
    font-size: 26px; font-weight: 700; color: var(--nw-ink);
    padding-bottom: 4px;
}
.nw-hero-sub {
    color: #4A5468; font-size: 14px; padding-bottom: 12px;
}

/* Cards on Home / Risk Profile */
.nw-card {
    background: #ffffff; border: 1px solid #D4DEEF; border-radius: 12px;
    padding: 16px 18px; box-shadow: 0 1px 2px rgba(15,26,44,0.04);
}

/* -------- Floating "Ask FinAdvisor" FAB --------
   Streamlit auto-adds an `st-key-<key>` class to every keyed widget's DOM
   node — that's the officially supported hook. We give the FAB button
   key="nw_fab" and use .st-key-nw_fab to fix-position that container to the
   bottom-right of the viewport regardless of scroll. */
.st-key-nw_fab {
    position: fixed !important;
    right: 28px !important;
    bottom: 28px !important;
    z-index: 9999 !important;
    width: auto !important;
    max-width: 260px !important;
    margin: 0 !important;
    padding: 0 !important;
}
.st-key-nw_fab div[data-testid="stButton"] > button,
.st-key-nw_fab button {
    background: var(--nw-blue) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 12px 22px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    box-shadow: 0 10px 24px rgba(46, 94, 255, 0.35),
                    0 2px 6px rgba(15, 26, 44, 0.18) !important;
    min-height: 48px !important;
    width: auto !important;
}
.st-key-nw_fab div[data-testid="stButton"] > button:hover,
.st-key-nw_fab button:hover {
    background: var(--nw-blue-hover) !important;
    box-shadow: 0 12px 28px rgba(46, 94, 255, 0.42),
                    0 3px 8px rgba(15, 26, 44, 0.22) !important;
}

/* -------- Docked right-side chat panel --------
   Panel is a st.container(key="nw_chat_panel", border=True). Streamlit
   emits the container's wrapper with class `st-key-nw_chat_panel`. */
.st-key-nw_chat_panel {
    position: fixed !important;
    top: 72px !important;
    right: 24px !important;
    bottom: 24px !important;
    width: 440px !important;
    max-width: calc(100vw - 48px) !important;
    background: var(--nw-card) !important;
    border: 1px solid var(--nw-card-border) !important;
    border-radius: 14px !important;
    box-shadow: -6px 0 24px rgba(15, 26, 44, 0.14),
                    0 4px 12px rgba(15, 26, 44, 0.10) !important;
    z-index: 9998 !important;
    padding: 14px 16px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    margin: 0 !important;
}

/* Shift main content left when the panel exists on the page. */
section[data-testid="stMain"]:has(.st-key-nw_chat_panel) div[data-testid="stMainBlockContainer"] {
    padding-right: 480px !important;
}

/* Tighten default 1rem block gap inside the panel. */
.st-key-nw_chat_panel div[data-testid="stVerticalBlock"] {
    gap: 0.35rem !important;
}

/* Scrollable chat history — keeps a long conversation inside its own frame
   so the input row stays anchored below and doesn't get pushed off-screen.
   render_history() wraps its bubbles in st.container(key="nw_chat_scroll")
   (or nw_chat_scroll_fc inside the floating panel, which is already a
   smaller frame). */
.st-key-nw_chat_scroll {
    max-height: 65vh !important;
    overflow-y: auto !important;
    padding-right: 6px !important;
    scrollbar-width: thin;
}
.st-key-nw_chat_scroll_fc {
    max-height: 50vh !important;
    overflow-y: auto !important;
    padding-right: 6px !important;
    scrollbar-width: thin;
}
.st-key-nw_chat_scroll::-webkit-scrollbar,
.st-key-nw_chat_scroll_fc::-webkit-scrollbar {
    width: 6px;
}
.st-key-nw_chat_scroll::-webkit-scrollbar-thumb,
.st-key-nw_chat_scroll_fc::-webkit-scrollbar-thumb {
    background: var(--nw-card-border);
    border-radius: 3px;
}

/* Chat bubble alignment — bot on the left, user on the right.
   render_history() renders bot turns via st.chat_message("assistant") and
   user turns via a plain markdown block wrapped in .nw-chat-user so we
   don't depend on Streamlit's internal chat DOM. The bot bubble gets a
   soft card background; the user bubble is right-aligned with a filled
   blue background, mirroring standard chat conventions. */
.nw-chat-user {
    display: flex;
    justify-content: flex-end;
    margin: 6px 0;
}
.nw-chat-user-bubble {
    background: var(--nw-blue, #2563EB);
    color: #FFFFFF;
    border-radius: 14px 14px 4px 14px;
    padding: 8px 14px;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.45;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.nw-chat-user-bubble a { color: #E5EDFF; text-decoration: underline; }

[data-testid="stChatMessage"] {
    background: transparent !important;
    padding-left: 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    background: var(--nw-card, #FFFFFF);
    border: 1px solid var(--nw-card-border, #E5E7EB);
    border-radius: 14px 14px 14px 4px;
    padding: 8px 14px;
    max-width: 78%;
}

/* Panel header title row */
.nw-chat-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 2px 0 8px 0; border-bottom: 1px solid var(--nw-card-border);
    margin-bottom: 6px;
}
.nw-chat-title {
    font-size: 15px; font-weight: 700; color: var(--nw-ink);
    display: flex; align-items: center; gap: 8px;
}
.nw-chat-title .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--nw-blue);
}

/* Close button — keyed nw_chat_close, absolute-positioned in the panel header. */
.st-key-nw_chat_close {
    position: absolute !important;
    top: 14px !important;
    right: 14px !important;
    z-index: 2 !important;
    width: auto !important;
    margin: 0 !important;
    padding: 0 !important;
}
.st-key-nw_chat_close button {
    background: transparent !important;
    color: var(--nw-muted) !important;
    border: 1px solid var(--nw-card-border) !important;
    border-radius: 8px !important;
    padding: 2px 12px !important;
    min-height: 30px !important;
    font-size: 13px !important;
    line-height: 1 !important;
    box-shadow: none !important;
    width: auto !important;
    min-width: 0 !important;
}
.st-key-nw_chat_close button:hover {
    background: var(--nw-blue-soft) !important;
    color: var(--nw-ink) !important;
    border-color: var(--nw-blue-border) !important;
}
</style>
"""


def apply_theme(page_key: str | None = None) -> Customer | None:
    """Inject the theme CSS and render the shared sidebar. Returns active customer.

    ``page_key`` is one of the NAV_ITEMS labels ("Home", "FinAdvisor", ...) and
    controls which item is highlighted as active in the sidebar nav.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
    _render_sidebar(page_key)
    return active_customer()


def _render_sidebar(page_key: str | None) -> None:
    with st.sidebar:
        # Brand row — NexWealth AI is the brand, FinAdvisor is the product surface.
        st.markdown(
            f"""
            <div class="nw-brand-row">
              <div class="nw-brand-logo">◆</div>
              <div class="nw-brand-name">{BRAND_NAME}</div>
            </div>
            <div class="nw-brand-product">Product · {PRODUCT_NAME}</div>
            <div class="nw-brand-tag">{PRODUCT_TAGLINE}</div>
            <div class="nw-brand-sub">{PRODUCT_SUBTITLE}</div>
            """,
            unsafe_allow_html=True,
        )

        _customer_chip_and_picker()
        _sidebar_nav(page_key)
        _pipeline_block()
        _sidebar_footer()


def _customer_chip_and_picker() -> None:
    customers = customer_options()
    st.markdown('<div class="nw-section-label">Active customer</div>', unsafe_allow_html=True)

    if not customers:
        st.markdown(
            '<div class="nw-chip"><div class="lbl">None</div>'
            '<div class="meta">Run <code>make seed</code> to load hero customers.</div></div>',
            unsafe_allow_html=True,
        )
        return

    label_by_id = {c.id: c.name for c in customers}
    current_id = st.session_state.get(KEY_ACTIVE_CUSTOMER)
    if current_id not in label_by_id:
        current_id = customers[0].id

    # Streamlit selectboxes with an explicit `key` remember their own value
    # across reruns and ignore `index=` once that key exists — so a
    # programmatic change to the active customer (e.g. from onboarding)
    # would otherwise be silently overridden back to the widget's stale
    # stored value on the very next render. Force the widget's own state to
    # follow the true active customer before instantiating it.
    if st.session_state.get("sidebar_customer_selectbox") != current_id:
        st.session_state["sidebar_customer_selectbox"] = current_id

    active = next((c for c in customers if c.id == current_id), customers[0])
    st.markdown(
        f"""
        <div class="nw-chip">
          <div class="lbl">Active</div>
          <div class="name">{active.name}</div>
          <div class="meta">{active.country} · {active.currency} · #{active.external_id}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ids = list(label_by_id.keys())
    idx = ids.index(current_id) if current_id in ids else 0
    chosen_id = st.selectbox(
        "Switch customer",
        options=ids,
        format_func=lambda i: label_by_id[i],
        index=idx,
        label_visibility="collapsed",
        key="sidebar_customer_selectbox",
    )
    if chosen_id != st.session_state.get(KEY_ACTIVE_CUSTOMER):
        set_active_customer(chosen_id)
        st.rerun()


def _sidebar_nav(page_key: str | None) -> None:
    st.markdown('<div class="nw-section-label">Navigate</div>', unsafe_allow_html=True)
    for label, target in NAV_ITEMS:
        is_active = (label == page_key)
        wrapper_cls = "nw-nav-active" if is_active else "nw-nav-idle"
        bullet = "●" if is_active else "○"
        st.markdown(f'<div class="{wrapper_cls}">', unsafe_allow_html=True)
        if st.button(f"{bullet}  {label}", key=f"nav_{label}", use_container_width=True):
            if not is_active:
                st.switch_page(target)
        st.markdown("</div>", unsafe_allow_html=True)


def _pipeline_block() -> None:
    steps_html = "".join(f'<div class="step">{s}</div>' for s in _PIPELINE_STEPS)
    with st.expander("Agent pipeline", expanded=False):
        st.markdown(
            f"""
            <div class="nw-pipeline">
              <div class="title">Order of operations</div>
              {steps_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _sidebar_footer() -> None:
    if settings.llm_provider == "none" or not settings.hf_token:
        llm_line = "LLM: Rule-based fallback (no LLM configured)"
    else:
        llm_line = f"LLM: {settings.llm_provider} · {settings.llm_model_id.split('/')[-1]}"
    st.markdown(
        f"""
        <div class="nw-footer-status">{llm_line}</div>
        <div class="nw-footer-disclaimer">Educational demo — not financial advice.</div>
        """,
        unsafe_allow_html=True,
    )
    # Kept for callers that still reference provider_label()
    _ = provider_label
