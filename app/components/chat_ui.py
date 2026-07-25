"""Shared chat renderer used by Home, FinAdvisor, and the floating panel.

Two responsibilities:
  1. Render a chat history + input (identical shape across surfaces).
  2. Render a right-side "session context" rail (customer parameters,
     intake progress, API trace) — the layout shown in the reference
     screenshots.

The Q&A/change-detection turn handler still lives in floating_chat.py so
this module stays UI-only. Onboarding turn handling lives in
components/onboarding.py.
"""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


def _escape_and_break(text: str) -> str:
    """HTML-escape user text and preserve newlines for the right-side bubble."""
    return escape(text or "").replace("\n", "<br>")


def render_history(history: list[dict[str, str]],
                     scroll_key: str = "nw_chat_scroll") -> None:
    """Replay bubbles from a role/content list.

    Renders inside a keyed container so ``theme._CSS`` can pin a fixed
    height + ``overflow-y:auto`` on it — that keeps a long chat scrollable
    inside its own frame rather than pushing the input row off-screen.
    Callers can pass a distinct ``scroll_key`` (e.g. ``"nw_chat_scroll_fc"``
    for the floating panel) if they need a different height.

    After rendering, injects a small JS snippet that scrolls the frame to
    its bottom so the newest message is always visible without the user
    having to scroll manually. Runs from the parent document (Streamlit
    renders these bubbles in the top-level DOM, not the components iframe),
    which is why we use ``window.parent.document`` inside a components.html
    block that itself lives in an iframe.
    """
    with st.container(key=scroll_key):
        for turn in history:
            if turn["role"] == "user":
                # Render user turns as a right-aligned bubble (theme._CSS
                # styles .nw-chat-user + .nw-chat-user-bubble). Skipping
                # st.chat_message here keeps the alignment independent of
                # Streamlit's internal chat DOM, which shifts between
                # versions.
                content_html = _escape_and_break(turn["content"])
                st.markdown(
                    f'<div class="nw-chat-user">'
                    f'<div class="nw-chat-user-bubble">{content_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])

    # Length is folded into the JS so Streamlit gives us a fresh iframe on
    # every new turn — without it, the browser caches the previous script
    # and the effect never fires again after the first render.
    _scroll_to_bottom(scroll_key, len(history))


def _scroll_to_bottom(scroll_key: str, turn_count: int) -> None:
    import streamlit.components.v1 as components  # local: avoids cost on import
    components.html(
        f"""
        <script>
        (function() {{
            const key = "{scroll_key}";
            const nonce = {turn_count};  // forces re-run on every new turn
            const doc = window.parent.document;
            // The keyed container's DOM class is `st-key-<key>`.
            const el = doc.querySelector('.st-key-' + key);
            if (el) {{
                el.scrollTop = el.scrollHeight;
                // Some content (LLM markdown, code blocks) settles a tick
                // after mount — do it once more after a short delay so
                // late-loading layout doesn't leave us short.
                setTimeout(() => {{ el.scrollTop = el.scrollHeight; }}, 60);
            }}
        }})();
        </script>
        """,
        height=0,
    )


def render_context_panel(
    title: str,
    intake_label: str,
    intake_percent: int,
    parameters: list[tuple[str, str]],
    api_trace: list[str] | None = None,
) -> None:
    """Right-side session-context rail.

    Layout mirrors the reference: SESSION CONTEXT header, INTAKE PROGRESS
    bar, CURRENT PARAMETERS list, and (optionally) an API TRACE code block.
    """
    st.markdown(f"##### {title}")
    st.caption("INTAKE PROGRESS")
    st.progress(min(max(intake_percent, 0), 100) / 100.0, text=intake_label)

    st.markdown("###### CURRENT PARAMETERS")
    if not parameters:
        st.caption("_Nothing captured yet — start the chat to populate this._")
    else:
        for label, value in parameters:
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"padding:2px 0; font-size:13px;'>"
                f"<span style='color:#6B7280;'>{label}</span>"
                f"<span style='color:#1F2937; font-weight:600;'>{value}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if api_trace:
        st.markdown("###### API TRACE")
        st.code("\n".join(api_trace), language="text")


def push_bot(history: list[dict[str, str]], content: str) -> None:
    history.append({"role": "assistant", "content": content})


def push_user(history: list[dict[str, str]], content: str) -> None:
    history.append({"role": "user", "content": content})
