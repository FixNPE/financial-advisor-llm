"""LLM client wrapping Hugging Face Inference Providers (OpenAI-compatible)."""
from __future__ import annotations

import logging
import os

from huggingface_hub import InferenceClient

from advisor.config import Settings, settings

log = logging.getLogger(__name__)

_client: InferenceClient | None = None
_client_key: tuple[str, str, str] | None = None  # (provider, model, token-fingerprint)

# The most recent LLM failure reason — set by chat()/chat_text() when the
# HF call raises. Consumed by advisor._fallback_answer to render a specific
# banner ("401 unauthorized" / "402 payment required" / "connection error")
# instead of the generic "LLM temporarily unavailable" line. Cleared on the
# next successful call.
_last_error: str | None = None


def _fingerprint(token: str) -> str:
    """Short suffix of the token, safe to log — never the full secret."""
    return f"...{token[-6:]}" if token else "(empty)"


def _refresh_settings() -> Settings:
    """Re-read .env so a token rotation is picked up without a process restart.

    ``pydantic-settings`` reads env + .env at instantiation, so ``settings``
    (module-level singleton) freezes whatever was on disk at first import.
    We reinstantiate on every call — cheap (a few file reads) and it makes
    the standard workflow (edit .env, rerun a chat turn) actually work.
    """
    return Settings()


def get_client() -> InferenceClient:
    """Return a cached client, rebuilding it if provider/model/token changed.

    The old implementation cached the very first client forever, so a
    rotated HF_TOKEN never took effect until the process restarted. We
    now key the cache on (provider, model, token) and rebuild whenever
    any of them changes — os.environ takes precedence over .env, matching
    what ``pydantic-settings`` reads.
    """
    global _client, _client_key
    fresh = _refresh_settings()
    token = os.environ.get("HF_TOKEN") or fresh.hf_token
    provider = os.environ.get("LLM_PROVIDER") or fresh.llm_provider
    model = os.environ.get("LLM_MODEL_ID") or fresh.llm_model_id
    key = (provider, model, token)
    if _client is None or _client_key != key:
        log.info("Building new HF InferenceClient provider=%s model=%s token=%s",
                 provider, model, _fingerprint(token))
        _client = InferenceClient(provider=provider, api_key=token)
        _client_key = key
    return _client


def last_error() -> str | None:
    """The reason for the most recent LLM failure, if any. See module docstring."""
    return _last_error


def _classify_error(exc: BaseException) -> str:
    """Best-effort human label for an HF InferenceClient error."""
    msg = str(exc)
    lower = msg.lower()
    if "401" in msg or "unauthorized" in lower or "invalid" in lower and "token" in lower:
        return "HF token rejected (401) — check HF_TOKEN in .env"
    if "402" in msg or "payment required" in lower or "insufficient" in lower and "credit" in lower:
        return "HF Inference credit exhausted (402)"
    if "403" in msg or "forbidden" in lower:
        return "HF token lacks access to this model/provider (403)"
    if "404" in msg or "not found" in lower:
        return "Model or provider route not found (404)"
    if "429" in msg or "rate limit" in lower or "too many requests" in lower:
        return "Rate-limited by HF Inference (429)"
    if "timeout" in lower or "timed out" in lower:
        return "LLM request timed out"
    if "connection" in lower or "network" in lower:
        return "Network error reaching HF Inference"
    # Fall back to the exception type + a truncated message so the UI
    # banner is at least informative.
    return f"{type(exc).__name__}: {msg[:160]}"


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """Run a chat completion. Returns the raw OpenAI-shaped response object."""
    global _last_error
    fresh = _refresh_settings()
    kwargs: dict = {
        "model": os.environ.get("LLM_MODEL_ID") or fresh.llm_model_id,
        "messages": messages,
        "temperature": temperature if temperature is not None else fresh.llm_temperature,
        "max_tokens": max_tokens if max_tokens is not None else fresh.llm_max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    try:
        resp = get_client().chat.completions.create(**kwargs)
        _last_error = None
        return resp
    except Exception as exc:  # noqa: BLE001 — every failure funnels through _last_error
        _last_error = _classify_error(exc)
        log.warning("LLM call failed: %s", _last_error)
        raise


def chat_text(messages: list[dict], **kwargs) -> str:
    """Convenience wrapper that returns only the assistant text."""
    resp = chat(messages, **kwargs)
    return resp.choices[0].message.content or ""
