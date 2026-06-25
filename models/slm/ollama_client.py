"""
Shared Ollama client — one place both AI macro callers go through.

Works with local Ollama (default) and Ollama Cloud. When OLLAMA_API_KEY is set,
requests carry an Authorization: Bearer header and target the configured base URL
(e.g. Ollama Cloud running Nemotron 3 Ultra). Base URL and model come from the
active provider config (intelligence/provider_config.py), falling back to env.

Used by intelligence/macro_context.py and models/slm/infer.py so their behavior
never drifts apart.
"""
from __future__ import annotations

import logging

import requests

from config import settings
from intelligence.provider_config import get_active

log = logging.getLogger(__name__)


def _base_url() -> str:
    return (get_active().get("macro", {}).get("base_url") or settings.OLLAMA_URL).rstrip("/")


def _model() -> str:
    return get_active().get("macro", {}).get("model") or settings.SLM_MODEL


def _headers() -> dict:
    key = settings.OLLAMA_API_KEY
    return {"Authorization": f"Bearer {key}"} if key else {}


def ollama_available(timeout: int = 3) -> bool:
    """True if the configured Ollama endpoint (local or cloud) answers."""
    try:
        r = requests.get(f"{_base_url()}/api/tags", headers=_headers(), timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def ollama_generate(
    prompt: str,
    system: str = "",
    options: dict | None = None,
    *,
    format_json: bool = False,
    warm_up: bool = False,
    timeout: int = 60,
) -> str:
    """
    Single non-streaming generate call. Returns the response text ('' on failure).
    `warm_up` first pokes the model so a cold load doesn't eat the real call's clock
    (useful for large local models). `format_json` asks Ollama for JSON output.
    """
    url = f"{_base_url()}/api/generate"
    model = _model()
    headers = _headers()

    if warm_up:
        try:
            requests.post(
                url, headers=headers,
                json={"model": model, "prompt": "ok", "stream": False,
                      "options": {"num_predict": 1}},
                timeout=timeout,
            )
        except Exception as e:
            log.warning("Ollama warm-up failed: %s", e)

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options or {"temperature": 0.3},
    }
    if system:
        payload["system"] = system
    if format_json:
        payload["format"] = "json"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("response", "")
        log.warning("Ollama generate HTTP %s (%s)", resp.status_code, resp.text[:200])
    except Exception as e:
        log.warning("Ollama generate failed: %s", e)
    return ""
