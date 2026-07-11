"""
Provider configuration — which LLM engines power the two AI layers, plus the
market-data provider chain.

Two LLM layers are pluggable:
  - macro:     local Ollama or Ollama Cloud (Nemotron 3 Ultra). See models/slm/ollama_client.py.
  - synthesis: a bring-your-own CLI (claude | codex | gemini). See intelligence/llm_cli.py.

The user's choice is stored in the app_config table (data/db/schema_v5_providers.sql) and
chosen via the first-run modal. The synchronous LLM clients can't do async DB reads mid-call,
so the live choice is held in a module cache that the pipeline refreshes at the start of each
run (load_provider_config) and the backend refreshes whenever the config changes. Until loaded,
get_active() falls back to environment defaults so standalone scripts still work.

No secrets here: the Ollama Cloud key lives in .env only; we persist has_key, never the key.

Market-data provider chain (not LLM-related, but same "pluggable provider" shape):
  1. Upstox   — primary live feed once wired (data/pipeline/upstox_client.py / upstox_feed.py).
  2. Angel One — explicit fallback only. See backend/routers/market_data.py (index quotes,
                 with its own circuit breaker) and data/pipeline/fetch_angel_daily.py
                 (EOD backfill fallback inside scheduler/market_runner.py's
                 task_incremental_ohlcv). Angel One is IP-bound and frequently blocked on
                 home networks — it exists purely as a safety net, never call it as primary.
  3. None     — if both are unavailable, market-data endpoints degrade gracefully
                 (available=false) rather than raising.
  Groww's intraday snapshot path is retired/archived (see WHAT_TO_DO_NEXT.txt Section 2.8)
  and is NOT part of this chain — its code remains in data/pipeline/fetch_groww.py, unused,
  in case of resubscription.
"""
from __future__ import annotations

import logging

from config import settings

log = logging.getLogger(__name__)

SYNTH_BACKENDS = ("claude", "codex", "gemini")

# Market-data provider chain, in priority order. Purely descriptive/logging use today;
# consumers (backend/routers/market_data.py, scheduler/market_runner.py) each still do
# their own defensive try/import fallback, but should treat this tuple as the source of
# truth for ordering and labels when logging which provider served a given request.
MARKET_DATA_PROVIDERS = ("upstox", "angel_one")  # "upstox" primary, "angel_one" fallback-only


def market_data_provider_chain() -> tuple[str, ...]:
    """Priority-ordered market-data providers: Upstox (primary) -> Angel One (fallback)."""
    return MARKET_DATA_PROVIDERS

# Module cache of the active config, set by load_provider_config().
_active: dict | None = None


def _synth_models(backend: str) -> tuple[str, str]:
    """(fast_model, deep_model) env defaults for a synthesis backend."""
    if backend == "codex":
        return settings.CODEX_FAST_MODEL, settings.CODEX_DEEP_MODEL
    if backend == "gemini":
        return settings.GEMINI_FAST_MODEL, settings.GEMINI_DEEP_MODEL
    # claude (default)
    return settings.CLAUDE_SONNET_MODEL, settings.CLAUDE_OPUS_MODEL


def _env_defaults() -> dict:
    """Active config derived purely from env — the fallback before any DB load."""
    backend = settings.LLM_SYNTH_BACKEND if settings.LLM_SYNTH_BACKEND in SYNTH_BACKENDS else "claude"
    fast, deep = _synth_models(backend)
    return {
        "macro": {
            "mode": "cloud" if settings.OLLAMA_API_KEY else "local",
            "model": settings.SLM_MODEL,
            "base_url": settings.OLLAMA_URL,
        },
        "synthesis": {
            "backend": backend,
            "fast_model": fast,
            "deep_model": deep,
        },
    }


def get_active() -> dict:
    """Synchronous accessor for the live config. Used by the sync LLM/Ollama clients."""
    return _active if _active is not None else _env_defaults()


def _merge(stored_macro: dict | None, stored_synth: dict | None) -> dict:
    """Overlay stored (DB) choices on top of env defaults."""
    cfg = _env_defaults()
    if stored_macro:
        cfg["macro"].update({k: v for k, v in stored_macro.items() if v is not None})
    if stored_synth:
        backend = stored_synth.get("backend")
        if backend in SYNTH_BACKENDS:
            # rebuild model defaults for the chosen backend, then overlay stored models
            fast, deep = _synth_models(backend)
            cfg["synthesis"] = {"backend": backend, "fast_model": fast, "deep_model": deep}
        cfg["synthesis"].update({k: v for k, v in stored_synth.items() if v})
    return cfg


async def load_provider_config(conn) -> dict:
    """Read app_config into the module cache. Call at pipeline-run start / on config change."""
    global _active
    try:
        rows = await conn.fetch("SELECT key, value FROM app_config WHERE key = ANY($1)",
                                ["macro_provider", "synthesis_provider"])
        stored = {r["key"]: r["value"] for r in rows}
    except Exception as e:
        log.warning("app_config read failed (%s) — using env defaults", e)
        stored = {}
    import json
    macro = stored.get("macro_provider")
    synth = stored.get("synthesis_provider")
    # asyncpg returns jsonb as str; normalise to dict
    if isinstance(macro, str):
        macro = json.loads(macro)
    if isinstance(synth, str):
        synth = json.loads(synth)
    _active = _merge(macro, synth)
    return dict(_active)


async def is_configured(conn) -> bool:
    """True once the user has chosen a synthesis backend through the modal."""
    try:
        row = await conn.fetchrow(
            "SELECT 1 FROM app_config WHERE key = 'synthesis_provider' LIMIT 1"
        )
        return row is not None
    except Exception:
        return False


async def set_provider_config(conn, macro: dict | None = None, synthesis: dict | None = None) -> dict:
    """Persist chosen providers to app_config (no secrets) and refresh the cache."""
    import json
    if macro is not None:
        await conn.execute(
            """INSERT INTO app_config (key, value, updated_at)
               VALUES ('macro_provider', $1::jsonb, NOW())
               ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()""",
            json.dumps(macro),
        )
    if synthesis is not None:
        await conn.execute(
            """INSERT INTO app_config (key, value, updated_at)
               VALUES ('synthesis_provider', $1::jsonb, NOW())
               ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()""",
            json.dumps(synthesis),
        )
    return await load_provider_config(conn)
