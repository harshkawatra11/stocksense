"""
Providers API — choose and verify the LLM engines that power the two AI layers.

Backs the first-run modal: report what's configured and which CLIs are installed,
serve the catalog of options with install/login commands, verify a choice with a
real tiny call, and persist the selection (no secrets stored).
"""
from __future__ import annotations

import asyncpg
from fastapi import APIRouter
from pydantic import BaseModel

from config import settings
from intelligence import llm_cli
from intelligence.provider_config import (
    get_active, load_provider_config, set_provider_config, is_configured, SYNTH_BACKENDS,
)
from models.slm.ollama_client import ollama_available

router = APIRouter()
DB_DSN = settings.DATABASE_DSN


# Static catalog with the commands a new user runs to get each engine working.
OPTIONS = {
    "synthesis": [
        {"backend": "claude", "label": "Claude Code (Anthropic)",
         "install": "npm install -g @anthropic-ai/claude-code",
         "login": "run `claude` once, then `/login`"},
        {"backend": "codex", "label": "Codex (OpenAI)",
         "install": "npm install -g @openai/codex",
         "login": "run `codex login`"},
        {"backend": "gemini", "label": "Gemini (Google)",
         "install": "npm install -g @google/gemini-cli",
         "login": "run `gemini`, then `/auth`"},
    ],
    "macro": [
        {"mode": "local", "label": "Local Ollama",
         "install": "ollama pull qwen2.5:7b",
         "note": "Free, runs on your machine (a GPU helps). No key needed."},
        {"mode": "cloud", "label": "Ollama Cloud (e.g. Nemotron 3 Ultra)",
         "install": "set OLLAMA_URL + OLLAMA_API_KEY in .env, SLM_MODEL to a hosted model",
         "note": "No local GPU needed. Billed per call by Ollama Cloud."},
    ],
}


class ProviderConfigBody(BaseModel):
    macro_mode: str | None = None       # "local" | "cloud"
    macro_model: str | None = None
    synthesis_backend: str | None = None  # "claude" | "codex" | "gemini"
    fast_model: str | None = None
    deep_model: str | None = None


class VerifyBody(BaseModel):
    synthesis_backend: str | None = None
    macro_mode: str | None = None


@router.get("/status")
async def status():
    """Current resolved config + detection. is_configured gates the first-run modal."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        await load_provider_config(conn)
        configured = await is_configured(conn)
    finally:
        await conn.close()
    return {
        "is_configured": configured,
        "active": get_active(),
        "synthesis_clis": llm_cli.detect(),
        "macro": {
            "endpoint_reachable": ollama_available(),
            "has_cloud_key": bool(settings.OLLAMA_API_KEY),
        },
    }


@router.get("/options")
async def options():
    return OPTIONS


@router.post("/verify")
async def verify(body: VerifyBody):
    """Run a real tiny check on the selected synthesis CLI and macro endpoint."""
    result: dict = {}
    if body.synthesis_backend:
        ok, detail = llm_cli.verify_backend(body.synthesis_backend)
        result["synthesis"] = {"ok": ok, "detail": detail}
    if body.macro_mode == "cloud":
        ok = bool(settings.OLLAMA_API_KEY) and ollama_available()
        result["macro"] = {"ok": ok,
                           "detail": "cloud reachable" if ok else "set OLLAMA_URL + OLLAMA_API_KEY in .env"}
    elif body.macro_mode == "local":
        ok = ollama_available()
        result["macro"] = {"ok": ok, "detail": "local Ollama reachable" if ok else "start Ollama / pull a model"}
    return result


@router.post("/config")
async def save_config(body: ProviderConfigBody):
    """Persist the chosen engines (no secrets) and refresh the live config."""
    backend = (body.synthesis_backend or "").lower()
    if backend and backend not in SYNTH_BACKENDS:
        return {"ok": False, "error": f"unknown synthesis backend '{backend}'"}

    macro = None
    if body.macro_mode in ("local", "cloud"):
        macro = {
            "mode": body.macro_mode,
            "model": body.macro_model or settings.SLM_MODEL,
            "has_key": bool(settings.OLLAMA_API_KEY),
        }
    synthesis = None
    if backend:
        synthesis = {"backend": backend}
        if body.fast_model:
            synthesis["fast_model"] = body.fast_model
        if body.deep_model:
            synthesis["deep_model"] = body.deep_model

    conn = await asyncpg.connect(DB_DSN)
    try:
        active = await set_provider_config(conn, macro=macro, synthesis=synthesis)
    finally:
        await conn.close()
    return {"ok": True, "active": active}
