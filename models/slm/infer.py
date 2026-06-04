"""
SLM layer — Qwen2.5-7B-Instruct via Ollama.
Falls back gracefully if Ollama not running.
install: curl -fsSL https://ollama.ai/install.sh | sh && ollama pull qwen2.5:7b
fallback: ollama pull qwen2.5:3b  (if 7b OOMs during training)
"""
import os
import json
import time
import logging

import requests

from config import settings

log = logging.getLogger(__name__)

OLLAMA_URL = settings.OLLAMA_URL
SLM_MODEL = settings.SLM_MODEL

OLLAMA_RETRIES = 3
OLLAMA_RETRY_DELAY = 0.5  # seconds

# Default ATR multiplier for volatility-adjusted stops (see SKILL_risk_management).
ATR_STOP_MULT = 1.5
# Stop bounds from risk playbook: min 0.5%, max 4% from entry.
MIN_STOP_PCT = 0.005
MAX_STOP_PCT = 0.04

_SKILL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "intelligence", "skills")
_skills_cache: str | None = None


def _load_skills() -> str:
    """Load and cache the NSE specialist + intraday skill docs from disk."""
    global _skills_cache
    if _skills_cache is not None:
        return _skills_cache

    parts = []
    for fname in ("SKILL_nse_specialist.md", "SKILL_intraday.md"):
        path = os.path.normpath(os.path.join(_SKILL_DIR, fname))
        try:
            with open(path, encoding="utf-8") as f:
                parts.append(f.read())
        except FileNotFoundError:
            log.warning(f"Skill file not found: {path}")
    _skills_cache = "\n\n---\n\n".join(parts)
    return _skills_cache


def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_generate(user_prompt: str, system_prompt: str) -> str:
    """Call Ollama with retries. Returns raw response text ('' on total failure)."""
    last_err = None
    for attempt in range(OLLAMA_RETRIES):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": SLM_MODEL,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=60,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            last_err = f"HTTP {response.status_code}"
        except Exception as e:
            last_err = str(e)
        if attempt < OLLAMA_RETRIES - 1:
            time.sleep(OLLAMA_RETRY_DELAY)
    log.warning(f"Ollama generate failed after {OLLAMA_RETRIES} attempts: {last_err}")
    return ""


def _atr_stops(price: float, signal: str, atr: float | None) -> tuple[float, float]:
    """
    Volatility-adjusted stop/target. Uses ATR when available, clamped to
    the [0.5%, 4%] band from the risk playbook; falls back to 3% flat.
    Returns (stop_loss, target).
    """
    if atr and atr > 0:
        stop_dist = ATR_STOP_MULT * atr
        stop_dist = max(price * MIN_STOP_PCT, min(stop_dist, price * MAX_STOP_PCT))
    else:
        stop_dist = price * 0.03

    # Target at 2R (reward:risk = 2:1) per playbook.
    target_dist = 2.0 * stop_dist

    if signal == "BUY":
        return round(price - stop_dist, 2), round(price + target_dist, 2)
    else:  # SELL / SHORT
        return round(price + stop_dist, 2), round(price - target_dist, 2)


def slm_enrich(
    ticker: str,
    price: float,
    combined_signal: dict,
    ml_reasoning: str,
    kronos_reasoning: str,
    portfolio_held: bool,
    learnings_context: str = "",
    atr: float | None = None,
) -> dict:
    """
    SLM makes final call before Claude: filters, sets stop loss, adds NSE context.
    Only passes through strong signals to Claude.
    """
    signal = combined_signal.get("signal", "HOLD")
    confidence = combined_signal.get("confidence", 0.5)

    if signal == "SELL" and not portfolio_held:
        return {
            "signal": "SKIP",
            "reasoning": f"SLM: {ticker} not in portfolio — SELL signal skipped.",
            "confidence": 0.0,
            "pass_to_claude": False,
        }

    if not _ollama_available():
        log.warning("Ollama not running — SLM layer bypassed")
        return _fallback_slm(ticker, price, combined_signal, portfolio_held, atr)

    system_prompt = _build_system_prompt()
    position_str = "HELD" if portfolio_held else "NOT HELD"
    learnings_block = ("Recent Market Learnings:\n" + learnings_context) if learnings_context else ""
    atr_block = f"14-day ATR: ₹{atr:.2f} ({atr / price * 100:.2f}% of price)" if atr else ""
    suggested_stop, suggested_target = _atr_stops(price, signal if signal != "HOLD" else "BUY", atr)

    user_prompt = f"""
Stock: {ticker}
Current Price: ₹{price:.2f}
Signal from models: {signal} (confidence: {confidence*100:.1f}%)
Portfolio position: {position_str}
{atr_block}
ATR-based stop/target guidance: stop ≈ ₹{suggested_stop:.2f}, target ≈ ₹{suggested_target:.2f}

ML Model Reasoning:
{ml_reasoning}

Kronos Forecast Reasoning:
{kronos_reasoning}

{learnings_block}

Based on NSE market patterns, provide:
1. Your assessment (CONFIRM/REJECT/MODIFY the signal)
2. Stop loss level (in ₹) — volatility-adjusted, never tighter than 0.5% or wider than 4%
3. Target price (in ₹) — aim for at least 2:1 reward:risk
4. Brief reasoning (2-3 sentences, crisp, no fluff)
5. Confidence score (0.0-1.0)
6. Should this go to Claude for final check? (yes/no — only if confidence > 0.65)

Respond in JSON: {{"signal": "BUY|SELL|HOLD", "stop_loss": float, "target": float, "reasoning": "string", "confidence": float, "pass_to_claude": bool}}
"""

    raw = _ollama_generate(user_prompt, system_prompt)
    if not raw:
        return _fallback_slm(ticker, price, combined_signal, portfolio_held, atr)

    # Parse JSON from response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(raw[start:end])
        except json.JSONDecodeError:
            log.warning(f"SLM returned malformed JSON for {ticker}: {raw[:200]}")
            return _fallback_slm(ticker, price, combined_signal, portfolio_held, atr)

        # Backfill missing levels with ATR-based defaults
        result.setdefault("stop_loss", suggested_stop)
        result.setdefault("target", suggested_target)
        result["raw_output"] = raw
        result["reasoning"] = (
            f"SLM — NSE Specialist ({result.get('signal','?')} @ {result.get('confidence',0)*100:.1f}%):\n"
            f"  • {result.get('reasoning','')}\n"
            f"  • Stop loss: ₹{result.get('stop_loss', suggested_stop):.1f}\n"
            f"  • Target: ₹{result.get('target', suggested_target):.1f}"
        )
        return result

    log.warning(f"SLM returned non-JSON for {ticker}: {raw[:200]}")
    return _fallback_slm(ticker, price, combined_signal, portfolio_held, atr)


def _fallback_slm(
    ticker: str, price: float, combined: dict, portfolio_held: bool, atr: float | None = None
) -> dict:
    signal = combined.get("signal", "HOLD")
    conf = combined.get("confidence", 0.5)

    if signal == "SELL" and not portfolio_held:
        return {"signal": "SKIP", "reasoning": "Not in portfolio", "confidence": 0.0, "pass_to_claude": False}

    stop_loss, target = _atr_stops(price, signal if signal != "HOLD" else "BUY", atr)
    basis = "ATR-adjusted" if atr else "flat 3%"

    return {
        "signal": signal,
        "stop_loss": stop_loss,
        "target": target,
        "confidence": conf,
        "pass_to_claude": conf > 0.65,
        "reasoning": (
            f"SLM (fallback — Ollama offline):\n"
            f"  • Signal: {signal} based on combined model output\n"
            f"  • Stop loss: ₹{stop_loss:.1f}  Target: ₹{target:.1f} ({basis})\n"
            f"  • Confidence: {conf*100:.1f}%"
        ),
    }


def _build_system_prompt() -> str:
    skills = _load_skills()
    persona = """You are a 25-year veteran NSE/BSE equity trader and analyst with deep,
hard-won knowledge of Indian markets: Nifty/Sensex dynamics, FII/DII flows, sector rotation
in the Indian context, SEBI regulations, corporate governance patterns in Indian companies,
and seasonal NSE patterns (budget rally, results season, expiry effects).

You always respond with crisp, actionable reasoning. No fluff. Numbers matter.
You never recommend selling what the user doesn't own.
Your stop losses are realistic and volatility-adjusted — never so tight that market noise
triggers them, never so wide that reward:risk breaks down.

Apply the institutional knowledge below to every decision:"""
    return f"{persona}\n\n{skills}" if skills else persona
