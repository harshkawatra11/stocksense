"""
SLM layer — Phi-3 Mini via Ollama.
Falls back gracefully if Ollama not running.
install: curl -fsSL https://ollama.ai/install.sh | sh && ollama pull phi3:mini
"""
import requests
import json
import logging
import os

log = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SLM_MODEL = os.getenv("SLM_MODEL", "phi3:mini")


def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def slm_enrich(
    ticker: str,
    price: float,
    combined_signal: dict,
    ml_reasoning: str,
    kronos_reasoning: str,
    portfolio_held: bool,
    learnings_context: str = "",
) -> dict:
    """
    SLM makes final call before Claude: filters, sets stop loss, adds NSE context.
    Only passes through strong signals to Claude.
    """
    if not _ollama_available():
        log.warning("Ollama not running — SLM layer bypassed")
        return _fallback_slm(ticker, price, combined_signal, portfolio_held)

    signal = combined_signal.get("signal", "HOLD")
    confidence = combined_signal.get("confidence", 0.5)

    if signal == "SELL" and not portfolio_held:
        return {
            "signal": "SKIP",
            "reasoning": f"SLM: {ticker} not in portfolio — SELL signal skipped.",
            "confidence": 0.0,
            "pass_to_claude": False,
        }

    system_prompt = _build_system_prompt()
    position_str = "HELD" if portfolio_held else "NOT HELD"
    learnings_block = ("Recent Market Learnings:\n" + learnings_context) if learnings_context else ""
    user_prompt = f"""
Stock: {ticker}
Current Price: ₹{price:.2f}
Signal from models: {signal} (confidence: {confidence*100:.1f}%)
Portfolio position: {position_str}

ML Model Reasoning:
{ml_reasoning}

Kronos Forecast Reasoning:
{kronos_reasoning}

{learnings_block}

Based on NSE market patterns, provide:
1. Your assessment (CONFIRM/REJECT/MODIFY the signal)
2. Stop loss level (in ₹)
3. Target price (in ₹)
4. Brief reasoning (2-3 sentences, crisp, no fluff)
5. Confidence score (0.0-1.0)
6. Should this go to Claude for final check? (yes/no — only if confidence > 0.65)

Respond in JSON: {{"signal": "BUY|SELL|HOLD", "stop_loss": float, "target": float, "reasoning": "string", "confidence": float, "pass_to_claude": bool}}
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": SLM_MODEL, "prompt": user_prompt, "system": system_prompt, "stream": False},
            timeout=30,
        )
        data = response.json()
        raw = data.get("response", "")

        # Parse JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
            result["raw_output"] = raw
            result["reasoning"] = (
                f"SLM — NSE Specialist ({result.get('signal','?')} @ {result.get('confidence',0)*100:.1f}%):\n"
                f"  • {result.get('reasoning','')}\n"
                f"  • Stop loss: ₹{result.get('stop_loss', price*0.97):.1f}\n"
                f"  • Target: ₹{result.get('target', price*1.03):.1f}"
            )
            return result
        else:
            log.warning(f"SLM returned non-JSON for {ticker}: {raw[:200]}")
            return _fallback_slm(ticker, price, combined_signal, portfolio_held)

    except Exception as e:
        log.error(f"SLM inference error for {ticker}: {e}")
        return _fallback_slm(ticker, price, combined_signal, portfolio_held)


def _fallback_slm(ticker: str, price: float, combined: dict, portfolio_held: bool) -> dict:
    signal = combined.get("signal", "HOLD")
    conf = combined.get("confidence", 0.5)

    if signal == "SELL" and not portfolio_held:
        return {"signal": "SKIP", "reasoning": "Not in portfolio", "confidence": 0.0, "pass_to_claude": False}

    stop_loss = price * 0.97 if signal == "BUY" else price * 1.03
    target = price * 1.03 if signal == "BUY" else price * 0.97

    return {
        "signal": signal,
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "confidence": conf,
        "pass_to_claude": conf > 0.65,
        "reasoning": (
            f"SLM (fallback — Ollama offline):\n"
            f"  • Signal: {signal} based on combined model output\n"
            f"  • Stop loss: ₹{stop_loss:.1f}  Target: ₹{target:.1f}\n"
            f"  • Confidence: {conf*100:.1f}%"
        ),
    }


def _build_system_prompt() -> str:
    return """You are an expert NSE/BSE stock market analyst with deep knowledge of Indian markets.
You understand Nifty/Sensex dynamics, FII/DII flows, sector rotation in Indian context,
SEBI regulations, corporate governance patterns in Indian companies, and seasonal patterns
on NSE (budget rally, results season, expiry effects).

You always respond with crisp, actionable reasoning. No fluff. Numbers matter.
You never recommend selling what the user doesn't own.
Your stop losses are realistic — not too tight to get stopped out on noise."""
