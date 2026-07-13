"""
Macro / news context layer.

This is the SLM's *new* job (see slm-repositioned-as-macro-news-layer):
instead of re-forecasting price, the SLM reads live financial news and
produces a per-sector sentiment read for the Indian market.

Pipeline role:
    Runs ONCE per pipeline cycle (macro context is market-wide, not per-ticker).
    Output is a dict of sector -> {score, drivers} that the signal stage uses
    to nudge per-ticker confidence up/down based on what's happening in the world.

Sources are plain RSS (reachable where Angel One / yfinance are network-blocked).
Scoring uses the same local Ollama qwen2.5 model as models/slm/infer.py.

Standalone test:
    python -m intelligence.macro_context
"""
from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from config import settings
from models.slm.ollama_client import (
    ollama_available as _ollama_available,
    ollama_generate,
    is_cloud as _ollama_is_cloud,
)

log = logging.getLogger(__name__)

# qwen2.5:7b on a 4GB GPU partially offloads to CPU, so a large structured
# generation is slow. Give it room and bound the output length.
MACRO_TIMEOUT = 300        # seconds per generate attempt
MACRO_NUM_PREDICT = 700    # cap output tokens — the JSON is bounded in size


def _macro_generate(user_prompt: str, system_prompt: str) -> str:
    """
    Dedicated macro-layer call via the shared Ollama client (local or cloud):
    longer timeout + bounded JSON output, with a warm-up so a cold local load
    doesn't eat the scoring call's clock.
    """
    return ollama_generate(
        user_prompt, system_prompt,
        options={"temperature": 0.2, "num_predict": MACRO_NUM_PREDICT},
        format_json=True, warm_up=True, timeout=MACRO_TIMEOUT,
    )

# ------------------------------------------------------------------ #
# News sources — RSS feeds confirmed reachable on the blocked network #
# ------------------------------------------------------------------ #
NEWS_FEEDS: list[tuple[str, str]] = [
    ("Moneycontrol", "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("Moneycontrol-Markets", "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("Moneycontrol-Business", "https://www.moneycontrol.com/rss/business.xml"),
    ("ET-Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("ET-Economy", "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"),
]

FEED_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
FEED_TIMEOUT = 12

# Broad NSE sector taxonomy — aligned to the Nifty sectoral indices.
# These are the buckets qwen2.5 scores and that we map tickers onto.
SECTORS: list[str] = [
    "IT",
    "BANKING",          # private + PSU banks
    "NBFC_FINANCE",     # NBFCs, finance, insurance, capital markets
    "AUTO",
    "PHARMA",           # pharma + healthcare
    "FMCG",
    "METALS",           # metals + mining
    "ENERGY",           # oil, gas, refiners
    "POWER",            # power generation, utilities
    "REALTY",
    "CEMENT_INFRA",     # cement + construction/infra
    "TELECOM",
    "MEDIA",
    "CHEMICALS",
]

# How long a macro read stays valid before we refetch + rescore.
CACHE_TTL_SECONDS = 30 * 60


@dataclass
class MacroContext:
    """One market-wide macro read."""
    generated_at: str
    market_bias: str                      # BULLISH | BEARISH | NEUTRAL (overall)
    market_summary: str                   # one-line headline read
    sectors: dict[str, dict]              # sector -> {score: float, drivers: str}
    headline_count: int
    source: str                           # "slm" | "fallback"  (legacy field, kept for compat)
    headlines: list[dict] = field(default_factory=list)
    # Stage 0 truth-layer status. status: "ok"|"degraded"|"unavailable".
    # component_source: "ollama_local" | "ollama_cloud" | "neutral_fallback" — tells
    # downstream code/health-endpoint whether sector scores are genuinely-neutral
    # (a real Ollama read that came back flat) or a fallback because Ollama was
    # unreachable/empty/unparseable (indistinguishable from "neutral" without this).
    component_status: str = "ok"
    component_source: str = "ollama_local"
    component_detail: str = ""

    def sector_score(self, sector: str | None) -> float:
        """Sentiment in [-1, 1] for a sector; 0.0 if unknown/missing."""
        if not sector:
            return 0.0
        entry = self.sectors.get(sector.upper())
        return float(entry["score"]) if entry else 0.0


# ------------------------------------------------------------------ #
# Step 1 — fetch headlines                                            #
# ------------------------------------------------------------------ #
def fetch_headlines(max_per_feed: int = 8, max_total: int = 25) -> list[dict]:
    """
    Pull recent headlines from the RSS feeds.
    Returns list of {title, summary, source, published} dicts, de-duplicated by title.
    """
    seen: set[str] = set()
    out: list[dict] = []

    for source, url in NEWS_FEEDS:
        try:
            resp = requests.get(url, headers=FEED_HEADERS, timeout=FEED_TIMEOUT)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            log.warning("News feed failed (%s): %s", source, e)
            continue

        items = root.findall(".//item")[:max_per_feed]
        for it in items:
            title = (it.findtext("title") or "").strip()
            if not title or title.lower() in seen:
                continue
            seen.add(title.lower())
            summary = (it.findtext("description") or "").strip()
            # Strip CDATA/HTML noise crudely — keep it short for the prompt.
            summary = summary.replace("<![CDATA[", "").replace("]]>", "")
            out.append({
                "title": title,
                "summary": summary[:240],
                "source": source,
                "published": (it.findtext("pubDate") or "").strip(),
            })

    log.info("Fetched %d unique headlines from %d feeds", len(out), len(NEWS_FEEDS))
    return out[:max_total]


# ------------------------------------------------------------------ #
# Step 2 — score sectors via qwen2.5                                  #
# ------------------------------------------------------------------ #
def _build_scoring_prompt(headlines: list[dict]) -> tuple[str, str]:
    system = (
        "You are a macro analyst for the Indian equity market (NSE/BSE). "
        "You read financial news and judge the near-term (next 1-5 trading days) "
        "directional impact on each market sector. You are decisive and quantitative. "
        "Geopolitics, commodity moves (crude/gold), RBI/rates, FII/DII flows, budget/policy, "
        "and global cues all feed your read. No hedging, no fluff."
    )

    headline_block = "\n".join(
        f"{i+1}. [{h['source']}] {h['title']}" + (f" — {h['summary']}" if h['summary'] else "")
        for i, h in enumerate(headlines)
    )

    sector_list = ", ".join(SECTORS)

    user = f"""Today's Indian market headlines:

{headline_block}

Score the near-term sentiment for EACH of these sectors: {sector_list}

For each sector give:
  - score: a float from -1.0 (strongly bearish) to +1.0 (strongly bullish), 0.0 if no clear signal
  - drivers: one short phrase naming the specific news driving the score (or "no specific news")

Also give an overall market_bias (BULLISH/BEARISH/NEUTRAL) and a one-line market_summary.

Respond ONLY with JSON in exactly this shape:
{{
  "market_bias": "BULLISH|BEARISH|NEUTRAL",
  "market_summary": "string",
  "sectors": {{
    "IT": {{"score": 0.0, "drivers": "string"}},
    "BANKING": {{"score": 0.0, "drivers": "string"}}
    // ... every sector listed above
  }}
}}"""
    return user, system


def score_sectors(headlines: list[dict]) -> MacroContext:
    """Run qwen2.5 over the headlines and return a MacroContext."""
    now = datetime.now(timezone.utc).isoformat()

    if not headlines:
        return _neutral_context(now, 0, "No headlines fetched")

    if not _ollama_available():
        log.warning("Ollama unavailable — macro layer returns neutral context")
        return _neutral_context(now, len(headlines), "Ollama offline")

    user_prompt, system_prompt = _build_scoring_prompt(headlines)
    raw = _macro_generate(user_prompt, system_prompt)
    if not raw:
        return _neutral_context(now, len(headlines), "SLM returned empty")

    start, end = raw.find("{"), raw.rfind("}") + 1
    if start < 0 or end <= start:
        log.warning("Macro: non-JSON SLM output: %s", raw[:200])
        return _neutral_context(now, len(headlines), "SLM non-JSON output")

    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        log.warning("Macro: malformed SLM JSON: %s", raw[:200])
        return _neutral_context(now, len(headlines), "SLM malformed JSON")

    # Normalise sectors: clamp scores, ensure every sector present.
    sectors: dict[str, dict] = {}
    raw_sectors = parsed.get("sectors", {}) or {}
    for sec in SECTORS:
        entry = raw_sectors.get(sec) or {}
        try:
            score = float(entry.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(-1.0, min(1.0, score))
        sectors[sec] = {
            "score": round(score, 2),
            "drivers": str(entry.get("drivers", "")).strip()[:160] or "no specific news",
        }

    return MacroContext(
        generated_at=now,
        market_bias=str(parsed.get("market_bias", "NEUTRAL")).upper(),
        market_summary=str(parsed.get("market_summary", "")).strip()[:300],
        sectors=sectors,
        headline_count=len(headlines),
        source="slm",
        headlines=headlines,
        component_status="ok",
        component_source="ollama_cloud" if _ollama_is_cloud() else "ollama_local",
        component_detail=f"scored {len(headlines)} headlines via qwen2.5",
    )


def _neutral_context(ts: str, n: int, reason: str) -> MacroContext:
    """
    All-zero sector context used whenever the macro layer couldn't produce a real
    read (Ollama unreachable, empty/unparseable output, no headlines). Stage 0:
    component_source="neutral_fallback" so downstream code and the health endpoint
    can tell this apart from a genuine "the market has no sector bias today" read
    (which would carry component_source="ollama_local"/"ollama_cloud" instead).
    """
    return MacroContext(
        generated_at=ts,
        market_bias="NEUTRAL",
        market_summary=f"Macro layer inactive ({reason})",
        sectors={s: {"score": 0.0, "drivers": "n/a"} for s in SECTORS},
        headline_count=n,
        source="fallback",
        component_status="unavailable",
        component_source="neutral_fallback",
        component_detail=reason,
    )


# ------------------------------------------------------------------ #
# Cached entry point                                                  #
# ------------------------------------------------------------------ #
_cache: dict = {"ctx": None, "ts": 0.0}


def get_macro_context(force_refresh: bool = False) -> MacroContext:
    """
    Main entry point for the pipeline. Fetches + scores news, cached for
    CACHE_TTL_SECONDS so we don't rescore on every ticker in a run.
    """
    now = time.monotonic()
    if not force_refresh and _cache["ctx"] is not None and (now - _cache["ts"]) < CACHE_TTL_SECONDS:
        return _cache["ctx"]

    headlines = fetch_headlines()
    ctx = score_sectors(headlines)
    _cache["ctx"] = ctx
    _cache["ts"] = now
    return ctx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    ctx = get_macro_context(force_refresh=True)
    dt = time.time() - t0

    print("\n" + "=" * 60)
    print(f"MACRO CONTEXT  ({ctx.source})   {dt:.1f}s   {ctx.headline_count} headlines")
    print("=" * 60)
    print(f"Market bias : {ctx.market_bias}")
    print(f"Summary     : {ctx.market_summary}")
    print("-" * 60)
    for sec, v in sorted(ctx.sectors.items(), key=lambda kv: kv[1]["score"], reverse=True):
        bar_n = int(abs(v["score"]) * 10)
        bar = ("+" if v["score"] >= 0 else "-") * bar_n
        print(f"  {sec:<14} {v['score']:+.2f}  {bar:<10}  {v['drivers']}")
    print("=" * 60)
