"""
Generates SLM commander training pairs via Claude CLI distillation.
Uses historical NSE market situations and asks Claude to reason like a 25yr NSE veteran.

Run: python -m models.slm.train_commander --count 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import pandas as pd

from config import settings

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ClaudeLimitReached(Exception):
    """Raised when the Claude CLI reports a usage/rate limit — stop the batch cleanly."""


# Phrases that signal we've hit the account/usage limit (stop, don't burn through pairs)
_LIMIT_PHRASES = (
    "usage limit",
    "rate limit",
    "limit reached",
    "you've reached your",
    "out of credits",
    "quota exceeded",
    "too many requests",
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
SLM_DATA_DIR = Path(str(settings.DATA_DIR)) / "slm_training"
JSONL_PATH = SLM_DATA_DIR / "training_pairs.jsonl"
CHECKPOINT_PATH = SLM_DATA_DIR / "checkpoint.json"
SLM_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Known Indian market events — major volatility anchors
# ---------------------------------------------------------------------------
MAJOR_MARKET_EVENTS: dict[date, str] = {
    date(2001, 9, 11): "9/11 global markets shock",
    date(2004, 5, 17): "UPA election surprise crash",
    date(2006, 5, 22): "Global commodities crash",
    date(2008, 1, 21): "Global financial crisis begins",
    date(2008, 10, 10): "Lehman aftermath, worst NSE day",
    date(2009, 5, 18): "UPA re-election massive rally",
    date(2013, 8, 28): "INR hits record low, taper tantrum",
    date(2015, 8, 24): "China Black Monday crash",
    date(2016, 11, 9): "Demonetization announcement",
    date(2018, 9, 21): "IL&FS default, NBFC crisis",
    date(2019, 5, 23): "Modi re-election massive rally",
    date(2020, 3, 23): "COVID crash lowest point",
    date(2020, 11, 9): "COVID vaccine announcement rally",
    date(2021, 2, 1): "Budget 2021 surprise rally",
    date(2022, 2, 24): "Russia-Ukraine war shock",
    date(2024, 6, 4): "Election result volatility",
}

# ---------------------------------------------------------------------------
# Sector context helpers
# ---------------------------------------------------------------------------
SECTOR_FII_CONTEXT = {
    "Banking": "Banking sector heavily influenced by RBI policy and FII flows",
    "IT": "IT sector tracks US tech sentiment and USD/INR rates",
    "Pharma": "Pharma sector driven by USFDA approvals and API pricing",
    "Energy": "Energy sector tracks Brent crude and government price policies",
    "Auto": "Auto sector sensitive to fuel prices, credit availability, rural demand",
    "FMCG": "FMCG sector defensive, tracks rural consumption and monsoon outlook",
    "Metals": "Metals sector tracks China demand and global commodity cycles",
    "NBFC": "NBFC sector sensitive to credit spreads and liquidity conditions",
    "Infra": "Infrastructure sector driven by government capex and order books",
    "RealEstate": "Real estate tracks home loan rates and affordability",
}

NSE_VETERAN_SYSTEM_PROMPT = """You are a 25-year NSE veteran trader and analyst. You started trading on NSE when it launched in 1994.
You have lived through: the Ketan Parekh scam (2001), the IT bubble bust, UPA election shocks (2004, 2009),
the 2008 global financial crisis (Nifty fell from 6357 to 2252), the INR crisis of 2013,
demonetization (2016), IL&FS collapse (2018), and COVID crash (2020).

Your expertise:
- You read F&O data better than most — PCR, OI build-up, max pain, futures premium
- You understand FII vs DII behaviour and their seasonal patterns
- You know which sectors lead and which lag in different market regimes
- You apply Darvas Box theory, Stage Analysis, and Indian market seasonality
- You are aware of Budget, RBI policy, election, and global macro impacts on NSE

Your response style:
- Direct and decisive — you never say "it could go either way"
- You cite specific technicals AND macro context
- You always specify stop loss and target with clear logic
- You rate your own confidence honestly

Always respond with valid JSON only. No markdown, no prose outside JSON."""


def _find_event_context(d: date) -> str | None:
    """Return event description if `d` is within 3 days of a major market event."""
    for event_date, description in MAJOR_MARKET_EVENTS.items():
        delta = abs((d - event_date).days)
        if delta <= 3:
            return description
    return None


def build_situation_context(
    ticker: str,
    trade_date: date,
    ohlcv_df: pd.DataFrame,
    features_df: pd.DataFrame | None = None,
) -> str:
    """
    Build a rich historical situation string for a given ticker + date.
    ohlcv_df: OHLCV for this ticker up to (and including) trade_date.
    features_df: optional pre-computed features for the same rows.
    """
    if ohlcv_df.empty:
        return f"Ticker: {ticker} | Date: {trade_date} | No data available"

    # Slice to the trade date. DB timestamps are tz-aware (UTC); drop the tz so
    # they compare cleanly against the tz-naive trade_date Timestamp.
    ohlcv_df = ohlcv_df.copy()
    idx = pd.to_datetime(ohlcv_df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    ohlcv_df.index = idx.normalize()
    trade_dt = pd.Timestamp(trade_date)

    df_to_date = ohlcv_df[ohlcv_df.index <= trade_dt]
    if df_to_date.empty:
        return f"Ticker: {ticker} | Date: {trade_date} | Insufficient history"

    latest = df_to_date.iloc[-1]
    price = float(latest["close"])

    # Price vs SMAs
    sma_20 = float(df_to_date["close"].tail(20).mean()) if len(df_to_date) >= 20 else price
    sma_50 = float(df_to_date["close"].tail(50).mean()) if len(df_to_date) >= 50 else price
    sma_200 = float(df_to_date["close"].tail(200).mean()) if len(df_to_date) >= 200 else price

    # 52-week range
    yr_data = df_to_date.tail(252)
    w52_high = float(yr_data["high"].max()) if not yr_data.empty else price
    w52_low = float(yr_data["low"].min()) if not yr_data.empty else price
    w52_pos_pct = ((price - w52_low) / (w52_high - w52_low + 1e-8)) * 100

    # Volume analysis
    avg_vol_20 = float(df_to_date["volume"].tail(20).mean()) if len(df_to_date) >= 20 else 1.0
    vol_ratio = float(latest["volume"]) / (avg_vol_20 + 1e-8)

    # Simple RSI(14)
    def _rsi(closes: pd.Series, period: int = 14) -> float:
        delta = closes.diff().dropna()
        gains = delta.clip(lower=0).tail(period)
        losses = (-delta.clip(upper=0)).tail(period)
        avg_gain = gains.mean() + 1e-8
        avg_loss = losses.mean() + 1e-8
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    rsi = _rsi(df_to_date["close"]) if len(df_to_date) >= 16 else 50.0

    # RSI label
    if rsi < 30:
        rsi_label = "extreme oversold"
    elif rsi < 40:
        rsi_label = "oversold"
    elif rsi > 70:
        rsi_label = "extreme overbought"
    elif rsi > 60:
        rsi_label = "overbought"
    else:
        rsi_label = "neutral"

    # SMA context
    dist_200 = (price - sma_200) / sma_200 * 100

    # Sector
    from data.pipeline.feature_engineering import SECTOR_MAP
    sector = SECTOR_MAP.get(ticker, "Unknown")
    sector_ctx = SECTOR_FII_CONTEXT.get(sector, f"Sector: {sector}")

    # YTD context (approximate from price action)
    ytd_start = df_to_date[df_to_date.index >= pd.Timestamp(date(trade_date.year, 1, 1))]
    if not ytd_start.empty:
        ytd_pct = (price - float(ytd_start.iloc[0]["close"])) / float(ytd_start.iloc[0]["close"]) * 100
        ytd_str = f"{ytd_pct:+.1f}% YTD"
    else:
        ytd_str = "YTD N/A"

    # Event context
    event_ctx = _find_event_context(trade_date)
    event_suffix = f" ({event_ctx})" if event_ctx else ""

    # MACD simple
    ema12 = df_to_date["close"].ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = df_to_date["close"].ewm(span=26, adjust=False).mean().iloc[-1]
    macd = float(ema12 - ema26)
    macd_label = "positive (bullish)" if macd > 0 else "negative (bearish)"

    lines = [
        f"Ticker: {ticker} | Date: {trade_date}{event_suffix}",
        f"Price: ₹{price:.2f} | 52W Range: ₹{w52_low:.1f}–₹{w52_high:.1f} ({w52_pos_pct:.0f}% of range)",
        f"RSI(14): {rsi:.1f} ({rsi_label}) | Volume: {vol_ratio:.1f}x 20-day average",
        f"MACD: {macd_label} ({macd:+.2f})",
        f"vs SMA20: {((price-sma_20)/sma_20*100):+.1f}% | vs SMA50: {((price-sma_50)/sma_50*100):+.1f}% | vs SMA200: {dist_200:+.1f}%",
        f"Sector: {sector} | {ytd_str}",
        f"Context: {sector_ctx}",
    ]

    if event_ctx:
        lines.append(f"Market Event: {event_ctx}")

    return "\n".join(lines)


def generate_claude_response(situation: str) -> str:
    """
    Calls claude CLI with a 25yr NSE veteran persona.
    Returns JSON string with: decision, stop_loss, target, confidence, reasoning.
    """
    prompt = f"""Here is a real historical NSE market situation:

{situation}

As a 25-year NSE veteran, analyze this situation and provide your trading decision.

Respond ONLY with this JSON structure (no markdown, no extra text):
{{
  "decision": "BUY" | "SELL" | "HOLD",
  "stop_loss_pct": <float — stop loss as % below entry for BUY, above entry for SELL>,
  "target_pct": <float — target as % above entry for BUY, below entry for SELL>,
  "confidence": <float 0.0 to 1.0>,
  "primary_reason": "<one sentence — the single most important factor driving your decision>",
  "technical_read": "<technical analysis in 1-2 sentences>",
  "macro_read": "<macro/sector context in 1-2 sentences>",
  "risk_factors": "<key risks that could invalidate this trade in 1 sentence>",
  "holding_period": "intraday" | "swing (2-5 days)" | "positional (2-4 weeks)"
}}"""

    # Resolve the claude CLI robustly. shutil.which respects PATHEXT on Windows,
    # so it finds the npm shim (claude.cmd) when the npm bin dir is on PATH.
    # A .cmd/.bat shim is not directly executable via CreateProcess — wrap it in
    # `cmd /c` so subprocess (shell=False) can launch it.
    resolved = shutil.which("claude")
    if not resolved:
        raise RuntimeError(
            "claude CLI not found on PATH. Launch this from a shell where "
            "`claude` resolves (e.g. PowerShell with the npm bin dir on PATH)."
        )
    # The system persona + task prompt are large, multi-line, and full of quotes,
    # braces and pipes. Passing them as CLI args gets mangled through the cmd.exe
    # + .cmd-shim layers, so we feed the whole thing via STDIN and keep only
    # simple flags as args. encoding=utf-8 keeps the ₹ symbol intact both ways.
    full_prompt = f"{NSE_VETERAN_SYSTEM_PROMPT}\n\n{prompt}"
    cli_args = ["-p", "--model", "claude-sonnet-4-6", "--thinking", "adaptive"]
    if resolved.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c", resolved, *cli_args]
    else:
        cmd = [resolved, *cli_args]

    try:
        result = subprocess.run(
            cmd,
            input=full_prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        combined = f"{result.stdout}\n{result.stderr}".lower()
        if any(p in combined for p in _LIMIT_PHRASES):
            raise ClaudeLimitReached(result.stdout.strip() or result.stderr.strip())
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            log.warning(f"Claude CLI error: {result.stderr[:300]}")
            return ""
    except subprocess.TimeoutExpired:
        log.warning("Claude CLI timeout for situation generation")
        return ""
    except FileNotFoundError:
        raise RuntimeError("claude CLI not found. Ensure Claude Code is installed.")


async def save_training_pair(
    conn,
    ticker: str,
    trade_date: date,
    situation: str,
    response: str,
) -> None:
    """
    Saves training pair to:
    1. slm_training_pairs table in DB
    2. data/slm_training/training_pairs.jsonl (JSONL format for fine-tuning)
    """
    # Parse response for structured fields
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
        else:
            parsed = {}
    except json.JSONDecodeError:
        parsed = {}

    decision = parsed.get("decision", "HOLD")
    confidence = float(parsed.get("confidence", 0.5))
    reasoning = parsed.get("primary_reason", "")

    # Save to DB
    try:
        await conn.execute(
            """
            INSERT INTO slm_training_pairs
                (ticker, trade_date, situation_context, claude_response,
                 decision, confidence, reasoning, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT DO NOTHING
            """,
            ticker,
            trade_date,
            situation,
            response,
            decision,
            confidence,
            reasoning,
        )
    except Exception as e:
        log.warning(f"DB insert failed for {ticker}/{trade_date}: {e}")

    # Append to JSONL file
    record = {
        "ticker": ticker,
        "date": str(trade_date),
        "situation": situation,
        "response": response,
        "parsed": parsed,
        "created_at": datetime.utcnow().isoformat(),
    }
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


async def select_training_dates(conn, n: int = 500) -> list[tuple[str, date]]:
    """
    Selects diverse (ticker, date) pairs across 2000-2026 for training pair generation.
    Strategy:
    - All major Indian market event dates (from MAJOR_MARKET_EVENTS)
    - High-volatility dates (top daily % moves from DB)
    - Random sample from remaining dates
    Ensures sector coverage.
    """
    # Fetch available tickers and their date ranges
    rows = await conn.fetch(
        """
        SELECT ticker, MIN(time)::date as first_date, MAX(time)::date as last_date,
               COUNT(*) as row_count
        FROM ohlcv_daily
        WHERE close IS NOT NULL
        GROUP BY ticker
        HAVING COUNT(*) >= 300
        """
    )
    if not rows:
        log.warning("No tickers with >= 300 rows found in DB")
        return []

    ticker_info = {
        r["ticker"]: {"first": r["first_date"], "last": r["last_date"]}
        for r in rows
    }
    all_tickers = list(ticker_info.keys())

    selected: list[tuple[str, date]] = []

    # 1. Event dates — one ticker per event (rotate tickers for variety)
    event_dates = sorted(MAJOR_MARKET_EVENTS.keys())
    for i, event_date in enumerate(event_dates):
        ticker = all_tickers[i % len(all_tickers)]
        info = ticker_info[ticker]
        if info["first"] <= event_date <= info["last"]:
            selected.append((ticker, event_date))
        else:
            # Try a different ticker that covers this date
            for t, ti in ticker_info.items():
                if ti["first"] <= event_date <= ti["last"]:
                    selected.append((t, event_date))
                    break

    # 2. High-volatility dates from DB
    try:
        vol_rows = await conn.fetch(
            """
            SELECT ticker, time::date as trade_date,
                   ABS((close - open) / NULLIF(open, 0)) as pct_move
            FROM ohlcv_daily
            WHERE close IS NOT NULL AND open > 0
            ORDER BY pct_move DESC
            LIMIT $1
            """,
            min(n // 2, 300),
        )
        for r in vol_rows:
            pair = (r["ticker"], r["trade_date"])
            if pair not in selected:
                selected.append(pair)
    except Exception as e:
        log.warning(f"Could not fetch high-volatility dates: {e}")

    # 3. Random sample from remaining dates weighted toward post-2010
    remaining_needed = max(0, n - len(selected))
    if remaining_needed > 0:
        try:
            sample_rows = await conn.fetch(
                """
                SELECT ticker, time::date as trade_date
                FROM ohlcv_daily
                WHERE close IS NOT NULL AND time >= '2000-01-01'
                ORDER BY RANDOM()
                LIMIT $1
                """,
                remaining_needed * 3,
            )
            existing_set = set(selected)
            for r in sample_rows:
                pair = (r["ticker"], r["trade_date"])
                if pair not in existing_set and len(selected) < n:
                    selected.append(pair)
                    existing_set.add(pair)
        except Exception as e:
            log.warning(f"Could not fetch random dates: {e}")

    # Shuffle for variety
    import random
    random.shuffle(selected)

    log.info(
        f"Selected {len(selected)} (ticker, date) pairs "
        f"({len(event_dates)} event anchors + high-vol + random)"
    )
    return selected[:n]


def _load_checkpoint() -> set[str]:
    """Load set of already-completed (ticker, date) pairs from checkpoint file."""
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH) as f:
                data = json.load(f)
            return set(data.get("completed", []))
        except Exception:
            pass
    return set()


def _save_checkpoint(completed: set[str]) -> None:
    """Save completed pairs to checkpoint file."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed": list(completed), "updated_at": datetime.utcnow().isoformat()}, f)


async def run(count: int = 500, rate_limit_secs: float = 2.0) -> None:
    """
    Main entry point.
    - Loads DB
    - Selects training (ticker, date) pairs
    - For each pair: builds situation context, calls Claude, saves training pair
    - Rate limited (default 1 call per 2 seconds to avoid CLI overload)
    - Checkpoints every 50 pairs
    """
    log.info(f"Starting SLM commander training pair generation (target: {count})")
    conn = await asyncpg.connect(settings.DATABASE_DSN)

    try:
        pairs = await select_training_dates(conn, n=count)
    except Exception as e:
        log.error(f"Failed to select training dates: {e}")
        await conn.close()
        return

    completed = _load_checkpoint()
    log.info(f"Resuming from checkpoint: {len(completed)} already completed")

    generated = 0
    errors = 0
    consecutive_fail = 0
    MAX_CONSECUTIVE_FAIL = 5  # stop if Claude keeps failing (likely limit/auth issue)

    for i, (ticker, trade_date) in enumerate(pairs):
        pair_key = f"{ticker}_{trade_date}"
        if pair_key in completed:
            continue

        # Fetch OHLCV for this ticker
        try:
            rows = await conn.fetch(
                """
                SELECT time, open, high, low, close, volume
                FROM ohlcv_daily
                WHERE ticker = $1 AND time <= $2 AND close IS NOT NULL
                ORDER BY time ASC
                """,
                ticker,
                datetime.combine(trade_date, datetime.min.time()),
            )
            if not rows:
                log.warning(f"No data for {ticker} up to {trade_date}")
                continue

            ohlcv_df = pd.DataFrame(
                rows, columns=["time", "open", "high", "low", "close", "volume"]
            )
            ohlcv_df["time"] = pd.to_datetime(ohlcv_df["time"])
            ohlcv_df = ohlcv_df.set_index("time")
            for col in ["open", "high", "low", "close", "volume"]:
                ohlcv_df[col] = ohlcv_df[col].astype(float)

        except Exception as e:
            log.warning(f"DB fetch failed for {ticker}/{trade_date}: {e}")
            errors += 1
            continue

        # Build situation context
        situation = build_situation_context(ticker, trade_date, ohlcv_df)

        # Call Claude
        try:
            response = generate_claude_response(situation)
        except ClaudeLimitReached as e:
            _save_checkpoint(completed)
            log.warning(
                "Claude usage limit reached — stopping cleanly at %d pairs. "
                "Re-run the same command later to resume (checkpoint saved). Detail: %s",
                generated, str(e)[:200],
            )
            break

        if not response:
            log.warning(f"Empty Claude response for {ticker}/{trade_date}")
            errors += 1
            consecutive_fail += 1
            if consecutive_fail >= MAX_CONSECUTIVE_FAIL:
                _save_checkpoint(completed)
                log.warning(
                    "%d consecutive failed Claude calls — stopping cleanly at %d pairs "
                    "(likely a limit or auth problem). Re-run to resume.",
                    consecutive_fail, generated,
                )
                break
            continue
        consecutive_fail = 0

        # Save training pair
        try:
            await save_training_pair(conn, ticker, trade_date, situation, response)
        except Exception as e:
            log.warning(f"Save failed for {ticker}/{trade_date}: {e}")
            errors += 1
            continue

        completed.add(pair_key)
        generated += 1

        log.info(
            f"Generated {generated}/{count} training pairs... "
            f"({ticker} {trade_date}, errors={errors})"
        )

        # Checkpoint every 50 pairs
        if generated % 50 == 0:
            _save_checkpoint(completed)
            log.info(f"Checkpoint saved at {generated} pairs")

        # Rate limiting
        time.sleep(rate_limit_secs)

    _save_checkpoint(completed)
    log.info(
        f"Training pair generation complete: {generated} generated, {errors} errors\n"
        f"JSONL file: {JSONL_PATH}"
    )

    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SLM commander training pairs via Claude distillation"
    )
    parser.add_argument(
        "--count", type=int, default=500, help="Number of training pairs to generate"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Seconds to wait between Claude calls (default: 2)",
    )
    args = parser.parse_args()

    import argparse  # already imported at top but needed for __main__ guard

    asyncio.run(run(count=args.count, rate_limit_secs=args.rate_limit))
