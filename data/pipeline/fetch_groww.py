"""
Groww trade API — primary live + daily data feed.

Why Groww over Angel One here: Groww's TOTP auth needs no IP whitelisting and
no daily key approval, so a home connection with a rotating IP works fine. The
access token is generated programmatically each session via pyotp.

Provides:
  - get_session()                 : authenticate and return a cached GrowwAPI client
  - get_ltp(tickers)              : last traded price for NSE equity tickers
  - get_quote(ticker)             : full quote for one ticker
  - run_groww_daily_backfill(...) : daily candles -> ohlcv_daily upsert
  - run_groww_intraday_snapshot() : live quotes -> ohlcv_1min upsert (flips
                                    data_freshness to live during market hours)

Everything degrades gracefully: if GROWW_* creds are absent or auth fails, all
entry points log a warning and return empty results — the system continues on
NSE Bhavcopy EOD data (offline-first design preserved).

Setup (one-time, user): subscribe to Groww Trade API, generate the TOTP secret
on https://groww.in/trade-api, then set in .env:
    GROWW_API_KEY=...
    GROWW_TOTP_SECRET=...        (preferred)
    # or GROWW_API_SECRET=...    (requires daily approval on Groww's site)
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone

import asyncpg

from config import settings

log = logging.getLogger(__name__)

_client = None
_client_at: float = 0.0
_SESSION_TTL = 6 * 3600  # re-auth every 6h to stay ahead of token expiry

# Conservative pacing under Groww's live-data limit (300/min).
_RATE_DELAY = 0.25


def creds_present() -> bool:
    return bool(settings.GROWW_API_KEY and
                (settings.GROWW_TOTP_SECRET or settings.GROWW_API_SECRET))


def get_session(force: bool = False):
    """Authenticated GrowwAPI client, cached. Returns None if creds absent/auth fails."""
    global _client, _client_at
    if not creds_present():
        log.warning("Groww credentials not configured — feed disabled (set GROWW_API_KEY + GROWW_TOTP_SECRET in .env)")
        return None
    if _client is not None and not force and (time.time() - _client_at) < _SESSION_TTL:
        return _client

    try:
        from growwapi import GrowwAPI
        if settings.GROWW_TOTP_SECRET:
            import pyotp
            totp = pyotp.TOTP(settings.GROWW_TOTP_SECRET).now()
            token_resp = GrowwAPI.get_access_token(api_key=settings.GROWW_API_KEY, totp=totp)
        else:
            token_resp = GrowwAPI.get_access_token(
                api_key=settings.GROWW_API_KEY, secret=settings.GROWW_API_SECRET
            )
        token = token_resp.get("token") if isinstance(token_resp, dict) else token_resp
        _client = GrowwAPI(token)
        _client_at = time.time()
        log.info("Groww session established.")
        return _client
    except Exception as e:
        log.warning("Groww auth failed: %s — feed disabled this cycle", e)
        _client = None
        return None


def _retry_auth(fn, *args, **kwargs):
    """Run an SDK call; on auth-ish failure refresh the session once and retry."""
    client = get_session()
    if client is None:
        return None
    try:
        return fn(client, *args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "unauthor" in msg or "token" in msg:
            log.info("Groww call failed (%s) — re-authenticating once", e)
            client = get_session(force=True)
            if client is None:
                return None
            return fn(client, *args, **kwargs)
        raise


def get_ltp(tickers: list[str]) -> dict[str, float]:
    """{ticker: ltp} for NSE equity tickers. Empty dict on failure."""
    if not tickers:
        return {}
    from growwapi import GrowwAPI

    def call(client):
        out: dict[str, float] = {}
        # SDK takes 'NSE_SYMBOL' keys, batched.
        BATCH = 50
        for i in range(0, len(tickers), BATCH):
            batch = tickers[i:i + BATCH]
            symbols = tuple(f"NSE_{t}" for t in batch)
            resp = client.get_ltp(exchange_trading_symbols=symbols,
                                  segment=GrowwAPI.SEGMENT_CASH)
            for key, val in (resp or {}).items():
                t = key.replace("NSE_", "", 1)
                try:
                    out[t] = float(val if not isinstance(val, dict) else val.get("ltp", 0))
                except (TypeError, ValueError):
                    continue
            time.sleep(_RATE_DELAY)
        return out

    try:
        return _retry_auth(call) or {}
    except Exception as e:
        log.warning("Groww get_ltp failed: %s", e)
        return {}


def get_quote(ticker: str) -> dict | None:
    """Full quote (OHLC, volume, depth) for one NSE equity ticker."""
    from growwapi import GrowwAPI

    def call(client):
        return client.get_quote(trading_symbol=ticker,
                                exchange=GrowwAPI.EXCHANGE_NSE,
                                segment=GrowwAPI.SEGMENT_CASH)

    try:
        return _retry_auth(call)
    except Exception as e:
        log.warning("Groww get_quote failed for %s: %s", ticker, e)
        return None


def _parse_candles(resp: dict) -> list[list]:
    """Normalize the historical-candle response to [[epoch_or_iso, o, h, l, c, v], ...]."""
    if not isinstance(resp, dict):
        return []
    return resp.get("candles") or resp.get("data", {}).get("candles") or []


def _candle_time(raw) -> datetime | None:
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, OSError, TypeError):
        return None


async def run_groww_daily_backfill(days_back: int = 5, tickers: list[str] | None = None) -> dict:
    """
    Pull recent daily candles from Groww and upsert into ohlcv_daily.
    Returns {"updated": n_tickers, "rows": n_rows}. Zero-result on missing creds.
    """
    client = get_session()
    if client is None:
        return {"updated": 0, "rows": 0, "skipped": "no groww session"}

    from growwapi import GrowwAPI
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        if tickers is None:
            rows = await conn.fetch(
                "SELECT ticker FROM stocks WHERE active = TRUE ORDER BY ticker LIMIT $1",
                settings.MAX_TICKERS_PER_RUN,
            )
            tickers = [r["ticker"] for r in rows]

        start = (date.today() - timedelta(days=days_back)).isoformat() + " 09:00:00"
        end = date.today().isoformat() + " 16:00:00"

        updated, total_rows = 0, 0
        for t in tickers:
            try:
                resp = client.get_historical_candle_data(
                    trading_symbol=t,
                    exchange=GrowwAPI.EXCHANGE_NSE,
                    segment=GrowwAPI.SEGMENT_CASH,
                    start_time=start, end_time=end,
                    interval_in_minutes=1440,
                )
            except Exception as e:
                log.debug("Groww daily candles failed %s: %s", t, e)
                time.sleep(_RATE_DELAY)
                continue

            candles = _parse_candles(resp)
            db_rows = []
            for c in candles:
                if not isinstance(c, (list, tuple)) or len(c) < 5:
                    continue
                ts = _candle_time(c[0])
                if ts is None:
                    continue
                db_rows.append((
                    ts, t,
                    float(c[1]), float(c[2]), float(c[3]), float(c[4]),
                    int(c[5]) if len(c) > 5 and c[5] is not None else 0,
                    float(c[4]),
                ))
            if db_rows:
                await conn.executemany(
                    """
                    INSERT INTO ohlcv_daily (time, ticker, open, high, low, close, volume, adj_close)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    ON CONFLICT (time, ticker) DO UPDATE SET
                        open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                        close=EXCLUDED.close, volume=EXCLUDED.volume, adj_close=EXCLUDED.adj_close
                    """,
                    db_rows,
                )
                updated += 1
                total_rows += len(db_rows)
            time.sleep(_RATE_DELAY)

        log.info("Groww daily backfill: %d tickers, %d rows", updated, total_rows)
        return {"updated": updated, "rows": total_rows}
    finally:
        await conn.close()


async def run_groww_intraday_snapshot(max_symbols: int = 50) -> dict:
    """
    Live quote snapshot for portfolio tickers + today's top-signal tickers,
    upserted into ohlcv_1min. This is what flips data_freshness.is_live during
    market hours and makes position reviews genuinely live.
    """
    client = get_session()
    if client is None:
        return {"rows": 0, "skipped": "no groww session"}

    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        held = [r["ticker"] for r in await conn.fetch(
            "SELECT ticker FROM portfolio WHERE active = TRUE"
        )]
        top = [r["ticker"] for r in await conn.fetch(
            """
            SELECT DISTINCT ticker FROM signals
            WHERE fired_at::date = CURRENT_DATE AND signal_type = 'BUY'
            ORDER BY ticker LIMIT $1
            """,
            max_symbols,
        )]
        tickers = list(dict.fromkeys(held + top))[:max_symbols]
        if not tickers:
            return {"rows": 0, "skipped": "no symbols to watch"}

        prices = get_ltp(tickers)
        if not prices:
            return {"rows": 0, "skipped": "no prices returned"}

        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        rows = [(now, t, p, p, p, p, 0) for t, p in prices.items() if p and p > 0]
        if rows:
            await conn.executemany(
                """
                INSERT INTO ohlcv_1min (time, ticker, open, high, low, close, volume)
                VALUES ($1,$2,$3,$4,$5,$6,$7)
                ON CONFLICT (time, ticker) DO UPDATE SET close=EXCLUDED.close
                """,
                rows,
            )
        log.info("Groww intraday snapshot: %d live prices stored", len(rows))
        return {"rows": len(rows), "tickers": [r[1] for r in rows]}
    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(run_groww_daily_backfill(days_back=3)))
