"""
Syncs real Angel One holdings into the portfolio table's watch_only rows.

Context: portfolio rows with watch_only=TRUE represent real external Angel
One holdings the brain only monitors/alerts on — it never auto-buys or
auto-sells them (see intelligence/auto_trader.auto_exit's docstring and
_open_position_count's exclusion, both already correct). Until now those 5
rows were a one-time manual sync from 2026-06-25 and never refreshed, so the
portfolio view silently went stale the moment anything changed on the real
Angel One account. This module replaces that one-time snapshot with a
periodic real sync via Angel One's SmartAPI holdings endpoint.

Never raises: on any failure (creds missing, session/login failure, API
error) returns a {status, detail} dict and leaves the existing portfolio
rows untouched — a sync failure must never wipe out or corrupt data that
was already there, same contract as the rest of this codebase's external
API wrappers (e.g. data/pipeline/upstox_orders.py).

Run standalone: python -m intelligence.angel_holdings_sync
"""
from __future__ import annotations

import logging

import asyncpg

from config import settings

log = logging.getLogger(__name__)


def creds_present() -> bool:
    return bool(
        settings.ANGEL_ONE_API_KEY and settings.ANGEL_ONE_CLIENT_ID
        and settings.ANGEL_ONE_PIN and settings.ANGEL_ONE_TOTP_KEY
    )


async def sync_angel_holdings(conn=None) -> dict:
    """
    Pull real holdings from Angel One and upsert them into portfolio as
    watch_only=TRUE rows. Returns {status: "ok"|"unavailable", detail: str,
    synced: [...]}. A ticker present in the DB's watch_only set but no
    longer in the real Angel One holdings response is deactivated
    (active=FALSE) — it was sold/moved and should stop being monitored.
    """
    if not creds_present():
        return {"status": "unavailable",
                "detail": "ANGEL_ONE_* credentials not set in .env — sync skipped",
                "synced": []}

    try:
        from data.pipeline.fetch_live import get_session
        obj = get_session()
        resp = obj.holding()
    except Exception as e:
        log.warning("Angel One holdings fetch failed: %s", e)
        return {"status": "unavailable", "detail": f"Angel One API error: {e}", "synced": []}

    if not resp or not resp.get("status"):
        detail = (resp or {}).get("message", "unknown error")
        log.warning("Angel One holdings response not ok: %s", detail)
        return {"status": "unavailable", "detail": f"Angel One responded not-ok: {detail}", "synced": []}

    holdings = resp.get("data") or []

    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        synced = []
        seen_tickers: set[str] = set()
        for h in holdings:
            ticker = h.get("tradingsymbol", "").replace("-EQ", "")
            qty = int(h.get("quantity") or 0)
            avg_price = float(h.get("averageprice") or 0)
            if not ticker or qty < 1 or avg_price <= 0:
                continue
            seen_tickers.add(ticker)

            await conn.execute(
                "INSERT INTO stocks (ticker, name) VALUES ($1, $1) ON CONFLICT DO NOTHING",
                ticker,
            )
            existing = await conn.fetchrow(
                "SELECT id FROM portfolio WHERE ticker = $1 AND watch_only = TRUE AND active = TRUE",
                ticker,
            )
            if existing:
                await conn.execute(
                    """
                    UPDATE portfolio SET quantity = $2, avg_price = $3,
                           notes = 'REAL Angel holding, live-synced'
                    WHERE id = $1
                    """,
                    existing["id"], qty, avg_price,
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO portfolio (ticker, quantity, avg_price, buy_date, watch_only, notes)
                    VALUES ($1, $2, $3, NOW(), TRUE, 'REAL Angel holding, live-synced')
                    """,
                    ticker, qty, avg_price,
                )
            synced.append({"ticker": ticker, "quantity": qty, "avg_price": avg_price})

        # Anything previously watch_only-tracked but absent from this real
        # holdings response was sold/moved on the real account — stop
        # monitoring it rather than showing a phantom position forever.
        stale = await conn.fetch(
            "SELECT ticker FROM portfolio WHERE watch_only = TRUE AND active = TRUE"
        )
        for row in stale:
            if row["ticker"] not in seen_tickers:
                await conn.execute(
                    "UPDATE portfolio SET active = FALSE WHERE ticker = $1 AND watch_only = TRUE",
                    row["ticker"],
                )
                log.info("Angel holding %s no longer present — deactivated", row["ticker"])

        log.info("Angel One holdings sync: %d position(s) synced", len(synced))
        return {"status": "ok", "detail": f"{len(synced)} holding(s) synced", "synced": synced}
    finally:
        if own_conn:
            await conn.close()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(sync_angel_holdings()))
