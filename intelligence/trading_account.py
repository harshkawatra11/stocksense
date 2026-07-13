"""
Trading account + decision ledger.

Tracks the user's capital (deployable + reserve) and every decision taken on a
signal — including passes — so the learning loop can later judge not just what
was acted on but what was skipped. Built for the initial low-capital phase
(₹500 deployable) where most signals are informational until capital grows.

Tables: account, decisions (see data/db/schema_v2_live.sql).

Epoch marker: the account was refunded from an insolvent ₹500 seed to a real
₹100,000 (see config.py CASH_AVAILABLE) — "epoch 2". Pre-reset decisions/signals
are kept (never deleted) but must not count toward the live PAPER->LIVE gate,
since they were computed against a budget that could almost never buy 1 share.
get_current_epoch_start() persists the reset timestamp in the existing generic
app_config table (data/db/schema_v5_providers.sql) under key 'account_epoch' —
reused rather than adding a new migration, since it's a single row, not a new
table shape. Seeded lazily on first read.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from intelligence.portfolio_guard import add_to_portfolio, remove_from_portfolio

log = logging.getLogger(__name__)

CURRENT_EPOCH = 2  # bump this (and reseed) if the account is ever refunded again


async def get_current_epoch_start(conn) -> datetime:
    """
    Returns the UTC timestamp marking the start of the current funding epoch
    (epoch 2 = the ₹100,000 refund). Signals/decisions before this timestamp
    belong to the insolvent ₹500-era ledger and must be excluded from any
    live accuracy/expectancy computation.

    Additive/idempotent: if no epoch row exists yet, seeds one dated NOW() the
    first time this is called (one-time, first-run-after-deploy behavior).
    Never deletes or modifies historical ledger rows.
    """
    row = await conn.fetchrow(
        "SELECT value FROM app_config WHERE key = 'account_epoch'"
    )
    if row:
        value = row["value"]
        if isinstance(value, str):
            value = json.loads(value)
        if value.get("epoch") == CURRENT_EPOCH and value.get("epoch_start"):
            return datetime.fromisoformat(value["epoch_start"])

    # No epoch row yet (or a stale/older epoch) — seed epoch 2 dated now.
    epoch_start = datetime.now(timezone.utc)
    await conn.execute(
        """INSERT INTO app_config (key, value, updated_at)
           VALUES ('account_epoch', $1::jsonb, NOW())
           ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()""",
        json.dumps({"epoch": CURRENT_EPOCH, "epoch_start": epoch_start.isoformat()}),
    )
    log.info("Seeded account epoch %s starting %s", CURRENT_EPOCH, epoch_start.isoformat())
    return epoch_start


async def get_account(conn) -> dict:
    """Current capital state (latest account row)."""
    row = await conn.fetchrow(
        "SELECT cash_available, cash_reserve, updated_at FROM account ORDER BY id DESC LIMIT 1"
    )
    if not row:
        return {"cash_available": 0.0, "cash_reserve": 0.0, "updated_at": None}
    return {
        "cash_available": float(row["cash_available"]),
        "cash_reserve": float(row["cash_reserve"]),
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


async def set_cash(conn, cash_available: float, cash_reserve: float) -> None:
    await conn.execute(
        "INSERT INTO account (cash_available, cash_reserve) VALUES ($1, $2)",
        cash_available, cash_reserve,
    )


async def record_decision(
    conn,
    *,
    action: str,                      # BUY | SELL | PASS | SKIP
    ticker: str | None = None,
    signal_id: int | None = None,
    quantity: int = 0,
    price: float | None = None,
    rationale: str = "",
) -> dict:
    """
    Log a decision. A BUY also decrements deployable cash and adds the position
    to the portfolio. Returns {decision_id, cash_after}. Raises on insufficient
    cash for a BUY so the caller can surface it.
    """
    # Paper vs live gate: until the track record qualifies, a BUY is a tracked
    # paper trade, not a real execution. The cash ledger still moves so the paper
    # P&L is real and feeds the learning loop — only the framing changes.
    from intelligence.trading_mode import get_trading_mode
    mode_info = await get_trading_mode(conn)
    is_paper = mode_info["mode"] == "PAPER"
    if is_paper and action == "BUY" and not rationale.startswith("[PAPER]"):
        rationale = f"[PAPER] {rationale}".rstrip()

    acct = await get_account(conn)
    cash = acct["cash_available"]
    cash_after = cash
    pnl: float | None = None
    outcome: str | None = None
    resolved_at = None

    if action == "BUY":
        if not price or quantity < 1:
            raise ValueError("BUY requires price and quantity >= 1")
        cost = price * quantity
        if cost > cash:
            raise ValueError(
                f"Insufficient cash: need ₹{cost:.2f}, have ₹{cash:.2f}"
            )
        cash_after = round(cash - cost, 2)
        await set_cash(conn, cash_after, acct["cash_reserve"])
        await add_to_portfolio(conn, ticker, quantity, price, notes=rationale)

    elif action == "SELL":
        if not price:
            raise ValueError("SELL requires a price")
        pos = await conn.fetchrow(
            "SELECT quantity, avg_price FROM portfolio WHERE ticker = $1 AND active = TRUE "
            "ORDER BY buy_date DESC LIMIT 1",
            ticker,
        )
        if not pos:
            raise ValueError(f"SELL rejected: no active position in {ticker}")
        held_qty = int(pos["quantity"])
        avg_price = float(pos["avg_price"])
        if quantity < 1 or quantity > held_qty:
            quantity = held_qty  # v1: full close
        proceeds = round(price * quantity, 2)
        pnl = round((price - avg_price) * quantity, 2)
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")
        resolved_at = datetime.now(timezone.utc)
        cash_after = round(cash + proceeds, 2)
        await set_cash(conn, cash_after, acct["cash_reserve"])
        await remove_from_portfolio(conn, ticker)
        # Resolve the originating open BUY decision so the ledger pairs up.
        await conn.execute(
            """
            UPDATE decisions SET outcome = $2, pnl = $3, resolved_at = $4
            WHERE id = (
                SELECT id FROM decisions
                WHERE ticker = $1 AND action = 'BUY' AND resolved_at IS NULL
                ORDER BY decided_at DESC LIMIT 1
            )
            """,
            ticker, outcome, pnl, resolved_at,
        )

    decision_id = await conn.fetchval(
        """
        INSERT INTO decisions (signal_id, ticker, action, quantity, price, cash_after,
                               rationale, outcome, pnl, decided_at, resolved_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING id
        """,
        signal_id, ticker, action, quantity, price, cash_after, rationale,
        outcome, pnl, datetime.now(timezone.utc), resolved_at,
    )

    # Mirror BUY/SELL into the activity lifecycle feed.
    if action in ("BUY", "SELL"):
        from intelligence.activity import log_activity
        await log_activity(
            conn, event_type="BOUGHT" if action == "BUY" else "SOLD",
            ticker=ticker, signal_id=signal_id, note=rationale,
            payload={
                "quantity": quantity, "price": price, "cash_after": cash_after,
                "pnl": pnl, "outcome": outcome,
                "mode": "PAPER" if is_paper else "LIVE",
            },
        )

    log.info("Decision %s: %s %s x%s @ %s (cash after ₹%s)%s",
             decision_id, action, ticker, quantity, price, cash_after,
             " [PAPER]" if is_paper else "")
    return {
        "decision_id": decision_id,
        "cash_after": cash_after,
        "mode": "PAPER" if is_paper else "LIVE",
    }


async def get_actionable_signals(conn, limit: int = 30, only_affordable: bool = False) -> list[dict]:
    """
    Latest BUY signals, ranked affordable-first then by confidence — matches the
    'annotate, don't restrict' policy: actionable-now ideas float to the top,
    premium ideas stay visible below.
    """
    where = "WHERE s.signal_type = 'BUY'"
    if only_affordable:
        where += " AND s.affordable = TRUE"
    # DISTINCT ON (s.ticker): a ticker can carry several signal rows within
    # the lookback window (one per timeframe, or repeated pipeline runs) —
    # without deduping, a couple of multi-timeframe tickers can dominate the
    # whole LIMIT and starve every other candidate (found live: 45 signals
    # produced across ~20 tickers, but the un-deduped top-20 held only 2
    # distinct tickers). Keep each ticker's single best (affordable-first,
    # then highest-confidence) row, then apply the caller's ranking/limit.
    rows = await conn.fetch(
        f"""
        SELECT id, ticker, timeframe, horizon_days, price_at_signal,
               target_price, stop_loss, final_confidence,
               affordable, shares_affordable, macro_sector_score,
               target_eta_days, expected_move_pct, predicted_path,
               fired_at, sector, name, components_json
        FROM (
            SELECT DISTINCT ON (s.ticker)
                   s.id, s.ticker, s.timeframe, s.horizon_days, s.price_at_signal,
                   s.target_price, s.stop_loss, s.final_confidence,
                   s.affordable, s.shares_affordable, s.macro_sector_score,
                   s.target_eta_days, s.expected_move_pct, s.predicted_path,
                   s.fired_at, st.sector, st.name, s.components_json
            FROM signals s
            JOIN stocks st ON st.ticker = s.ticker
            {where}
              AND st.active = TRUE      -- only currently-tradeable stocks (buyable on Angel One)
              AND s.fired_at >= NOW() - INTERVAL '1 day'
            ORDER BY s.ticker, s.affordable DESC NULLS LAST, s.final_confidence DESC
        ) best_per_ticker
        ORDER BY affordable DESC NULLS LAST, final_confidence DESC
        LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]


async def get_decisions(conn, limit: int = 50) -> list[dict]:
    """Decision history, most recent first."""
    rows = await conn.fetch(
        """
        SELECT d.id, d.ticker, d.action, d.quantity, d.price, d.cash_after,
               d.rationale, d.outcome, d.pnl, d.decided_at, d.resolved_at,
               s.timeframe, s.final_confidence
        FROM decisions d
        LEFT JOIN signals s ON s.id = d.signal_id
        ORDER BY d.decided_at DESC
        LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]
