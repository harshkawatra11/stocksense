"""
Trading account + decision ledger.

Tracks the user's capital (deployable + reserve) and every decision taken on a
signal — including passes — so the learning loop can later judge not just what
was acted on but what was skipped. Built for the initial low-capital phase
(₹500 deployable) where most signals are informational until capital grows.

Tables: account, decisions (see data/db/schema_v2_live.sql).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from intelligence.portfolio_guard import add_to_portfolio, remove_from_portfolio

log = logging.getLogger(__name__)


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
    rows = await conn.fetch(
        f"""
        SELECT s.id, s.ticker, s.timeframe, s.horizon_days, s.price_at_signal,
               s.target_price, s.stop_loss, s.final_confidence,
               s.affordable, s.shares_affordable, s.macro_sector_score,
               s.target_eta_days, s.expected_move_pct, s.predicted_path,
               s.fired_at, st.sector, st.name, s.components_json
        FROM signals s
        JOIN stocks st ON st.ticker = s.ticker
        {where}
          AND st.active = TRUE          -- only currently-tradeable stocks (buyable on Angel One)
          AND s.fired_at >= NOW() - INTERVAL '1 day'
        ORDER BY s.affordable DESC NULLS LAST, s.final_confidence DESC
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
