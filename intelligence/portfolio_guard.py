"""
Portfolio guard — ensures SELL signals only fire for held positions, and
enforces portfolio-level risk limits before a new paper BUY is allowed to
execute (see intelligence/auto_trader.py's auto_trade(), which calls these
as additional pre-trade guards on top of its existing budget-based sizing).
"""
import asyncpg
import os
from config import settings

DB_DSN = settings.DATABASE_DSN

# ------------------------------------------------------------------ #
# Portfolio-level risk limits                                          #
# ------------------------------------------------------------------ #
MAX_OPEN_POSITIONS = 8                  # hard cap on concurrent open positions
MAX_SECTOR_ALLOCATION_PCT = 0.25        # max share of deployable capital in one sector
DAILY_LOSS_CIRCUIT_BREAKER_PCT = 0.02   # halt new BUYs once today's P&L hits -2% of capital


async def get_portfolio_tickers(conn) -> set[str]:
    rows = await conn.fetch(
        "SELECT ticker FROM portfolio WHERE active = TRUE"
    )
    return {r["ticker"] for r in rows}


async def get_portfolio(conn) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT p.ticker, p.quantity, p.avg_price, p.buy_date, s.name, s.sector,
               (SELECT close FROM ohlcv_daily WHERE ticker = p.ticker ORDER BY time DESC LIMIT 1) as current_price
        FROM portfolio p
        JOIN stocks s ON s.ticker = p.ticker
        WHERE p.active = TRUE
        ORDER BY p.buy_date DESC
        """
    )
    result = []
    for r in rows:
        current = float(r["current_price"]) if r["current_price"] else float(r["avg_price"])
        avg = float(r["avg_price"])
        pnl = (current - avg) / avg * 100
        result.append({
            "ticker": r["ticker"],
            "name": r["name"],
            "sector": r["sector"],
            "quantity": r["quantity"],
            "avg_price": avg,
            "current_price": current,
            "pnl_pct": round(pnl, 2),
            "pnl_abs": round((current - avg) * r["quantity"], 2),
            "buy_date": r["buy_date"].isoformat() if r["buy_date"] else None,
        })
    return result


def is_held(ticker: str, portfolio_tickers: set[str]) -> bool:
    return ticker in portfolio_tickers


async def add_to_portfolio(conn, ticker: str, quantity: int, avg_price: float, notes: str = None):
    await conn.execute(
        """
        INSERT INTO portfolio (ticker, quantity, avg_price, buy_date, notes)
        VALUES ($1, $2, $3, NOW(), $4)
        ON CONFLICT DO NOTHING
        """,
        ticker, quantity, avg_price, notes,
    )


async def remove_from_portfolio(conn, ticker: str):
    await conn.execute(
        "UPDATE portfolio SET active = FALSE, updated_at = NOW() WHERE ticker = $1 AND active = TRUE",
        ticker,
    )


# ------------------------------------------------------------------ #
# Risk guards — called from auto_trader.auto_trade() before each BUY   #
# ------------------------------------------------------------------ #

async def check_position_limit(conn) -> tuple[bool, str]:
    """Block a new BUY once MAX_OPEN_POSITIONS concurrent positions are open."""
    open_count = int(await conn.fetchval(
        "SELECT COUNT(*) FROM portfolio WHERE active = TRUE"
    ) or 0)
    if open_count >= MAX_OPEN_POSITIONS:
        return False, f"max open positions reached ({open_count}/{MAX_OPEN_POSITIONS})"
    return True, ""


async def _deployable_capital(conn) -> tuple[float, float]:
    """(deployable_capital, positions_value) — cash + current value of all open positions."""
    acct = await conn.fetchrow("SELECT cash_available FROM account ORDER BY id DESC LIMIT 1")
    cash_available = float(acct["cash_available"]) if acct else 0.0
    positions_value = float(await conn.fetchval(
        """
        SELECT COALESCE(SUM(p.quantity * COALESCE(
            (SELECT close FROM ohlcv_daily WHERE ticker = p.ticker ORDER BY time DESC LIMIT 1),
            p.avg_price)), 0)
        FROM portfolio p WHERE p.active = TRUE
        """
    ) or 0)
    return cash_available + positions_value, positions_value


async def check_sector_concentration(conn, ticker: str, order_value: float) -> tuple[bool, str]:
    """
    Block a new BUY if it would push this ticker's sector above
    MAX_SECTOR_ALLOCATION_PCT of deployable capital (cash + all open positions'
    current value). Tickers with no known sector are never blocked (nothing to
    evaluate against).
    """
    sector = await conn.fetchval("SELECT sector FROM stocks WHERE ticker = $1", ticker)
    if not sector:
        return True, ""

    sector_rows = await conn.fetch(
        """
        SELECT p.quantity,
               COALESCE((SELECT close FROM ohlcv_daily WHERE ticker = p.ticker ORDER BY time DESC LIMIT 1),
                         p.avg_price) AS current_price
        FROM portfolio p
        JOIN stocks s ON s.ticker = p.ticker
        WHERE p.active = TRUE AND s.sector = $1
        """,
        sector,
    )
    sector_value = sum(float(r["quantity"]) * float(r["current_price"]) for r in sector_rows)

    deployable, _ = await _deployable_capital(conn)
    if deployable <= 0:
        return True, ""

    pct = (sector_value + order_value) / deployable
    if pct > MAX_SECTOR_ALLOCATION_PCT:
        return False, (f"sector '{sector}' would reach {pct*100:.0f}% of deployable capital "
                        f"(limit {MAX_SECTOR_ALLOCATION_PCT*100:.0f}%)")
    return True, ""


async def check_daily_loss_circuit_breaker(conn) -> tuple[bool, str]:
    """
    Block all new BUYs for the rest of the day once today's realized + open
    unrealized P&L hits -DAILY_LOSS_CIRCUIT_BREAKER_PCT of deployable capital.
    """
    deployable, _ = await _deployable_capital(conn)
    if deployable <= 0:
        return True, ""

    unrealized = 0.0
    for p in await get_portfolio(conn):
        unrealized += p["pnl_abs"]

    realized_today = float(await conn.fetchval(
        """
        SELECT COALESCE(SUM(pnl), 0) FROM decisions
        WHERE action = 'SELL' AND decided_at::date = CURRENT_DATE
        """
    ) or 0)

    daily_pnl_pct = (realized_today + unrealized) / deployable
    if daily_pnl_pct <= -DAILY_LOSS_CIRCUIT_BREAKER_PCT:
        return False, (f"daily-loss circuit breaker tripped: {daily_pnl_pct*100:.2f}% of capital "
                        f"(limit -{DAILY_LOSS_CIRCUIT_BREAKER_PCT*100:.0f}%) — no new BUYs today")
    return True, ""
