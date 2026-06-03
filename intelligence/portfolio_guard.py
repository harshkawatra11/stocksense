"""
Portfolio guard — ensures SELL signals only fire for held positions.
"""
import asyncpg
import os

DB_DSN = os.getenv("DATABASE_DSN", "postgresql://stocksense:stocksense@localhost:5432/stocksense")


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
