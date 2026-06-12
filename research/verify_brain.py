"""
End-to-end verification of the autonomous brain loop (uses a synthetic ticker
so real data is untouched; cleans up after itself).
Run: venv\\Scripts\\python.exe -m research.verify_brain
"""
import asyncio
import asyncpg
from config import settings

TICKER = "ZZBRAINTEST"


async def main():
    conn = await asyncpg.connect(settings.DATABASE_DSN)

    # --- setup: synthetic stock, price history, fresh high-confidence signal ---
    await conn.execute(
        "INSERT INTO stocks (ticker, name, exchange) VALUES ($1,$1,'NSE') ON CONFLICT (ticker) DO NOTHING",
        TICKER,
    )
    await conn.execute(
        """INSERT INTO ohlcv_daily (time, ticker, open, high, low, close, volume, adj_close)
           VALUES (NOW(), $1, 10, 10.5, 9.8, 10.0, 1000, 10.0)
           ON CONFLICT (time, ticker) DO NOTHING""",
        TICKER,
    )
    sid = await conn.fetchval(
        """INSERT INTO signals (ticker, signal_type, timeframe, horizon_days, price_at_signal,
                                target_price, stop_loss, final_confidence, affordable,
                                shares_affordable, target_eta_days, fired_at)
           VALUES ($1,'BUY','3D',3, 10.0, 11.0, 9.5, 0.85, TRUE, 4, 2.5, NOW())
           RETURNING id""",
        TICKER,
    )
    print(f"1. seeded signal id={sid}")

    acct_before = await conn.fetchrow("SELECT cash_available FROM account ORDER BY id DESC LIMIT 1")
    print(f"   cash before: {acct_before['cash_available']}")

    # --- auto_trade ---
    from intelligence.auto_trader import auto_trade, auto_exit
    s1 = await auto_trade(conn)
    print(f"2. auto_trade #1: bought={s1['bought']} passed={[p['ticker'] for p in s1['passed']]}")
    assert any(b["ticker"] == TICKER for b in s1["bought"]), "expected AUTO BUY of test ticker"

    # idempotency
    s2 = await auto_trade(conn)
    assert not any(b["ticker"] == TICKER for b in s2["bought"]), "double-buy! idempotency broken"
    print(f"3. auto_trade #2 (rerun): idempotent OK (skipped={[x['why'] for x in s2['skipped'] if x['ticker']==TICKER]})")

    pos = await conn.fetchrow(
        "SELECT quantity, avg_price FROM portfolio WHERE ticker=$1 AND active=TRUE", TICKER)
    print(f"4. portfolio: {dict(pos)}")
    acct_mid = await conn.fetchrow("SELECT cash_available FROM account ORDER BY id DESC LIMIT 1")
    print(f"   cash after buy: {acct_mid['cash_available']}")

    # --- auto_exit on a fake EXIT verdict ---
    reviews = [{"ticker": TICKER, "verdict": "EXIT", "status": "target_hit",
                "current_price": 11.0, "progress_pct": 100.0, "reasoning": "target reached (test)"}]
    e1 = await auto_exit(reviews, conn)
    print(f"5. auto_exit: {e1}")
    assert any(s["ticker"] == TICKER for s in e1["sold"]), "expected AUTO SELL"

    sell = await conn.fetchrow(
        "SELECT action, quantity, price, pnl, outcome FROM decisions "
        "WHERE ticker=$1 AND action='SELL' ORDER BY id DESC LIMIT 1", TICKER)
    print(f"6. SELL decision: {dict(sell)}")
    buy_resolved = await conn.fetchrow(
        "SELECT outcome, pnl, resolved_at IS NOT NULL AS resolved FROM decisions "
        "WHERE ticker=$1 AND action='BUY' ORDER BY id DESC LIMIT 1", TICKER)
    print(f"7. originating BUY resolved: {dict(buy_resolved)}")
    acct_after = await conn.fetchrow("SELECT cash_available FROM account ORDER BY id DESC LIMIT 1")
    print(f"   cash after sell: {acct_after['cash_available']}")

    acts = await conn.fetch(
        "SELECT event_type FROM activity_log WHERE ticker=$1 ORDER BY id", TICKER)
    print(f"8. activity events: {[a['event_type'] for a in acts]}")

    # --- cleanup (restore cash to its original value) ---
    await conn.execute("DELETE FROM activity_log WHERE ticker=$1", TICKER)
    await conn.execute("DELETE FROM decisions WHERE ticker=$1", TICKER)
    await conn.execute("DELETE FROM portfolio WHERE ticker=$1", TICKER)
    await conn.execute("DELETE FROM signal_reasoning WHERE signal_id=$1", sid)
    await conn.execute("DELETE FROM signals WHERE ticker=$1", TICKER)
    await conn.execute("DELETE FROM ohlcv_daily WHERE ticker=$1", TICKER)
    await conn.execute("DELETE FROM stocks WHERE ticker=$1", TICKER)
    await conn.execute(
        "INSERT INTO account (cash_available, cash_reserve) SELECT $1, cash_reserve FROM account ORDER BY id DESC LIMIT 1",
        float(acct_before["cash_available"]),
    )
    print("9. cleanup done, cash restored")
    await conn.close()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
