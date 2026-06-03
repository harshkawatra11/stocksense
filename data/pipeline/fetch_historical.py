"""
Pull 2000-2026 daily OHLCV for all NSE stocks via yfinance.
Stores into PostgreSQL. Run once to seed, then incremental.
"""
import yfinance as yf
import pandas as pd
import asyncpg
import asyncio
import os
import logging
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_DSN = os.getenv("DATABASE_DSN", "postgresql://stocksense:stocksense@localhost:5432/stocksense")

# Nifty 500 + additional liquid NSE stocks
NSE_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "SBIN", "INFY",
    "LICI", "ITC", "HINDUNILVR", "LT", "BAJFINANCE", "HCLTECH", "MARUTI",
    "SUNPHARMA", "ADANIENT", "KOTAKBANK", "TITAN", "ONGC", "NTPC", "WIPRO",
    "POWERGRID", "ULTRACEMCO", "BAJAJFINSV", "ASIANPAINT", "HDFCLIFE",
    "SBILIFE", "JSWSTEEL", "TATACONSUM", "COALINDIA", "NESTLEIND", "TATAMOTORS",
    "TECHM", "ADANIPORTS", "INDUSINDBK", "GRASIM", "CIPLA", "DIVISLAB",
    "HEROMOTOCO", "EICHERMOT", "BPCL", "BRITANNIA", "DRREDDY", "APOLLOHOSP",
    "BAJAJ-AUTO", "TATAPOWER", "ETERNAL", "PIDILITIND", "DABUR", "MARICO",
    "BERGEPAINT", "MUTHOOTFIN", "CHOLAFIN", "LTIM", "PERSISTENT", "COFORGE",
    "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR", "DELHIVERY", "CARTRADE",
    "IRCTC", "RAILVIKAS", "RVNL", "IRFC", "CONCOR", "CONTAINER",
    "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "IDFCFIRSTB", "FEDERALBNK",
    "BANDHANBNK", "RBLBANK", "YESBANK", "INDIAMART", "JUSTDIAL", "INFO EDGE",
    "NAUKRI", "TRENT", "DMART", "VBL", "UNITEDSPR", "PAGEIND",
    "HAVELLS", "VOLTAS", "BLUESTARCO", "CROMPTON", "AMBER", "DIXON",
    "VEDL", "HINDALCO", "NMDC", "SAIL", "JINDALSTEL", "TATASTEEL",
    "MOTHERSON", "BOSCHLTD", "BHARATFORG", "CUMMINSIND", "SCHAEFFLER",
    "AUROPHARMA", "TORNTPHARM", "IPCALAB", "ALKEM", "GLAXO",
    "SYNGENE", "BIOCON", "LALPATHLAB", "METROPOLIS", "THYROCARE",
    "HFCL", "TEJASNET", "STLTECH", "ROUTE", "TANLA",
    "MPHASIS", "KPITTECH", "TATAELXSI", "CYIENT", "NIIT",
    "OBEROIRLTY", "DLF", "GODREJPROP", "PRESTIGE", "BRIGADE",
    "PHOENIXLTD", "NESCO", "SUNTECK", "MAHLIFE",
    "INDIGO", "SPICEJET", "BLUEDART", "MAHINDCIE",
    "APLAPOLLO", "JINDALSAW", "WELCORP",
    "RELAXO", "BATA", "VMART", "SHOPERSTOP",
    "SUNTVNETWORK", "ZEEL", "PVRINOX", "INOXWIND",
    "TORNTPOWER", "CESC", "TATAPOWER", "ADANIGREEN",
    "ADANITRANS", "ADANIGAS", "IGL", "MGL", "GUJGASLTD",
    "HPCL", "IOC", "GAIL", "PETRONET", "GSPL",
    "MFSL", "ICICIGI", "HDFCAMC", "NIPPONLIFE", "UTIAMC",
]


def ticker_to_yf(ticker: str) -> str:
    ticker = ticker.replace(" ", "")
    return f"{ticker}.NS"


async def upsert_stock(conn, ticker: str, name: str = None):
    await conn.execute(
        """
        INSERT INTO stocks (ticker, name, exchange)
        VALUES ($1, $2, 'NSE')
        ON CONFLICT (ticker) DO NOTHING
        """,
        ticker,
        name or ticker,
    )


async def upsert_ohlcv_batch(conn, rows: list[tuple]):
    await conn.executemany(
        """
        INSERT INTO ohlcv_daily (time, ticker, open, high, low, close, volume, adj_close)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (time, ticker) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            adj_close = EXCLUDED.adj_close
        """,
        rows,
    )


async def fetch_and_store(conn, ticker: str, start: str = "2000-01-01", end: str = "2026-06-04"):
    yf_ticker = ticker_to_yf(ticker)
    try:
        df = yf.download(yf_ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            log.warning(f"No data for {ticker}")
            return 0

        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

        await upsert_stock(conn, ticker, ticker)

        rows = [
            (
                row.Index.to_pydatetime(),
                ticker,
                float(row.open) if pd.notna(row.open) else None,
                float(row.high) if pd.notna(row.high) else None,
                float(row.low) if pd.notna(row.low) else None,
                float(row.close) if pd.notna(row.close) else None,
                int(row.volume) if pd.notna(row.volume) else None,
                float(row.close) if pd.notna(row.close) else None,
            )
            for row in df.itertuples()
        ]

        await upsert_ohlcv_batch(conn, rows)
        log.info(f"{ticker}: stored {len(rows)} rows")
        return len(rows)

    except Exception as e:
        log.error(f"Error fetching {ticker}: {e}")
        return 0


async def run_full_historical(tickers: list[str] = None, workers: int = 5):
    tickers = tickers or NSE_TICKERS
    conn = await asyncpg.connect(DB_DSN)

    semaphore = asyncio.Semaphore(workers)

    async def bounded_fetch(ticker):
        async with semaphore:
            return await fetch_and_store(conn, ticker)

    tasks = [bounded_fetch(t) for t in tickers]
    results = await asyncio.gather(*tasks)

    total = sum(results)
    log.info(f"Historical pull complete. Total rows: {total:,}")
    await conn.close()


async def run_incremental(tickers: list[str] = None):
    """Pull yesterday's data for all tickers."""
    from datetime import timedelta
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=5)).isoformat()
    conn = await asyncpg.connect(DB_DSN)
    for ticker in (tickers or NSE_TICKERS):
        await fetch_and_store(conn, ticker, start=start, end=end)
    await conn.close()


if __name__ == "__main__":
    asyncio.run(run_full_historical())
