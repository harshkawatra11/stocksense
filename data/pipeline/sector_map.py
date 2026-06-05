"""
Static NSE sector classification for the liquid universe.

Maps each ticker to one of the macro sectors used by intelligence/macro_context
(IT, BANKING, NBFC_FINANCE, AUTO, PHARMA, FMCG, METALS, ENERGY, POWER, REALTY,
CEMENT_INFRA, TELECOM, MEDIA, CHEMICALS).

Why static: NSE's official industry CSV is network-blocked on this machine
(same block as Bhavcopy). This covers the ~150 liquid names the pipeline
actually runs (MAX_TICKERS_PER_RUN). Ambiguous conglomerates/new-age names
are best-effort by primary business.

Apply to DB:
    python -m data.pipeline.sector_map
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

# ticker -> macro sector
SECTOR_MAP: dict[str, str] = {
    # IT / tech / internet platforms
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "LTIM": "IT", "PERSISTENT": "IT", "COFORGE": "IT", "MPHASIS": "IT",
    "KPITTECH": "IT", "TATAELXSI": "IT", "CYIENT": "IT", "DIXON": "IT",
    "INDIAMART": "IT", "NAUKRI": "IT",
    # Banks
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING",
    "KOTAKBANK": "BANKING", "INDUSINDBK": "BANKING", "BANKBARODA": "BANKING",
    "PNB": "BANKING", "CANBK": "BANKING", "UNIONBANK": "BANKING",
    "IDFCFIRSTB": "BANKING", "FEDERALBNK": "BANKING", "BANDHANBNK": "BANKING",
    "RBLBANK": "BANKING", "YESBANK": "BANKING",
    # NBFC / finance / insurance / capital markets
    "BAJFINANCE": "NBFC_FINANCE", "BAJAJFINSV": "NBFC_FINANCE", "LICI": "NBFC_FINANCE",
    "HDFCLIFE": "NBFC_FINANCE", "SBILIFE": "NBFC_FINANCE", "MUTHOOTFIN": "NBFC_FINANCE",
    "CHOLAFIN": "NBFC_FINANCE", "PAYTM": "NBFC_FINANCE", "POLICYBZR": "NBFC_FINANCE",
    "IRFC": "NBFC_FINANCE", "MFSL": "NBFC_FINANCE", "ICICIGI": "NBFC_FINANCE",
    "HDFCAMC": "NBFC_FINANCE", "NIPPONLIFE": "NBFC_FINANCE", "KFINTECH": "NBFC_FINANCE",
    "CAMS": "NBFC_FINANCE", "CDSL": "NBFC_FINANCE", "BSE": "NBFC_FINANCE",
    "MCX": "NBFC_FINANCE", "ANGELONE": "NBFC_FINANCE", "5PAISA": "NBFC_FINANCE",
    # Auto + components
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "HEROMOTOCO": "AUTO", "EICHERMOT": "AUTO",
    "BAJAJ-AUTO": "AUTO", "MOTHERSON": "AUTO", "BOSCHLTD": "AUTO", "BHARATFORG": "AUTO",
    "CUMMINSIND": "AUTO", "SCHAEFFLER": "AUTO",
    # Pharma / healthcare
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DIVISLAB": "PHARMA", "DRREDDY": "PHARMA",
    "APOLLOHOSP": "PHARMA", "AUROPHARMA": "PHARMA", "TORNTPHARM": "PHARMA",
    "IPCALAB": "PHARMA", "ALKEM": "PHARMA", "SYNGENE": "PHARMA", "BIOCON": "PHARMA",
    "LALPATHLAB": "PHARMA", "METROPOLIS": "PHARMA", "MANKIND": "PHARMA",
    "ZYDUSLIFE": "PHARMA", "LUPIN": "PHARMA", "ABBOTINDIA": "PHARMA", "PFIZER": "PHARMA",
    "AJANTPHARM": "PHARMA", "LAURUSLABS": "PHARMA", "GRANULES": "PHARMA", "NATCOPHARM": "PHARMA",
    # FMCG / consumer staples + discretionary retail
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "TATACONSUM": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG", "TITAN": "FMCG",
    "ZOMATO": "FMCG", "NYKAA": "FMCG", "TRENT": "FMCG", "DMART": "FMCG", "VBL": "FMCG",
    "PAGEIND": "FMCG", "RELAXO": "FMCG", "BATA": "FMCG", "HAVELLS": "FMCG",
    "VOLTAS": "FMCG", "CROMPTON": "FMCG",
    # Metals + mining
    "JSWSTEEL": "METALS", "TATASTEEL": "METALS", "VEDL": "METALS", "HINDALCO": "METALS",
    "NMDC": "METALS", "SAIL": "METALS", "APLAPOLLO": "METALS",
    # Energy — oil / gas / refiners / coal
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY", "HPCL": "ENERGY",
    "IOC": "ENERGY", "GAIL": "ENERGY", "PETRONET": "ENERGY", "COALINDIA": "ENERGY",
    "ADANIGAS": "ENERGY", "IGL": "ENERGY", "MGL": "ENERGY", "GUJGASLTD": "ENERGY",
    # Power / utilities / electricals-cables
    "NTPC": "POWER", "POWERGRID": "POWER", "TATAPOWER": "POWER", "TORNTPOWER": "POWER",
    "CESC": "POWER", "ADANIGREEN": "POWER", "POLYCAB": "POWER",
    # Realty
    "OBEROIRLTY": "REALTY", "DLF": "REALTY", "GODREJPROP": "REALTY",
    "PRESTIGE": "REALTY", "BRIGADE": "REALTY", "PHOENIXLTD": "REALTY",
    # Cement / construction / infra / logistics
    "LT": "CEMENT_INFRA", "ULTRACEMCO": "CEMENT_INFRA", "AMBUJACEM": "CEMENT_INFRA",
    "ACC": "CEMENT_INFRA", "SHREECEM": "CEMENT_INFRA", "DALMIACEM": "CEMENT_INFRA",
    "JKCEMENT": "CEMENT_INFRA", "RAMCOCEM": "CEMENT_INFRA", "GRASIM": "CEMENT_INFRA",
    "ADANIENT": "CEMENT_INFRA", "ADANIPORTS": "CEMENT_INFRA", "RVNL": "CEMENT_INFRA",
    "CONCOR": "CEMENT_INFRA", "IRCTC": "CEMENT_INFRA", "INDIGO": "CEMENT_INFRA",
    "BLUEDART": "CEMENT_INFRA", "ASTRAL": "CEMENT_INFRA", "SUPREMEIND": "CEMENT_INFRA",
    # Telecom
    "BHARTIARTL": "TELECOM",
    # Media
    "SUNTVNETWORK": "MEDIA", "PVRINOX": "MEDIA",
    # Chemicals / paints
    "ASIANPAINT": "CHEMICALS", "BERGEPAINT": "CHEMICALS", "PIDILITIND": "CHEMICALS",
}


async def populate_sectors(conn) -> int:
    """Write SECTOR_MAP into stocks.sector. Returns rows updated."""
    updated = 0
    for ticker, sector in SECTOR_MAP.items():
        res = await conn.execute(
            "UPDATE stocks SET sector = $2 WHERE ticker = $1", ticker, sector
        )
        # asyncpg returns 'UPDATE <n>'
        if res.split()[-1] != "0":
            updated += 1
    log.info("Populated sector for %d tickers", updated)
    return updated


if __name__ == "__main__":
    import asyncpg
    from config import settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    async def main():
        conn = await asyncpg.connect(settings.DATABASE_DSN)
        n = await populate_sectors(conn)
        total = await conn.fetchval("SELECT count(*) FROM stocks WHERE sector IS NOT NULL")
        by_sec = await conn.fetch(
            "SELECT sector, count(*) c FROM stocks WHERE sector IS NOT NULL GROUP BY sector ORDER BY c DESC"
        )
        await conn.close()
        print(f"\nUpdated {n} tickers; {total} now have a sector.")
        for r in by_sec:
            print(f"  {r['c']:>3}  {r['sector']}")

    asyncio.run(main())
