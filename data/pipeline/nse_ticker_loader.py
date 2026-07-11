"""
NSE equity ticker loader.
Loads the complete NSE equity list from public NSE CSV or falls back to hardcoded list.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

NSE_EQUITY_CSV_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

# Fallback list: 150+ major NSE tickers (Nifty 500 representative sample)
FALLBACK_TICKERS = [
    {"ticker": "RELIANCE", "name": "Reliance Industries Ltd", "series": "EQ", "isin": "INE002A01018"},
    {"ticker": "TCS", "name": "Tata Consultancy Services Ltd", "series": "EQ", "isin": "INE467B01029"},
    {"ticker": "HDFCBANK", "name": "HDFC Bank Ltd", "series": "EQ", "isin": "INE040A01034"},
    {"ticker": "ICICIBANK", "name": "ICICI Bank Ltd", "series": "EQ", "isin": "INE090A01021"},
    {"ticker": "BHARTIARTL", "name": "Bharti Airtel Ltd", "series": "EQ", "isin": "INE397D01024"},
    {"ticker": "SBIN", "name": "State Bank of India", "series": "EQ", "isin": "INE062A01020"},
    {"ticker": "INFY", "name": "Infosys Ltd", "series": "EQ", "isin": "INE009A01021"},
    {"ticker": "LICI", "name": "Life Insurance Corporation of India", "series": "EQ", "isin": "INE0J1Y01017"},
    {"ticker": "ITC", "name": "ITC Ltd", "series": "EQ", "isin": "INE154A01025"},
    {"ticker": "HINDUNILVR", "name": "Hindustan Unilever Ltd", "series": "EQ", "isin": "INE030A01027"},
    {"ticker": "LT", "name": "Larsen & Toubro Ltd", "series": "EQ", "isin": "INE018A01030"},
    {"ticker": "BAJFINANCE", "name": "Bajaj Finance Ltd", "series": "EQ", "isin": "INE296A01024"},
    {"ticker": "HCLTECH", "name": "HCL Technologies Ltd", "series": "EQ", "isin": "INE860A01027"},
    {"ticker": "MARUTI", "name": "Maruti Suzuki India Ltd", "series": "EQ", "isin": "INE585B01010"},
    {"ticker": "SUNPHARMA", "name": "Sun Pharmaceutical Industries Ltd", "series": "EQ", "isin": "INE044A01036"},
    {"ticker": "ADANIENT", "name": "Adani Enterprises Ltd", "series": "EQ", "isin": "INE423A01024"},
    {"ticker": "KOTAKBANK", "name": "Kotak Mahindra Bank Ltd", "series": "EQ", "isin": "INE237A01028"},
    {"ticker": "TITAN", "name": "Titan Company Ltd", "series": "EQ", "isin": "INE280A01028"},
    {"ticker": "ONGC", "name": "Oil and Natural Gas Corporation Ltd", "series": "EQ", "isin": "INE213A01029"},
    {"ticker": "NTPC", "name": "NTPC Ltd", "series": "EQ", "isin": "INE733E01010"},
    {"ticker": "WIPRO", "name": "Wipro Ltd", "series": "EQ", "isin": "INE075A01022"},
    {"ticker": "POWERGRID", "name": "Power Grid Corporation of India Ltd", "series": "EQ", "isin": "INE752E01010"},
    {"ticker": "ULTRACEMCO", "name": "UltraTech Cement Ltd", "series": "EQ", "isin": "INE481G01011"},
    {"ticker": "BAJAJFINSV", "name": "Bajaj Finserv Ltd", "series": "EQ", "isin": "INE918I01026"},
    {"ticker": "ASIANPAINT", "name": "Asian Paints Ltd", "series": "EQ", "isin": "INE021A01026"},
    {"ticker": "HDFCLIFE", "name": "HDFC Life Insurance Company Ltd", "series": "EQ", "isin": "INE795G01014"},
    {"ticker": "SBILIFE", "name": "SBI Life Insurance Company Ltd", "series": "EQ", "isin": "INE123W01016"},
    {"ticker": "JSWSTEEL", "name": "JSW Steel Ltd", "series": "EQ", "isin": "INE019A01038"},
    {"ticker": "TATACONSUM", "name": "Tata Consumer Products Ltd", "series": "EQ", "isin": "INE192A01025"},
    {"ticker": "COALINDIA", "name": "Coal India Ltd", "series": "EQ", "isin": "INE522F01014"},
    {"ticker": "NESTLEIND", "name": "Nestle India Ltd", "series": "EQ", "isin": "INE239A01024"},
    {"ticker": "TATAMOTORS", "name": "Tata Motors Ltd", "series": "EQ", "isin": "INE155A01022"},
    {"ticker": "TECHM", "name": "Tech Mahindra Ltd", "series": "EQ", "isin": "INE669C01036"},
    {"ticker": "ADANIPORTS", "name": "Adani Ports and Special Economic Zone Ltd", "series": "EQ", "isin": "INE742F01042"},
    {"ticker": "INDUSINDBK", "name": "IndusInd Bank Ltd", "series": "EQ", "isin": "INE095A01012"},
    {"ticker": "GRASIM", "name": "Grasim Industries Ltd", "series": "EQ", "isin": "INE047A01021"},
    {"ticker": "CIPLA", "name": "Cipla Ltd", "series": "EQ", "isin": "INE059A01026"},
    {"ticker": "DIVISLAB", "name": "Divi's Laboratories Ltd", "series": "EQ", "isin": "INE361B01024"},
    {"ticker": "HEROMOTOCO", "name": "Hero MotoCorp Ltd", "series": "EQ", "isin": "INE158A01026"},
    {"ticker": "EICHERMOT", "name": "Eicher Motors Ltd", "series": "EQ", "isin": "INE066A01021"},
    {"ticker": "BPCL", "name": "Bharat Petroleum Corporation Ltd", "series": "EQ", "isin": "INE029A01011"},
    {"ticker": "BRITANNIA", "name": "Britannia Industries Ltd", "series": "EQ", "isin": "INE216A01030"},
    {"ticker": "DRREDDY", "name": "Dr. Reddy's Laboratories Ltd", "series": "EQ", "isin": "INE089A01023"},
    {"ticker": "APOLLOHOSP", "name": "Apollo Hospitals Enterprise Ltd", "series": "EQ", "isin": "INE437A01024"},
    {"ticker": "BAJAJ-AUTO", "name": "Bajaj Auto Ltd", "series": "EQ", "isin": "INE917I01010"},
    {"ticker": "TATAPOWER", "name": "Tata Power Company Ltd", "series": "EQ", "isin": "INE245A01021"},
    {"ticker": "PIDILITIND", "name": "Pidilite Industries Ltd", "series": "EQ", "isin": "INE318A01026"},
    {"ticker": "DABUR", "name": "Dabur India Ltd", "series": "EQ", "isin": "INE016A01026"},
    {"ticker": "MARICO", "name": "Marico Ltd", "series": "EQ", "isin": "INE196A01026"},
    {"ticker": "BERGEPAINT", "name": "Berger Paints India Ltd", "series": "EQ", "isin": "INE463A01038"},
    {"ticker": "MUTHOOTFIN", "name": "Muthoot Finance Ltd", "series": "EQ", "isin": "INE414G01012"},
    {"ticker": "CHOLAFIN", "name": "Cholamandalam Investment and Finance Company Ltd", "series": "EQ", "isin": "INE121A01024"},
    {"ticker": "LTIM", "name": "LTIMindtree Ltd", "series": "EQ", "isin": "INE214T01019"},
    {"ticker": "PERSISTENT", "name": "Persistent Systems Ltd", "series": "EQ", "isin": "INE262H01021"},
    {"ticker": "COFORGE", "name": "Coforge Ltd", "series": "EQ", "isin": "INE591G01017"},
    {"ticker": "ZOMATO", "name": "Zomato Ltd", "series": "EQ", "isin": "INE758T01015"},
    {"ticker": "NYKAA", "name": "FSN E-Commerce Ventures Ltd", "series": "EQ", "isin": "INE388Y01029"},
    {"ticker": "PAYTM", "name": "One 97 Communications Ltd", "series": "EQ", "isin": "INE982J01020"},
    {"ticker": "POLICYBZR", "name": "PB Fintech Ltd", "series": "EQ", "isin": "INE417T01026"},
    {"ticker": "IRCTC", "name": "Indian Railway Catering and Tourism Corporation Ltd", "series": "EQ", "isin": "INE335Y01020"},
    {"ticker": "RVNL", "name": "Rail Vikas Nigam Ltd", "series": "EQ", "isin": "INE415G01027"},
    {"ticker": "IRFC", "name": "Indian Railway Finance Corporation Ltd", "series": "EQ", "isin": "INE053F01010"},
    {"ticker": "CONCOR", "name": "Container Corporation of India Ltd", "series": "EQ", "isin": "INE111A01025"},
    {"ticker": "BANKBARODA", "name": "Bank of Baroda", "series": "EQ", "isin": "INE028A01039"},
    {"ticker": "PNB", "name": "Punjab National Bank", "series": "EQ", "isin": "INE160A01022"},
    {"ticker": "CANBK", "name": "Canara Bank", "series": "EQ", "isin": "INE476A01022"},
    {"ticker": "UNIONBANK", "name": "Union Bank of India", "series": "EQ", "isin": "INE692A01016"},
    {"ticker": "IDFCFIRSTB", "name": "IDFC First Bank Ltd", "series": "EQ", "isin": "INE0DD801020"},
    {"ticker": "FEDERALBNK", "name": "Federal Bank Ltd", "series": "EQ", "isin": "INE171A01029"},
    {"ticker": "BANDHANBNK", "name": "Bandhan Bank Ltd", "series": "EQ", "isin": "INE545U01014"},
    {"ticker": "RBLBANK", "name": "RBL Bank Ltd", "series": "EQ", "isin": "INE976G01028"},
    {"ticker": "YESBANK", "name": "Yes Bank Ltd", "series": "EQ", "isin": "INE528G01035"},
    {"ticker": "INDIAMART", "name": "Indiamart Intermesh Ltd", "series": "EQ", "isin": "INE065J01010"},
    {"ticker": "NAUKRI", "name": "Info Edge (India) Ltd", "series": "EQ", "isin": "INE663F01024"},
    {"ticker": "TRENT", "name": "Trent Ltd", "series": "EQ", "isin": "INE849A01020"},
    {"ticker": "DMART", "name": "Avenue Supermarts Ltd", "series": "EQ", "isin": "INE192R01011"},
    {"ticker": "VBL", "name": "Varun Beverages Ltd", "series": "EQ", "isin": "INE200M01021"},
    {"ticker": "PAGEIND", "name": "Page Industries Ltd", "series": "EQ", "isin": "INE761H01022"},
    {"ticker": "HAVELLS", "name": "Havells India Ltd", "series": "EQ", "isin": "INE176B01034"},
    {"ticker": "VOLTAS", "name": "Voltas Ltd", "series": "EQ", "isin": "INE226A01021"},
    {"ticker": "CROMPTON", "name": "Crompton Greaves Consumer Electricals Ltd", "series": "EQ", "isin": "INE299U01018"},
    {"ticker": "DIXON", "name": "Dixon Technologies (India) Ltd", "series": "EQ", "isin": "INE935N01012"},
    {"ticker": "VEDL", "name": "Vedanta Ltd", "series": "EQ", "isin": "INE205A01025"},
    {"ticker": "HINDALCO", "name": "Hindalco Industries Ltd", "series": "EQ", "isin": "INE038A01020"},
    {"ticker": "NMDC", "name": "NMDC Ltd", "series": "EQ", "isin": "INE584A01023"},
    {"ticker": "SAIL", "name": "Steel Authority of India Ltd", "series": "EQ", "isin": "INE114A01011"},
    {"ticker": "TATASTEEL", "name": "Tata Steel Ltd", "series": "EQ", "isin": "INE081A01020"},
    {"ticker": "MOTHERSON", "name": "Samvardhana Motherson International Ltd", "series": "EQ", "isin": "INE775A01035"},
    {"ticker": "BOSCHLTD", "name": "Bosch Ltd", "series": "EQ", "isin": "INE323A01026"},
    {"ticker": "BHARATFORG", "name": "Bharat Forge Ltd", "series": "EQ", "isin": "INE465A01025"},
    {"ticker": "CUMMINSIND", "name": "Cummins India Ltd", "series": "EQ", "isin": "INE298A01020"},
    {"ticker": "AUROPHARMA", "name": "Aurobindo Pharma Ltd", "series": "EQ", "isin": "INE406A01037"},
    {"ticker": "TORNTPHARM", "name": "Torrent Pharmaceuticals Ltd", "series": "EQ", "isin": "INE685A01028"},
    {"ticker": "IPCALAB", "name": "IPCA Laboratories Ltd", "series": "EQ", "isin": "INE571A01020"},
    {"ticker": "ALKEM", "name": "Alkem Laboratories Ltd", "series": "EQ", "isin": "INE540L01014"},
    {"ticker": "SYNGENE", "name": "Syngene International Ltd", "series": "EQ", "isin": "INE398R01022"},
    {"ticker": "BIOCON", "name": "Biocon Ltd", "series": "EQ", "isin": "INE376G01013"},
    {"ticker": "LALPATHLAB", "name": "Dr Lal PathLabs Ltd", "series": "EQ", "isin": "INE600L01024"},
    {"ticker": "METROPOLIS", "name": "Metropolis Healthcare Ltd", "series": "EQ", "isin": "INE875R01011"},
    {"ticker": "MPHASIS", "name": "Mphasis Ltd", "series": "EQ", "isin": "INE356A01018"},
    {"ticker": "KPITTECH", "name": "KPIT Technologies Ltd", "series": "EQ", "isin": "INE0IV801015"},
    {"ticker": "TATAELXSI", "name": "Tata Elxsi Ltd", "series": "EQ", "isin": "INE670A01012"},
    {"ticker": "CYIENT", "name": "Cyient Ltd", "series": "EQ", "isin": "INE136B01020"},
    {"ticker": "OBEROIRLTY", "name": "Oberoi Realty Ltd", "series": "EQ", "isin": "INE093I01010"},
    {"ticker": "DLF", "name": "DLF Ltd", "series": "EQ", "isin": "INE271C01023"},
    {"ticker": "GODREJPROP", "name": "Godrej Properties Ltd", "series": "EQ", "isin": "INE484J01027"},
    {"ticker": "PRESTIGE", "name": "Prestige Estates Projects Ltd", "series": "EQ", "isin": "INE811K01011"},
    {"ticker": "BRIGADE", "name": "Brigade Enterprises Ltd", "series": "EQ", "isin": "INE791I01019"},
    {"ticker": "PHOENIXLTD", "name": "The Phoenix Mills Ltd", "series": "EQ", "isin": "INE211B01039"},
    {"ticker": "INDIGO", "name": "InterGlobe Aviation Ltd", "series": "EQ", "isin": "INE646L01027"},
    {"ticker": "BLUEDART", "name": "Blue Dart Express Ltd", "series": "EQ", "isin": "INE233B01017"},
    {"ticker": "APLAPOLLO", "name": "APL Apollo Tubes Ltd", "series": "EQ", "isin": "INE702C01027"},
    {"ticker": "RELAXO", "name": "Relaxo Footwears Ltd", "series": "EQ", "isin": "INE131B01039"},
    {"ticker": "BATA", "name": "Bata India Ltd", "series": "EQ", "isin": "INE176A01028"},
    {"ticker": "SUNTVNETWORK", "name": "Sun TV Network Ltd", "series": "EQ", "isin": "INE424H01027"},
    {"ticker": "PVRINOX", "name": "PVR INOX Ltd", "series": "EQ", "isin": "INE191H01014"},
    {"ticker": "TORNTPOWER", "name": "Torrent Power Ltd", "series": "EQ", "isin": "INE813H01021"},
    {"ticker": "CESC", "name": "CESC Ltd", "series": "EQ", "isin": "INE178A01016"},
    {"ticker": "ADANIGREEN", "name": "Adani Green Energy Ltd", "series": "EQ", "isin": "INE364U01010"},
    {"ticker": "ADANIGAS", "name": "Adani Total Gas Ltd", "series": "EQ", "isin": "INE399L01023"},
    {"ticker": "IGL", "name": "Indraprastha Gas Ltd", "series": "EQ", "isin": "INE203G01027"},
    {"ticker": "MGL", "name": "Mahanagar Gas Ltd", "series": "EQ", "isin": "INE002S01010"},
    {"ticker": "GUJGASLTD", "name": "Gujarat Gas Ltd", "series": "EQ", "isin": "INE844O01030"},
    {"ticker": "HPCL", "name": "Hindustan Petroleum Corporation Ltd", "series": "EQ", "isin": "INE094A01015"},
    {"ticker": "IOC", "name": "Indian Oil Corporation Ltd", "series": "EQ", "isin": "INE242A01010"},
    {"ticker": "GAIL", "name": "GAIL (India) Ltd", "series": "EQ", "isin": "INE129A01019"},
    {"ticker": "PETRONET", "name": "Petronet LNG Ltd", "series": "EQ", "isin": "INE347G01014"},
    {"ticker": "MFSL", "name": "Max Financial Services Ltd", "series": "EQ", "isin": "INE180A01020"},
    {"ticker": "ICICIGI", "name": "ICICI Lombard General Insurance Company Ltd", "series": "EQ", "isin": "INE765G01017"},
    {"ticker": "HDFCAMC", "name": "HDFC Asset Management Company Ltd", "series": "EQ", "isin": "INE127D01025"},
    {"ticker": "NIPPONLIFE", "name": "Nippon Life India Asset Management Ltd", "series": "EQ", "isin": "INE298J01013"},
    {"ticker": "SCHAEFFLER", "name": "Schaeffler India Ltd", "series": "EQ", "isin": "INE513A01022"},
    {"ticker": "AMBUJACEM", "name": "Ambuja Cements Ltd", "series": "EQ", "isin": "INE079A01024"},
    {"ticker": "ACC", "name": "ACC Ltd", "series": "EQ", "isin": "INE012A01025"},
    {"ticker": "SHREECEM", "name": "Shree Cement Ltd", "series": "EQ", "isin": "INE070A01015"},
    {"ticker": "DALMIACEM", "name": "Dalmia Bharat Ltd", "series": "EQ", "isin": "INE00R701025"},
    {"ticker": "JKCEMENT", "name": "JK Cement Ltd", "series": "EQ", "isin": "INE823G01014"},
    {"ticker": "RAMCOCEM", "name": "The Ramco Cements Ltd", "series": "EQ", "isin": "INE331A01037"},
    {"ticker": "ASTRAL", "name": "Astral Ltd", "series": "EQ", "isin": "INE006I01046"},
    {"ticker": "SUPREMEIND", "name": "Supreme Industries Ltd", "series": "EQ", "isin": "INE195A01028"},
    {"ticker": "POLYCAB", "name": "Polycab India Ltd", "series": "EQ", "isin": "INE455K01017"},
    {"ticker": "KFINTECH", "name": "KFin Technologies Ltd", "series": "EQ", "isin": "INE138Y01010"},
    {"ticker": "CAMS", "name": "Computer Age Management Services Ltd", "series": "EQ", "isin": "INE596I01012"},
    {"ticker": "CDSL", "name": "Central Depository Services (India) Ltd", "series": "EQ", "isin": "INE736A01011"},
    {"ticker": "BSE", "name": "BSE Ltd", "series": "EQ", "isin": "INE118H01025"},
    {"ticker": "MCX", "name": "Multi Commodity Exchange of India Ltd", "series": "EQ", "isin": "INE745G01035"},
    {"ticker": "ANGELONE", "name": "Angel One Ltd", "series": "EQ", "isin": "INE732I01013"},
    {"ticker": "5PAISA", "name": "5paisa Capital Ltd", "series": "EQ", "isin": "INE618L01018"},
    {"ticker": "MANKIND", "name": "Mankind Pharma Ltd", "series": "EQ", "isin": "INE634S01028"},
    {"ticker": "ZYDUSLIFE", "name": "Zydus Lifesciences Ltd", "series": "EQ", "isin": "INE010B01027"},
    {"ticker": "LUPIN", "name": "Lupin Ltd", "series": "EQ", "isin": "INE326A01037"},
    {"ticker": "ABBOTINDIA", "name": "Abbott India Ltd", "series": "EQ", "isin": "INE358A01014"},
    {"ticker": "PFIZER", "name": "Pfizer Ltd", "series": "EQ", "isin": "INE182A01018"},
    {"ticker": "AJANTPHARM", "name": "Ajanta Pharma Ltd", "series": "EQ", "isin": "INE031B01049"},
    {"ticker": "LAURUSLABS", "name": "Laurus Labs Ltd", "series": "EQ", "isin": "INE947Q01010"},
    {"ticker": "GRANULES", "name": "Granules India Ltd", "series": "EQ", "isin": "INE101D01020"},
    {"ticker": "NATCOPHARM", "name": "Natco Pharma Ltd", "series": "EQ", "isin": "INE987B01026"},
]


def load_from_nse_csv() -> list[dict[str, Any]]:
    """
    Downloads NSE equity list CSV from public NSE URL.
    Returns list of dicts with keys: ticker, name, series, isin
    Only EQ series stocks are included.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    try:
        resp = requests.get(NSE_EQUITY_CSV_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        import pandas as pd
        df = pd.read_csv(io.StringIO(resp.text))
        # NSE CSV columns: SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING,PAID UP VALUE,MARKET LOT,ISIN NUMBER,FACE VALUE
        df.columns = [c.strip() for c in df.columns]
        # Filter EQ series only
        df = df[df["SERIES"].str.strip() == "EQ"].copy()
        result = []
        for _, row in df.iterrows():
            ticker = str(row.get("SYMBOL", "")).strip()
            name = str(row.get("NAME OF COMPANY", "")).strip()
            isin = str(row.get("ISIN NUMBER", "")).strip()
            if ticker:
                result.append({"ticker": ticker, "name": name, "series": "EQ", "isin": isin})
        log.info(f"Loaded {len(result)} EQ tickers from NSE CSV")
        return result
    except Exception as e:
        log.warning(f"Failed to load NSE equity CSV: {e}. Using fallback list.")
        return load_fallback_list()


def load_fallback_list() -> list[dict[str, Any]]:
    """Returns hardcoded list of 150+ major NSE tickers as fallback."""
    log.info(f"Using fallback ticker list with {len(FALLBACK_TICKERS)} tickers")
    return FALLBACK_TICKERS


async def sync_to_db(conn, tickers: list[dict[str, Any]]) -> int:
    """
    Upserts all tickers into stocks table.
    Returns number of tickers upserted.
    """
    count = 0
    for t in tickers:
        await conn.execute(
            """
            INSERT INTO stocks (ticker, name, exchange, active, isin)
            VALUES ($1, $2, 'NSE', TRUE, $3)
            ON CONFLICT (ticker) DO UPDATE SET
                name = EXCLUDED.name,
                active = TRUE,
                isin = COALESCE(EXCLUDED.isin, stocks.isin)
            """,
            t["ticker"],
            t["name"],
            t.get("isin") or None,
        )
        count += 1
    log.info(f"Upserted {count} tickers into stocks table")
    return count


async def get_active_tickers(conn) -> list[str]:
    """Returns list of active ticker symbols from DB."""
    rows = await conn.fetch(
        "SELECT ticker FROM stocks WHERE active = TRUE AND exchange = 'NSE' ORDER BY ticker"
    )
    return [r["ticker"] for r in rows]


if __name__ == "__main__":
    import asyncio
    import asyncpg
    from config import settings

    async def main():
        tickers = load_from_nse_csv()
        conn = await asyncpg.connect(settings.DATABASE_DSN)
        await sync_to_db(conn, tickers)
        active = await get_active_tickers(conn)
        print(f"Active tickers in DB: {len(active)}")
        await conn.close()

    asyncio.run(main())
