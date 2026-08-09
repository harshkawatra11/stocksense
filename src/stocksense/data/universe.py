"""
Phase 0 universe: a fixed list of liquid NSE large/mid-cap symbols.

This is a deliberate, documented simplification against
docs/02-data-layer.md's point-in-time universe requirement. A true
point-in-time universe (correct historical index membership, delisted
names included as of the date they were still listed) requires NSE
archive ingestion, which is deferred past Phase 0. Using today's liquid
names for the full lookback period introduces survivorship bias: today's
survivors are systematically the historical winners.

This is recorded, not hidden: Phase 0's sweep results are read as "alpha
achievable on names that turned out to survive and stay liquid," which is
an upper bound on the addressable-universe result, not the final number.
docs/10-evaluation.md's point-in-time universe reconstruction is required
before any result from this universe is used for real capital.
"""

from __future__ import annotations

# NIFTY 100-ish liquid large/mid caps, hand-curated for Phase 0.
# Deliberately not the full ~2000+ tradeable universe (docs/02) — this
# keeps backfill time and API load manageable while still giving a
# large enough cross-section for meaningful cross-sectional ranking.
PHASE0_UNIVERSE: list[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
    "WIPRO", "BAJFINANCE", "HCLTECH", "M&M", "NTPC", "TATAMOTORS",
    "POWERGRID", "TATASTEEL", "ADANIENT", "ADANIPORTS", "JSWSTEEL", "ONGC",
    "COALINDIA", "BAJAJFINSV", "TECHM", "GRASIM", "HINDALCO", "DRREDDY",
    "CIPLA", "EICHERMOT", "BRITANNIA", "DIVISLAB", "BPCL", "HEROMOTOCO",
    "APOLLOHOSP", "INDUSINDBK", "TATACONSUM", "SBILIFE", "HDFCLIFE",
    "BAJAJ-AUTO", "UPL", "SHREECEM", "PIDILITIND", "DABUR", "GODREJCP",
    "MARICO", "SIEMENS", "HAVELLS", "DLF", "AMBUJACEM", "ACC",
    "BANKBARODA", "PNB", "CANBK", "IDFCFIRSTB", "FEDERALBNK", "BANDHANBNK",
    "MOTHERSON", "BOSCHLTD", "MRF", "TVSMOTOR", "ASHOKLEY", "BEL",
    "HAL", "GAIL", "IOC", "VEDL", "NMDC", "SAIL",
    "LUPIN", "AUROPHARMA", "BIOCON", "TORNTPHARM", "ALKEM",
    "COLPAL", "PGHH", "BERGEPAINT", "PAGEIND", "MUTHOOTFIN",
    "CHOLAFIN", "LICHSGFIN", "PFC", "RECLTD", "SRTRANSFIN",
    "ZEEL", "PVR", "INDIGO", "JUBLFOOD", "TRENT",
    "NAUKRI", "MPHASIS", "LTIM", "PERSISTENT", "COFORGE",
]

assert len(set(PHASE0_UNIVERSE)) == len(PHASE0_UNIVERSE), "duplicate symbols in universe"
