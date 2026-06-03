"""
Feature engineering for LightGBM. Takes OHLCV DataFrame, returns feature DataFrame.
All features computed in-place, no lookahead.
Includes F&O features: delivery_pct, oi_change_pct, pcr (put-call ratio).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


# Expanded SECTOR_MAP covering 200+ NSE stocks as fallback
SECTOR_MAP = {
    # Energy / Oil & Gas
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "IOC": "Energy",
    "HPCL": "Energy", "GAIL": "Energy", "PETRONET": "Energy", "GSPL": "Energy",
    "ADANIGAS": "Energy", "IGL": "Energy", "MGL": "Energy", "GUJGASLTD": "Energy",
    "MRPL": "Energy", "CPCL": "Energy",
    # Banking
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", "KOTAKBANK": "Banking",
    "AXISBANK": "Banking", "INDUSINDBK": "Banking", "BANKBARODA": "Banking", "PNB": "Banking",
    "CANBK": "Banking", "UNIONBANK": "Banking", "IDFCFIRSTB": "Banking", "FEDERALBNK": "Banking",
    "BANDHANBNK": "Banking", "RBLBANK": "Banking", "YESBANK": "Banking", "INDIANB": "Banking",
    "BANKINDIA": "Banking", "IOB": "Banking", "UCOBANK": "Banking", "CENTRALBK": "Banking",
    "MAHABANK": "Banking", "PSB": "Banking",
    # IT / Technology
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "PERSISTENT": "IT", "COFORGE": "IT", "MPHASIS": "IT",
    "KPITTECH": "IT", "TATAELXSI": "IT", "CYIENT": "IT", "NIIT": "IT",
    "TANLA": "IT", "ROUTE": "IT", "STLTECH": "IT", "HFCL": "IT", "TEJASNET": "IT",
    "MASTEK": "IT", "HEXAWARE": "IT", "L&TTECH": "IT", "BIRLASOFT": "IT",
    # Pharma / Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "AUROPHARMA": "Pharma", "TORNTPHARM": "Pharma", "IPCALAB": "Pharma", "ALKEM": "Pharma",
    "SYNGENE": "Pharma", "BIOCON": "Pharma", "LUPIN": "Pharma", "ZYDUSLIFE": "Pharma",
    "MANKIND": "Pharma", "ABBOTINDIA": "Pharma", "PFIZER": "Pharma", "AJANTPHARM": "Pharma",
    "LAURUSLABS": "Pharma", "GRANULES": "Pharma", "NATCOPHARM": "Pharma",
    "LALPATHLAB": "Healthcare", "METROPOLIS": "Healthcare", "THYROCARE": "Healthcare",
    "APOLLOHOSP": "Healthcare", "FORTIS": "Healthcare", "NARAYANAHRUL": "Healthcare",
    # Auto / Auto Ancillary
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "HEROMOTOCO": "Auto", "BAJAJ-AUTO": "Auto",
    "EICHERMOT": "Auto", "MOTHERSON": "Auto", "BOSCHLTD": "Auto", "BHARATFORG": "Auto",
    "CUMMINSIND": "Auto", "SCHAEFFLER": "Auto", "MAHINDCIE": "Auto", "EXIDEIND": "Auto",
    "AMARAJABAT": "Auto", "SUNDARMFIN": "Auto", "TIINDIA": "Auto", "MRF": "Auto",
    "APOLLOTYRE": "Auto", "CEATLTD": "Auto", "BALKRISIND": "Auto",
    # FMCG / Consumer
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "DABUR": "FMCG", "MARICO": "FMCG", "TATACONSUM": "FMCG", "EMAMILTD": "FMCG",
    "GODREJCP": "FMCG", "COLPAL": "FMCG", "PGHH": "FMCG", "GILLETTE": "FMCG",
    "VBL": "FMCG", "RADICO": "FMCG", "UNITEDBRW": "FMCG",
    # Retail / Consumer Discretionary
    "TRENT": "Retail", "DMART": "Retail", "PAGEIND": "Retail", "RELAXO": "Retail",
    "BATA": "Retail", "VMART": "Retail", "SHOPERSTOP": "Retail", "NYKAA": "Retail",
    # Telecom
    "BHARTIARTL": "Telecom", "INDIAMART": "Telecom",
    # Infrastructure / Capital Goods
    "LT": "Infra", "SIEMENS": "Infra", "ABB": "Infra", "BHEL": "Infra",
    "THERMAX": "Infra", "CUMMINSIND": "Infra", "APLAPOLLO": "Infra",
    "JINDALSAW": "Infra", "WELCORP": "Infra",
    # Power / Utilities
    "NTPC": "Power", "POWERGRID": "Power", "ADANIGREEN": "Power", "ADANITRANS": "Power",
    "TATAPOWER": "Power", "TORNTPOWER": "Power", "CESC": "Power",
    "NHPC": "Power", "SJVN": "Power", "INOXWIND": "Power",
    # Cement
    "ULTRACEMCO": "Cement", "AMBUJACEM": "Cement", "ACC": "Cement",
    "SHREECEM": "Cement", "DALMIACEM": "Cement", "JKCEMENT": "Cement",
    "RAMCOCEM": "Cement", "HEIDELBERG": "Cement", "BIRLACORPN": "Cement",
    # Metals / Mining
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "NMDC": "Metals", "SAIL": "Metals", "JINDALSTEL": "Metals",
    "COALINDIA": "Metals", "MOIL": "Metals", "NATIONALUM": "Metals",
    # Real Estate
    "DLF": "RealEstate", "GODREJPROP": "RealEstate", "OBEROIRLTY": "RealEstate",
    "PRESTIGE": "RealEstate", "BRIGADE": "RealEstate", "PHOENIXLTD": "RealEstate",
    "MAHLIFE": "RealEstate",
    # NBFC / Financial Services
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC",
    "MUTHOOTFIN": "NBFC", "MANAPPURAM": "NBFC", "SHRIRAMFIN": "NBFC",
    "LTFH": "NBFC", "RECLTD": "NBFC", "PFC": "NBFC", "IRFC": "NBFC",
    # Insurance
    "HDFCLIFE": "Insurance", "SBILIFE": "Insurance", "LICI": "Insurance",
    "MFSL": "Insurance", "ICICIGI": "Insurance", "ICICIPRULI": "Insurance",
    # Asset Management / Capital Markets
    "HDFCAMC": "AssetMgmt", "NIPPONLIFE": "AssetMgmt", "UTIAMC": "AssetMgmt",
    "CDSL": "AssetMgmt", "BSE": "AssetMgmt", "MCX": "AssetMgmt",
    "ANGELONE": "AssetMgmt", "CAMS": "AssetMgmt", "KFINTECH": "AssetMgmt",
    # Paints
    "ASIANPAINT": "Paints", "BERGEPAINT": "Paints", "KANSAINER": "Paints",
    "INDIGO": "Paints",  # actually aviation but misplaced here — overridden below
    # Aviation / Logistics
    "INDIGO": "Aviation", "BLUEDART": "Logistics", "CONCOR": "Logistics",
    "DELHIVERY": "Logistics", "IRCTC": "Logistics", "RVNL": "Logistics",
    # Consumer Electronics
    "HAVELLS": "ConsumerElec", "VOLTAS": "ConsumerElec", "CROMPTON": "ConsumerElec",
    "DIXON": "ConsumerElec", "AMBER": "ConsumerElec", "BLUESTARCO": "ConsumerElec",
    # Gems & Jewelry
    "TITAN": "Gems", "RAJESHEXPO": "Gems", "KALYANKJIL": "Gems",
    # Specialty Chemicals
    "PIDILITIND": "Chemicals", "ASTRAL": "Chemicals", "SUPREMEIND": "Chemicals",
    "POLYCAB": "Chemicals", "SRF": "Chemicals", "DEEPAKNTR": "Chemicals",
    "NAVINFLUOR": "Chemicals", "AARTI": "Chemicals", "VINATIORG": "Chemicals",
    # Media / Entertainment
    "SUNTVNETWORK": "Media", "ZEEL": "Media", "PVRINOX": "Media",
    # E-commerce / New Age
    "ZOMATO": "ECommerce", "PAYTM": "ECommerce", "POLICYBZR": "ECommerce",
    "NAUKRI": "ECommerce",
    # Ports / Adani Group
    "ADANIPORTS": "Ports", "ADANIENT": "Conglomerate",
}

SECTOR_ENCODING = {s: i for i, s in enumerate(sorted(set(SECTOR_MAP.values())))}


def compute_features(df: pd.DataFrame, ticker: str = "", fo_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Input: df with columns [open, high, low, close, volume], DatetimeIndex
           fo_df: optional DataFrame with F&O features indexed by date
                  (columns: total_oi, oi_change, put_oi, call_oi, pcr)
    Output: feature DataFrame aligned to same index
    """
    df = df.copy().sort_index()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan)

    feat = pd.DataFrame(index=df.index)

    # RSI
    feat["rsi_14"] = RSIIndicator(close, window=14).rsi()
    feat["rsi_7"] = RSIIndicator(close, window=7).rsi()
    feat["rsi_21"] = RSIIndicator(close, window=21).rsi()
    feat["rsi_14_slope"] = feat["rsi_14"].diff(3)

    # MACD
    macd = MACD(close)
    feat["macd"] = macd.macd()
    feat["macd_signal"] = macd.macd_signal()
    feat["macd_diff"] = macd.macd_diff()
    feat["macd_cross"] = (feat["macd"] > feat["macd_signal"]).astype(int)

    # Bollinger Bands
    bb = BollingerBands(close, window=20)
    feat["bb_upper"] = bb.bollinger_hband()
    feat["bb_lower"] = bb.bollinger_lband()
    feat["bb_mid"] = bb.bollinger_mavg()
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / feat["bb_mid"]
    feat["bb_position"] = (close - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"] + 1e-8)

    # Moving averages
    for w in [5, 10, 20, 50, 100, 200]:
        sma = SMAIndicator(close, window=w).sma_indicator()
        feat[f"sma_{w}"] = sma
        feat[f"dist_sma_{w}"] = (close - sma) / sma

    feat["ema_9"] = EMAIndicator(close, window=9).ema_indicator()
    feat["ema_21"] = EMAIndicator(close, window=21).ema_indicator()
    feat["ema_cross_9_21"] = (feat["ema_9"] > feat["ema_21"]).astype(int)

    # ATR
    atr = AverageTrueRange(high, low, close, window=14)
    feat["atr_14"] = atr.average_true_range()
    feat["atr_pct"] = feat["atr_14"] / close

    # OBV
    feat["obv"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    feat["obv_slope"] = feat["obv"].diff(5) / (feat["obv"].abs() + 1e-8)

    # Volume features
    feat["volume"] = volume
    feat["volume_sma20"] = volume.rolling(20).mean()
    feat["volume_ratio"] = volume / feat["volume_sma20"]
    feat["volume_spike"] = (feat["volume_ratio"] > 2.0).astype(int)

    # Price returns
    for d in [1, 3, 5, 10, 20, 60]:
        feat[f"return_{d}d"] = close.pct_change(d)

    # Candle patterns
    feat["candle_body"] = (df["close"] - df["open"]) / df["open"]
    feat["candle_upper_shadow"] = (high - df[["close", "open"]].max(axis=1)) / (high - low + 1e-8)
    feat["candle_lower_shadow"] = (df[["close", "open"]].min(axis=1) - low) / (high - low + 1e-8)

    # 52-week high/low
    feat["high_52w"] = high.rolling(252).max()
    feat["low_52w"] = low.rolling(252).min()
    feat["dist_52w_high"] = (close - feat["high_52w"]) / feat["high_52w"]
    feat["dist_52w_low"] = (close - feat["low_52w"]) / feat["low_52w"]

    # Stochastic
    stoch = StochasticOscillator(high, low, close)
    feat["stoch_k"] = stoch.stoch()
    feat["stoch_d"] = stoch.stoch_signal()

    # Calendar effects
    feat["day_of_week"] = df.index.dayofweek
    feat["month"] = df.index.month
    feat["quarter"] = df.index.quarter
    feat["week_of_year"] = df.index.isocalendar().week.astype(int)

    # Sector encoding (fallback to SECTOR_MAP if no DB sector available)
    sector = SECTOR_MAP.get(ticker, "Other")
    feat["sector"] = SECTOR_ENCODING.get(sector, len(SECTOR_ENCODING))

    # F&O features — fill with NaN if not provided
    if fo_df is not None and not fo_df.empty:
        fo_aligned = fo_df.reindex(df.index)
        total_oi = fo_aligned.get("total_oi", pd.Series(dtype=float, index=df.index))
        oi_change = fo_aligned.get("oi_change", pd.Series(dtype=float, index=df.index))
        call_oi = fo_aligned.get("call_oi", pd.Series(dtype=float, index=df.index))
        pcr = fo_aligned.get("pcr", pd.Series(dtype=float, index=df.index))
        # delivery_pct: estimated from volume vs OI (placeholder when not directly available)
        delivery = fo_aligned.get("delivery_pct", pd.Series(dtype=float, index=df.index))

        feat["oi_change_pct"] = oi_change / (total_oi.replace(0, np.nan))
        feat["pcr"] = pcr
        feat["delivery_pct"] = delivery
    else:
        feat["oi_change_pct"] = np.nan
        feat["pcr"] = np.nan
        feat["delivery_pct"] = np.nan

    # Target: next-day return (for training only, drop before inference)
    feat["target_1d_return"] = close.pct_change(1).shift(-1)
    feat["target_buy"] = (feat["target_1d_return"] > 0.015).astype(int)

    return feat


def compute_features_with_sector(
    df: pd.DataFrame,
    ticker: str = "",
    sector: str | None = None,
    fo_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Same as compute_features but allows passing sector from DB directly,
    bypassing the static SECTOR_MAP.
    """
    feat = compute_features(df, ticker=ticker, fo_df=fo_df)
    if sector is not None:
        feat["sector"] = SECTOR_ENCODING.get(sector, len(SECTOR_ENCODING))
    return feat


def get_feature_columns() -> list[str]:
    """All feature column names (exclude targets, no F&O)."""
    return [
        "rsi_14", "rsi_7", "rsi_21", "rsi_14_slope",
        "macd", "macd_signal", "macd_diff", "macd_cross",
        "bb_width", "bb_position",
        "dist_sma_5", "dist_sma_10", "dist_sma_20", "dist_sma_50", "dist_sma_100", "dist_sma_200",
        "ema_cross_9_21",
        "atr_pct", "obv_slope",
        "volume_ratio", "volume_spike",
        "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_60d",
        "candle_body", "candle_upper_shadow", "candle_lower_shadow",
        "dist_52w_high", "dist_52w_low",
        "stoch_k", "stoch_d",
        "day_of_week", "month", "quarter",
        "sector",
    ]


def get_all_feature_columns() -> list[str]:
    """All feature columns including F&O features."""
    return get_feature_columns() + [
        "oi_change_pct",
        "pcr",
        "delivery_pct",
    ]
