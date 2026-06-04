"""
Feature engineering for LightGBM. Pure pandas/numpy — no external TA library required.
All features computed in-place, no lookahead bias.
Includes F&O features: delivery_pct, oi_change_pct, pcr (put-call ratio).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sector map — 200+ NSE stocks
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "IOC": "Energy",
    "HPCL": "Energy", "GAIL": "Energy", "PETRONET": "Energy", "IGL": "Energy",
    "MGL": "Energy", "GUJGASLTD": "Energy", "MRPL": "Energy",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BANKBARODA": "Banking", "PNB": "Banking", "CANBK": "Banking",
    "UNIONBANK": "Banking", "IDFCFIRSTB": "Banking", "FEDERALBNK": "Banking",
    "BANDHANBNK": "Banking", "RBLBANK": "Banking", "YESBANK": "Banking",
    "INDIANB": "Banking", "BANKINDIA": "Banking",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "PERSISTENT": "IT", "COFORGE": "IT", "MPHASIS": "IT",
    "KPITTECH": "IT", "TATAELXSI": "IT", "CYIENT": "IT", "TANLA": "IT",
    "ROUTE": "IT", "MASTEK": "IT", "BIRLASOFT": "IT",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "AUROPHARMA": "Pharma", "TORNTPHARM": "Pharma",
    "IPCALAB": "Pharma", "ALKEM": "Pharma", "SYNGENE": "Pharma",
    "BIOCON": "Pharma", "LUPIN": "Pharma", "ZYDUSLIFE": "Pharma",
    "MANKIND": "Pharma", "LAURUSLABS": "Pharma", "GRANULES": "Pharma",
    "LALPATHLAB": "Healthcare", "METROPOLIS": "Healthcare",
    "APOLLOHOSP": "Healthcare", "FORTIS": "Healthcare",
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "HEROMOTOCO": "Auto",
    "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto", "MOTHERSON": "Auto",
    "BOSCHLTD": "Auto", "BHARATFORG": "Auto", "EXIDEIND": "Auto",
    "MRF": "Auto", "APOLLOTYRE": "Auto", "CEATLTD": "Auto", "BALKRISIND": "Auto",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "TATACONSUM": "FMCG", "EMAMILTD": "FMCG", "GODREJCP": "FMCG",
    "COLPAL": "FMCG", "VBL": "FMCG",
    "TRENT": "Retail", "DMART": "Retail", "PAGEIND": "Retail",
    "RELAXO": "Retail", "BATA": "Retail", "NYKAA": "Retail",
    "BHARTIARTL": "Telecom",
    "LT": "Infra", "SIEMENS": "Infra", "ABB": "Infra", "BHEL": "Infra",
    "NTPC": "Power", "POWERGRID": "Power", "ADANIGREEN": "Power",
    "TATAPOWER": "Power", "TORNTPOWER": "Power", "CESC": "Power",
    "ULTRACEMCO": "Cement", "AMBUJACEM": "Cement", "ACC": "Cement",
    "SHREECEM": "Cement", "DALMIACEM": "Cement",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "NMDC": "Metals", "SAIL": "Metals", "COALINDIA": "Metals",
    "DLF": "RealEstate", "GODREJPROP": "RealEstate", "OBEROIRLTY": "RealEstate",
    "PRESTIGE": "RealEstate", "BRIGADE": "RealEstate",
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC",
    "MUTHOOTFIN": "NBFC", "SHRIRAMFIN": "NBFC", "RECLTD": "NBFC", "PFC": "NBFC",
    "HDFCLIFE": "Insurance", "SBILIFE": "Insurance", "LICI": "Insurance",
    "ICICIGI": "Insurance",
    "HDFCAMC": "AssetMgmt", "NIPPONLIFE": "AssetMgmt", "CDSL": "AssetMgmt",
    "ANGELONE": "AssetMgmt",
    "ASIANPAINT": "Paints", "BERGEPAINT": "Paints", "KANSAINER": "Paints",
    "HAVELLS": "ConsumerElec", "VOLTAS": "ConsumerElec", "CROMPTON": "ConsumerElec",
    "DIXON": "ConsumerElec", "AMBER": "ConsumerElec",
    "TITAN": "Gems", "KALYANKJIL": "Gems",
    "PIDILITIND": "Chemicals", "SRF": "Chemicals", "DEEPAKNTR": "Chemicals",
    "NAVINFLUOR": "Chemicals", "AARTI": "Chemicals",
    "ZOMATO": "ECommerce", "PAYTM": "ECommerce", "NAUKRI": "ECommerce",
    "ADANIPORTS": "Ports", "ADANIENT": "Conglomerate",
    "INDIGO": "Aviation", "BLUEDART": "Logistics", "IRCTC": "Logistics",
    "SUNTVNETWORK": "Media", "PVRINOX": "Media",
}

SECTOR_ENCODING = {s: i for i, s in enumerate(sorted(set(SECTOR_MAP.values())))}


# ---------------------------------------------------------------------------
# Pure pandas/numpy indicator helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series):
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    return macd_line, signal_line, macd_line - signal_line


def _bollinger(close: pd.Series, window: int = 20):
    mid = _sma(close, window)
    std = close.rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(3).mean()
    return k, d


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(
    df: pd.DataFrame,
    ticker: str = "",
    fo_df: pd.DataFrame = None,
    sector: str | None = None,
) -> pd.DataFrame:
    """
    Input:
        df     — OHLCV DataFrame with DatetimeIndex
        ticker — NSE ticker symbol (for sector lookup fallback)
        fo_df  — optional F&O DataFrame (total_oi, oi_change, put_oi, call_oi, pcr)
        sector — sector name from the stocks DB table; falls back to SECTOR_MAP,
                 then "Other", when not supplied. Keeps this function I/O-free.
    Output:
        Feature DataFrame aligned to same index (includes target columns for training)
    """
    df = df.copy().sort_index()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float).replace(0, np.nan)

    feat = pd.DataFrame(index=df.index)

    # RSI
    feat["rsi_14"] = _rsi(close, 14)
    feat["rsi_7"] = _rsi(close, 7)
    feat["rsi_21"] = _rsi(close, 21)
    feat["rsi_14_slope"] = feat["rsi_14"].diff(3)

    # MACD
    macd, macd_sig, macd_hist = _macd(close)
    feat["macd"] = macd
    feat["macd_signal"] = macd_sig
    feat["macd_diff"] = macd_hist
    feat["macd_cross"] = (macd > macd_sig).astype(int)

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = _bollinger(close, 20)
    feat["bb_upper"] = bb_upper
    feat["bb_lower"] = bb_lower
    feat["bb_mid"] = bb_mid
    feat["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
    feat["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)

    # Moving averages
    for w in [5, 10, 20, 50, 100, 200]:
        sma = _sma(close, w)
        feat[f"sma_{w}"] = sma
        feat[f"dist_sma_{w}"] = (close - sma) / (sma + 1e-8)

    feat["ema_9"] = _ema(close, 9)
    feat["ema_21"] = _ema(close, 21)
    feat["ema_cross_9_21"] = (feat["ema_9"] > feat["ema_21"]).astype(int)

    # ATR
    atr = _atr(high, low, close, 14)
    feat["atr_14"] = atr
    feat["atr_pct"] = atr / (close + 1e-8)

    # OBV
    obv = _obv(close, volume.fillna(0))
    feat["obv"] = obv
    feat["obv_slope"] = obv.diff(5) / (obv.abs() + 1e-8)

    # Volume
    vol_sma20 = _sma(volume, 20)
    feat["volume_ratio"] = volume / (vol_sma20 + 1e-8)
    feat["volume_spike"] = (feat["volume_ratio"] > 2.0).astype(int)

    # Returns
    for d in [1, 3, 5, 10, 20, 60]:
        feat[f"return_{d}d"] = close.pct_change(d)

    # Candle patterns
    feat["candle_body"] = (df["close"] - df["open"]) / (df["open"].astype(float) + 1e-8)
    candle_range = (high - low).replace(0, 1e-8)
    feat["candle_upper_shadow"] = (high - df[["close", "open"]].astype(float).max(axis=1)) / candle_range
    feat["candle_lower_shadow"] = (df[["close", "open"]].astype(float).min(axis=1) - low) / candle_range

    # 52-week
    feat["high_52w"] = high.rolling(252).max()
    feat["low_52w"] = low.rolling(252).min()
    feat["dist_52w_high"] = (close - feat["high_52w"]) / (feat["high_52w"] + 1e-8)
    feat["dist_52w_low"] = (close - feat["low_52w"]) / (feat["low_52w"] + 1e-8)

    # Stochastic
    stoch_k, stoch_d = _stochastic(high, low, close)
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_d

    # Calendar
    feat["day_of_week"] = df.index.dayofweek
    feat["month"] = df.index.month
    feat["quarter"] = df.index.quarter

    # Sector — prefer DB-provided value, fall back to static map, then "Other"
    sector_name = sector or SECTOR_MAP.get(ticker, "Other")
    feat["sector"] = SECTOR_ENCODING.get(sector_name, len(SECTOR_ENCODING))

    # F&O features
    if fo_df is not None and not fo_df.empty:
        fo_aligned = fo_df.reindex(df.index)
        total_oi = fo_aligned.get("total_oi", pd.Series(dtype=float))
        oi_change = fo_aligned.get("oi_change", pd.Series(dtype=float))
        pcr = fo_aligned.get("pcr", pd.Series(dtype=float))
        feat["oi_change_pct"] = oi_change / (total_oi.shift(1) + 1e-8)
        feat["pcr"] = pcr
        feat["oi_rising"] = (oi_change > 0).astype(int)
    else:
        feat["oi_change_pct"] = 0.0
        feat["pcr"] = 1.0
        feat["oi_rising"] = 0

    # Targets (drop before inference, used for training only)
    feat["target_1d_return"] = close.pct_change(1).shift(-1)
    feat["target_buy"] = (feat["target_1d_return"] > 0.015).astype(int)

    return feat


def get_feature_columns() -> list[str]:
    """Core feature columns (no F&O)."""
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
    """All features including F&O."""
    return get_feature_columns() + ["oi_change_pct", "pcr", "oi_rising"]
