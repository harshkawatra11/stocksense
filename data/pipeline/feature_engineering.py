"""
Feature engineering for LightGBM. Takes OHLCV DataFrame, returns feature DataFrame.
All features computed in-place, no lookahead.
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


SECTOR_MAP = {
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "IOC": "Energy",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", "KOTAKBANK": "Banking",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "HEROMOTOCO": "Auto", "BAJAJ-AUTO": "Auto",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "LT": "Infra", "NTPC": "Power", "POWERGRID": "Power", "ADANIGREEN": "Power",
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC",
}

SECTOR_ENCODING = {s: i for i, s in enumerate(sorted(set(SECTOR_MAP.values())))}


def compute_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Input: df with columns [open, high, low, close, volume], DatetimeIndex
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

    # Sector
    sector = SECTOR_MAP.get(ticker, "Other")
    feat["sector"] = SECTOR_ENCODING.get(sector, len(SECTOR_ENCODING))

    # Target: next-day return (for training only, drop before inference)
    feat["target_1d_return"] = close.pct_change(1).shift(-1)
    feat["target_buy"] = (feat["target_1d_return"] > 0.015).astype(int)

    return feat


def get_feature_columns() -> list[str]:
    """All feature column names (exclude targets)."""
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
