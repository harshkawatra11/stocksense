"""
Intraday feature engine (Phase E2). Builds on the raw 1-minute bar spine
(data/upstox_intraday.py) resampled to a 5-minute research grain, per the
setups named in the recovered v1 spec (intelligence/skills/SKILL_intraday.md
at commit 06aed5819dd5443815264110441851a705a1511d): opening-range
breakout, VWAP reversion, volume-spike, and time-of-day context.

Hard invariant, same as features/engine.py: every feature at row
(symbol, ts) may use only information timestamped <= ts. The one new
invariant intraday adds on top: every feature is also SESSION-BOUNDED --
computed fresh each trading day, never reaching back across a session
boundary into a prior day's bars (an MIS position does not exist
overnight, so a feature that quietly carries yesterday's VWAP into
today's open would be measuring something that was never tradeable).

Volume-spike is deliberately measured against a trailing SAME-TIME-OF-DAY
baseline, not a flat session average -- intraday volume is strongly
U-shaped (heavy at 09:15 and 15:25, thin at midday), so a flat baseline
fires a false spike at every single open and close.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SESSION_OPEN = pd.Timestamp("09:15:00").time()
SESSION_CLOSE = pd.Timestamp("15:30:00").time()
OPENING_RANGE_MINUTES = 15

IntradayFeatureFrame = pd.DataFrame


def resample_to_bars(bars_1min: pd.DataFrame, interval: str = "5min") -> pd.DataFrame:
    """Aggregates 1-minute bars up to a coarser research grain, resampled
    SEPARATELY per (symbol, trading day) so a bar never blends volume or
    price action across a session boundary -- naive pandas resample on
    the raw multi-day series would otherwise happily produce a bucket
    spanning 15:27 of one day through 09:18 of the next.
    """
    df = bars_1min.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["_session_date"] = df["ts"].dt.date

    def _resample_one_session(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("ts").sort_index()
        out = g.resample(interval, label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        })
        return out.dropna(subset=["open"])

    # group_keys=True (the default) so the (symbol, _session_date) group
    # labels survive as index levels on the concatenated result -- with
    # group_keys=False they're silently dropped, and _resample_one_session's
    # own output only carries `ts` in its index, not which symbol/session
    # it came from.
    resampled = (
        df.groupby(["symbol", "_session_date"])
        .apply(_resample_one_session)
        .reset_index()
    )
    resampled["interval"] = interval
    return resampled[["symbol", "ts", "interval", "open", "high", "low", "close", "volume"]]


def resample_to_bars_sql(bars_1min: pd.DataFrame, interval: str = "5min") -> pd.DataFrame:
    """SQL equivalent of resample_to_bars (Phase E4 performance fix):
    the pandas groupby(...).apply(...) version measured ~48 minutes at
    full scale (244 symbols, ~280k session groups) -- DuckDB's
    time_bucket over the same data measured 3.4s for a comparable
    aggregation. Kept as a SEPARATE function rather than a silent
    rewrite of the tested pandas path: tests/unit/test_intraday_features.py
    proves both produce IDENTICAL frames on the existing fixtures before
    this is trusted as a drop-in replacement for full-scale runs.

    Only `interval` values expressible as a DuckDB INTERVAL literal are
    supported (e.g. '5min' -> '5 minutes', '15min' -> '15 minutes') --
    this covers every grain the research pipeline actually uses.
    """
    unit_map = {"min": "minutes", "T": "minutes", "h": "hours", "H": "hours"}
    digits = "".join(c for c in interval if c.isdigit())
    suffix = interval[len(digits):]
    if suffix not in unit_map:
        raise ValueError(f"unsupported interval suffix {suffix!r} in {interval!r} -- expected one of {list(unit_map)}")
    duckdb_interval = f"{digits} {unit_map[suffix]}"

    df = bars_1min.copy()
    df["ts"] = pd.to_datetime(df["ts"])

    con = duckdb.connect(":memory:")
    con.register("_bars_1min", df)
    resampled = con.execute(
        f"""
        SELECT
            symbol,
            time_bucket(INTERVAL '{duckdb_interval}', ts) AS ts,
            '{interval}' AS interval,
            FIRST(open ORDER BY ts) AS open,
            MAX(high) AS high,
            MIN(low) AS low,
            LAST(close ORDER BY ts) AS close,
            SUM(volume) AS volume
        FROM _bars_1min
        GROUP BY symbol, CAST(ts AS DATE), time_bucket(INTERVAL '{duckdb_interval}', ts)
        ORDER BY symbol, ts
        """
    ).fetchdf()
    con.close()
    return resampled[["symbol", "ts", "interval", "open", "high", "low", "close", "volume"]]


def _session_features(g: pd.DataFrame) -> pd.DataFrame:
    """All features for ONE (symbol, trading day) session, `g` sorted by
    ts ascending and already restricted to that single session."""
    out = pd.DataFrame(index=g.index)
    close = g["close"]
    volume = g["volume"]

    session_open_price = close.iloc[0]
    session_open_ts = g["ts"].iloc[0]

    # ---- time-of-day ----
    out["minutes_since_open"] = (g["ts"] - session_open_ts).dt.total_seconds() / 60.0

    # ---- opening range (first OPENING_RANGE_MINUTES only) ----
    in_or_window = out["minutes_since_open"] < OPENING_RANGE_MINUTES
    or_high = g.loc[in_or_window, "high"].max() if in_or_window.any() else np.nan
    or_low = g.loc[in_or_window, "low"].min() if in_or_window.any() else np.nan
    # only defined for bars AFTER the opening range window has closed --
    # a bar inside the window cannot break a range that hasn't formed yet
    out["or_high"] = np.where(in_or_window, np.nan, or_high)
    out["or_low"] = np.where(in_or_window, np.nan, or_low)
    out["or_breakout_up"] = (~in_or_window) & (close > or_high)
    out["or_breakout_down"] = (~in_or_window) & (close < or_low)

    # ---- session VWAP (cumulative -- backward-looking by construction) ----
    typical_price = (g["high"] + g["low"] + close) / 3.0
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_vol
    out["vwap"] = vwap
    out["dist_from_vwap"] = (close / vwap) - 1.0

    # ---- session RSI (reset each session, not carried across days) ----
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ---- session return so far ----
    out["session_ret_so_far"] = (close / session_open_price) - 1.0

    return out


def build_intraday_features(bars: pd.DataFrame) -> IntradayFeatureFrame:
    """Build session-bounded intraday features from a resampled bar
    frame (symbol, ts, interval, open, high, low, close, volume).

    Returns one row per (symbol, ts) with all feature columns plus join
    keys, sorted by (symbol, ts). Volume-spike (cross-session, so it
    lives outside _session_features) is added afterward.
    """
    df = bars.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"])
    df["_session_date"] = df["ts"].dt.date

    per_session = df.groupby(["symbol", "_session_date"], group_keys=False).apply(_session_features)
    feats = pd.concat([df[["symbol", "ts"]], per_session], axis=1)

    # ---- volume spike vs trailing SAME-TIME-OF-DAY baseline ----
    # grouped by (symbol, clock time) across sessions, shift(1) excludes
    # the current session's own bar so the baseline never includes the
    # value it's being compared against (and can't leak same-day
    # information into itself). .transform() returns a Series aligned to
    # df's original index labels regardless of the sort used to compute
    # it, so this assigns back onto df correctly by label, not position.
    df["_time_of_day"] = df["ts"].dt.time
    df["_vol_baseline"] = (
        df.sort_values(["symbol", "_time_of_day", "_session_date"])
        .groupby(["symbol", "_time_of_day"])["volume"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )

    feats = feats.merge(df[["symbol", "ts", "volume", "_vol_baseline"]], on=["symbol", "ts"], how="left")
    feats["volume_spike_ratio"] = feats["volume"] / feats["_vol_baseline"].replace(0, np.nan)
    feats = feats.drop(columns=["volume", "_vol_baseline"])

    return feats.sort_values(["symbol", "ts"]).reset_index(drop=True)


def build_intraday_features_cached(
    bars: pd.DataFrame, cache_key: str, parquet_dir: Path | None = None, force_rebuild: bool = False,
) -> IntradayFeatureFrame:
    """Caches build_intraday_features' output to parquet (Phase E4
    performance fix): the feature build itself measured ~78 minutes at
    full scale (244 symbols, ~280k session groups through pandas
    groupby.apply -- see build_intraday_features' docstring) and,
    unlike resample_to_bars, is NOT rewritten to SQL here -- VWAP/RSI/
    opening-range logic already has 11 passing tests, and a one-time
    cost doesn't justify re-deriving tested session-feature logic in
    SQL. Instead: build once, cache to parquet, every subsequent E4
    sweep run against the SAME inputs reads the cache in seconds.

    `cache_key` is caller-supplied (e.g. derived from the date range and
    symbol count actually used), not auto-derived from hashing `bars`
    itself -- hashing 18M+ rows would cost nearly as much as the
    resample it's trying to avoid paying for twice.
    """
    from stocksense.core.config import get_settings

    directory = parquet_dir or get_settings().parquet_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"intraday_features_{cache_key}.parquet"

    if path.exists() and not force_rebuild:
        return pd.read_parquet(path)

    feats = build_intraday_features(bars)
    feats.to_parquet(path, index=False)
    return feats


def feature_columns(feats: IntradayFeatureFrame) -> list[str]:
    """Names of the actual feature columns (excludes join keys)."""
    exclude = {"symbol", "ts"}
    return [c for c in feats.columns if c not in exclude]


def attach_prior_day_context(
    feats: IntradayFeatureFrame, daily_bhavcopy: pd.DataFrame,
) -> IntradayFeatureFrame:
    """Joins prior-day gap/range context from the daily bhavcopy spine
    onto each intraday row -- the point where the daily and intraday
    spines meet. `daily_bhavcopy` must carry (symbol, date, open, high,
    low, close, prev_close); only PRIOR trading days relative to a bar's
    own session are ever used, joined on the immediately preceding daily
    row so today's own daily bar (which hasn't closed yet, intraday)
    never leaks in.
    """
    daily = daily_bhavcopy.sort_values(["symbol", "date"]).copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["prior_range_pct"] = (daily["high"] - daily["low"]) / daily["prev_close"]
    daily["gap_pct"] = (daily["open"] / daily["prev_close"]) - 1.0
    # shift so each row carries what a bar on THIS date may legitimately
    # see: the prior day's own realized range, not today's (which is
    # still in progress intraday and would be a leakage vector).
    daily["prev_day_range_pct"] = daily.groupby("symbol")["prior_range_pct"].shift(1)

    out = feats.copy()
    out["_session_date"] = pd.to_datetime(out["ts"]).dt.normalize()
    merged = out.merge(
        daily[["symbol", "date", "gap_pct", "prev_day_range_pct"]].rename(columns={"date": "_session_date"}),
        on=["symbol", "_session_date"], how="left",
    )
    return merged.drop(columns=["_session_date"])
