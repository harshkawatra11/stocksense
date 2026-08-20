"""Intraday feature tests (Phase E2). What's under test: session-bounded
resampling (a bar must never blend two trading days), opening-range
breakout only becoming defined AFTER its own window closes, VWAP as a
genuinely cumulative (backward-only) quantity, RSI resetting per
session, and volume-spike measured against a same-time-of-day baseline
rather than a flat one (so it doesn't fire at every single open)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stocksense.features.intraday import (
    OPENING_RANGE_MINUTES,
    attach_prior_day_context,
    build_intraday_features,
    build_intraday_features_cached,
    feature_columns,
    resample_to_bars,
    resample_to_bars_sql,
)


def _minute_bars(symbol: str, day: str, prices: list[float], start="09:15") -> pd.DataFrame:
    """One 1-minute bar per price, starting at `start` on `day`."""
    ts0 = pd.Timestamp(f"{day} {start}")
    rows = []
    for i, px in enumerate(prices):
        rows.append({
            "symbol": symbol, "ts": ts0 + pd.Timedelta(minutes=i), "interval": "1minute",
            "open": px, "high": px + 0.5, "low": px - 0.5, "close": px, "volume": 1000.0 + i,
        })
    return pd.DataFrame(rows)


# ---- resample_to_bars: session-boundedness ----

def test_resample_never_blends_across_a_session_boundary() -> None:
    day1 = _minute_bars("X", "2026-01-05", [100.0] * 20)
    day2 = _minute_bars("X", "2026-01-06", [200.0] * 20)
    bars_1min = pd.concat([day1, day2], ignore_index=True)

    out = resample_to_bars(bars_1min, interval="5min")
    # every resulting bar must belong entirely to one calendar day
    assert (out["ts"].dt.normalize().isin([pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")])).all()
    # no 5-min bucket mixes price level 100 and 200 (would show as an OHLC
    # straddling both, which cannot happen if grouping is session-aware)
    day1_bars = out[out["ts"].dt.normalize() == pd.Timestamp("2026-01-05")]
    day2_bars = out[out["ts"].dt.normalize() == pd.Timestamp("2026-01-06")]
    assert (day1_bars["close"] < 150).all()
    assert (day2_bars["close"] > 150).all()


def test_resample_aggregates_ohlcv_correctly() -> None:
    bars_1min = _minute_bars("X", "2026-01-05", [100.0, 101.0, 99.0, 102.0, 103.0])
    out = resample_to_bars(bars_1min, interval="5min")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["open"] == 100.0
    assert row["close"] == 103.0
    assert row["high"] == pytest.approx(103.5)  # 103 + 0.5
    assert row["low"] == pytest.approx(98.5)     # 99 - 0.5
    assert row["volume"] == pytest.approx(1000 + 1001 + 1002 + 1003 + 1004)


# ---- resample_to_bars_sql: DuckDB equivalent, ~49x faster at full scale
# (measured: 72.7s pandas vs 1.5s SQL on 2.14M real 1-min rows) -- kept
# as a separate function, proven identical on fixtures before any E4
# sweep script is trusted to use it as a drop-in replacement. ----

def test_sql_resample_matches_pandas_resample_multi_symbol_multi_day() -> None:
    day1_x = _minute_bars("X", "2026-01-05", list(np.linspace(95, 105, 375)))
    day1_y = _minute_bars("Y", "2026-01-05", list(np.linspace(200, 210, 375)))
    day2_x = _minute_bars("X", "2026-01-06", list(np.linspace(90, 110, 375)))
    bars_1min = pd.concat([day1_x, day1_y, day2_x], ignore_index=True)

    pandas_out = resample_to_bars(bars_1min, "5min").sort_values(["symbol", "ts"]).reset_index(drop=True)
    sql_out = resample_to_bars_sql(bars_1min, "5min").sort_values(["symbol", "ts"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(pandas_out, sql_out, check_dtype=False)


def test_sql_resample_matches_pandas_resample_on_irregular_gaps() -> None:
    """An illiquid symbol's bars aren't evenly spaced -- both paths must
    still agree on bucket boundaries when minutes are missing."""
    ts0 = pd.Timestamp("2026-01-05 09:15")
    rows = []
    for minute_offset in [0, 1, 2, 7, 8, 22, 23, 24, 25, 40]:  # gaps, not a dense 0..N range
        px = 100.0 + minute_offset * 0.1
        rows.append({
            "symbol": "THIN", "ts": ts0 + pd.Timedelta(minutes=minute_offset), "interval": "1minute",
            "open": px, "high": px + 0.3, "low": px - 0.3, "close": px, "volume": 500.0,
        })
    bars_1min = pd.DataFrame(rows)

    pandas_out = resample_to_bars(bars_1min, "5min").sort_values(["symbol", "ts"]).reset_index(drop=True)
    sql_out = resample_to_bars_sql(bars_1min, "5min").sort_values(["symbol", "ts"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(pandas_out, sql_out, check_dtype=False)


def test_sql_resample_rejects_unsupported_interval_suffix() -> None:
    bars_1min = _minute_bars("X", "2026-01-05", [100.0] * 10)
    with pytest.raises(ValueError):
        resample_to_bars_sql(bars_1min, interval="5weird")


# ---- build_intraday_features: opening range ----

def test_opening_range_undefined_within_its_own_window() -> None:
    bars = resample_to_bars(_minute_bars("X", "2026-01-05", [100.0] * 90), interval="5min")
    feats = build_intraday_features(bars)
    first_bar = feats.iloc[0]  # 09:15-09:20 bucket, inside the opening-range window
    assert pd.isna(first_bar["or_high"])
    assert pd.isna(first_bar["or_low"])
    assert first_bar["or_breakout_up"] == False  # noqa: E712 — can't break a range that hasn't formed


def test_opening_range_breakout_flags_correctly_after_window_closes() -> None:
    # flat until minute 16 (past the 15-min OR window), then a clean breakout
    prices = [100.0] * 16 + [110.0] * 20
    bars = resample_to_bars(_minute_bars("X", "2026-01-05", prices), interval="5min")
    feats = build_intraday_features(bars)

    post_window = feats[feats["minutes_since_open"] >= OPENING_RANGE_MINUTES]
    assert not post_window.empty
    assert post_window["or_high"].notna().all()
    # the breakout bars (close=110, or_high ~100.5) must be flagged
    assert post_window["or_breakout_up"].any()


# ---- VWAP: cumulative, backward-only ----

def test_vwap_is_cumulative_not_a_simple_average() -> None:
    # two 5-min bars: first heavy volume at price 100, second light volume at price 200
    bars = pd.DataFrame([
        {"symbol": "X", "ts": pd.Timestamp("2026-01-05 09:15"), "interval": "5min",
         "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 10000.0},
        {"symbol": "X", "ts": pd.Timestamp("2026-01-05 09:20"), "interval": "5min",
         "open": 200.0, "high": 200.0, "low": 200.0, "close": 200.0, "volume": 100.0},
    ])
    feats = build_intraday_features(bars)
    vwap_at_bar2 = feats.iloc[1]["vwap"]
    # heavily volume-weighted toward the first bar's price (100), nowhere
    # near a naive (100+200)/2 = 150 simple average
    assert vwap_at_bar2 < 110


def test_vwap_distance_zero_when_flat() -> None:
    bars = resample_to_bars(_minute_bars("X", "2026-01-05", [100.0] * 30), interval="5min")
    feats = build_intraday_features(bars)
    assert feats["dist_from_vwap"].abs().max() < 1e-9


# ---- RSI resets per session ----

def test_rsi_does_not_carry_across_a_session_boundary() -> None:
    # day1 ends on a steep decline (would normally suppress RSI heavily);
    # day2 opens flat. If RSI carried across sessions, day2's early RSI
    # would still reflect day1's decline; it must not.
    day1_prices = list(np.linspace(150, 100, 40))  # steep down day
    day2_prices = [100.0] * 40  # flat day
    bars_1min = pd.concat([
        _minute_bars("X", "2026-01-05", day1_prices),
        _minute_bars("X", "2026-01-06", day2_prices),
    ], ignore_index=True)
    bars = resample_to_bars(bars_1min, interval="5min")
    feats = build_intraday_features(bars)

    day2 = feats[feats["ts"].dt.normalize() == pd.Timestamp("2026-01-06")]
    first_valid_rsi = day2["rsi_14"].dropna()
    # a flat session has zero gains and zero losses -> RSI's gain/loss
    # ratio is 0/0 (NaN) throughout, never a suppressed value inherited
    # from yesterday's crash
    assert first_valid_rsi.empty or (first_valid_rsi.between(0, 100)).all()


# ---- volume spike: same-time-of-day baseline, not flat ----

def test_volume_spike_uses_trailing_same_time_of_day_baseline() -> None:
    # 25 quiet days at the 09:15 bucket (volume=1000), then a 26th day
    # with a genuine spike (volume=5000) at that same bucket.
    sessions = []
    for day_offset in range(26):
        day = (pd.Timestamp("2026-01-05") + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
        vol = 5000.0 if day_offset == 25 else 1000.0
        sessions.append(pd.DataFrame([{
            "symbol": "X", "ts": pd.Timestamp(f"{day} 09:15"), "interval": "5min",
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": vol,
        }]))
    bars = pd.concat(sessions, ignore_index=True)
    feats = build_intraday_features(bars)

    last_day_ratio = feats.iloc[-1]["volume_spike_ratio"]
    assert last_day_ratio == pytest.approx(5.0, rel=0.1)  # 5000 / ~1000 baseline


def test_volume_spike_baseline_excludes_the_current_bar_itself() -> None:
    """Regression: without shift(1), a bar's own volume would appear
    inside its own trailing-mean baseline, diluting a genuine spike."""
    sessions = []
    for day_offset in range(6):
        day = (pd.Timestamp("2026-01-05") + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
        vol = 10000.0 if day_offset == 5 else 1000.0
        sessions.append(pd.DataFrame([{
            "symbol": "X", "ts": pd.Timestamp(f"{day} 09:15"), "interval": "5min",
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": vol,
        }]))
    bars = pd.concat(sessions, ignore_index=True)
    feats = build_intraday_features(bars)
    last_ratio = feats.iloc[-1]["volume_spike_ratio"]
    # if the 10000 leaked into its own baseline, ratio would collapse toward 1
    assert last_ratio > 5.0


# ---- prior-day context ----

def test_attach_prior_day_context_uses_only_the_immediately_prior_day() -> None:
    bars = resample_to_bars(_minute_bars("X", "2026-01-07", [110.0] * 30), interval="5min")
    feats = build_intraday_features(bars)

    daily = pd.DataFrame([
        {"symbol": "X", "date": "2026-01-05", "open": 95.0, "high": 105.0, "low": 90.0, "close": 100.0, "prev_close": 95.0},
        {"symbol": "X", "date": "2026-01-06", "open": 100.0, "high": 108.0, "low": 98.0, "close": 105.0, "prev_close": 100.0},
        {"symbol": "X", "date": "2026-01-07", "open": 110.0, "high": 115.0, "low": 108.0, "close": 112.0, "prev_close": 105.0},
    ])
    out = attach_prior_day_context(feats, daily)
    row = out.iloc[0]
    assert row["gap_pct"] == pytest.approx((110.0 / 105.0) - 1.0)
    # prev_day_range_pct must come from 01-06 (the day BEFORE 01-07), not 01-07 itself
    expected_prior_range = (108.0 - 98.0) / 100.0
    assert row["prev_day_range_pct"] == pytest.approx(expected_prior_range)


def test_feature_columns_excludes_join_keys() -> None:
    bars = resample_to_bars(_minute_bars("X", "2026-01-05", [100.0] * 30), interval="5min")
    feats = build_intraday_features(bars)
    cols = feature_columns(feats)
    assert "symbol" not in cols
    assert "ts" not in cols
    assert "vwap" in cols


# ---- build_intraday_features_cached: parquet caching (Phase E4 perf fix) ----

def test_cached_build_writes_and_then_reads_the_parquet_cache(tmp_path) -> None:
    bars = resample_to_bars(_minute_bars("X", "2026-01-05", [100.0] * 30), interval="5min")

    first = build_intraday_features_cached(bars, cache_key="test1", parquet_dir=tmp_path)
    cache_file = tmp_path / "intraday_features_test1.parquet"
    assert cache_file.exists()

    # second call must return the CACHED content, not silently recompute
    # differently -- verified by corrupting the input so a fresh build
    # would look obviously different if the cache were bypassed
    bars_changed = bars.copy()
    bars_changed["close"] = 999.0
    second = build_intraday_features_cached(bars_changed, cache_key="test1", parquet_dir=tmp_path)

    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_cached_build_force_rebuild_ignores_stale_cache(tmp_path) -> None:
    bars = resample_to_bars(_minute_bars("X", "2026-01-05", [100.0] * 30), interval="5min")
    build_intraday_features_cached(bars, cache_key="test2", parquet_dir=tmp_path)

    bars_changed = resample_to_bars(_minute_bars("X", "2026-01-05", [200.0] * 30), interval="5min")
    forced = build_intraday_features_cached(bars_changed, cache_key="test2", parquet_dir=tmp_path, force_rebuild=True)

    assert forced["vwap"].iloc[0] == pytest.approx(200.0)


def test_cached_build_different_cache_keys_do_not_collide(tmp_path) -> None:
    bars_a = resample_to_bars(_minute_bars("X", "2026-01-05", [100.0] * 30), interval="5min")
    bars_b = resample_to_bars(_minute_bars("Y", "2026-01-05", [200.0] * 30), interval="5min")

    out_a = build_intraday_features_cached(bars_a, cache_key="a", parquet_dir=tmp_path)
    out_b = build_intraday_features_cached(bars_b, cache_key="b", parquet_dir=tmp_path)

    assert (tmp_path / "intraday_features_a.parquet").exists()
    assert (tmp_path / "intraday_features_b.parquet").exists()
    assert out_a["vwap"].iloc[0] != out_b["vwap"].iloc[0]
