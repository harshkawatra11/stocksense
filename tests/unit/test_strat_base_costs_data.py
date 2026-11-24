from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.costs.india import trade_cost
from stocksense.costs.slippage import SlippageTable, slip_frac
from stocksense.data import io
from stocksense.strategies import base
from stocksense.strategies import data as D
from stocksense.strategies.costs import trade_cost_frac


def spec(key: str, name: str) -> base.StrategySpec:
    return base.StrategySpec(
        key=key,
        name=name,
        source="tier",
        tier="none",
        family="m-swing",
        timeframe="1d",
        kind="trend",
        direction="long",
        exits=("time5",),
        default_exit="time5",
        signal=None,
    )


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(base, "REGISTRY", {})
    monkeypatch.setattr(base, "load_library", lambda: base.REGISTRY)


def test_register_rejects_duplicate_keys_and_names(clean_registry):
    base.register(spec("a", "Alpha"))
    with pytest.raises(KeyError):
        base.register(spec("a", "Other"))
    with pytest.raises(KeyError):
        base.register(spec("b", "Alpha"))


def test_lookup_by_key_or_display_name(clean_registry):
    s = base.register(spec("a", "Alpha Beta"))
    assert base.by_name_or_key("a") is s and base.by_name_or_key("alpha beta") is s
    with pytest.raises(KeyError):
        base.by_name_or_key("nope")


def test_code_hash_is_ten_hex_and_stable():
    h = base.code_hash()
    assert len(h) == 10 and int(h, 16) >= 0 and base.code_hash() == h


def test_strengths():
    f = {
        "c": np.array([10.0, 12.0]),
        "hh20": np.array([10.0, 11.0]),
        "atr": np.array([1.0, 0.5]),
        "vwap": np.array([10.0, 11.0]),
        "rs63": np.array([5.0, np.nan]),
        "n": 2,
    }
    sig = np.array([False, True])
    assert base.STRENGTHS["signal"](f, sig).tolist() == [0.0, 1.0]
    assert base.STRENGTHS["breakout_atr"](f, sig).tolist() == [0.0, 2.0]
    assert base.STRENGTHS["dev_sigma"](f, sig).tolist() == [0.0, 2.0]
    assert base.STRENGTHS["rs63"](f, np.array([True, True])).tolist() == [5.0, 0.0]


def test_trade_cost_frac_is_statutory_plus_slippage_on_both_sides(tmp_data_root):
    slip = SlippageTable.load_latest(tmp_data_root)
    got = trade_cost_frac("delivery", 100_000.0, tmp_data_root, slip, 100, 0.02, 5e8, "open", "now")
    stat = (
        trade_cost("BUY", "delivery", 100_000.0, tmp_data_root.costs).total
        + trade_cost("SELL", "delivery", 100_000.0, tmp_data_root.costs).total
    ) / 100_000.0
    s_in = slip_frac(100, 0.02, 100_000.0, 5e8, "open", slip)
    s_out = slip_frac(100, 0.02, 100_000.0, 5e8, "now", slip)
    assert got == pytest.approx(stat + s_in + s_out, rel=1e-12) and 0.002 < got < 0.02


def test_open_and_close_fills_cost_more_than_a_stop_fill_and_unknown_rank_uses_200(tmp_data_root):
    slip = SlippageTable.load_latest(tmp_data_root)
    args = ("delivery", 100_000.0, tmp_data_root, slip, 100, 0.02, 5e8)
    assert trade_cost_frac(*args, "open", "close") > trade_cost_frac(*args, "open", "now")
    a = trade_cost_frac("delivery", 1e5, tmp_data_root, slip, None, 0.02, 5e8, "open", "now")
    assert a == pytest.approx(
        trade_cost_frac("delivery", 1e5, tmp_data_root, slip, 200, 0.02, 5e8, "open", "now")
    )


def test_intraday_costs_less_statutory_than_delivery(tmp_data_root):
    slip = SlippageTable.load_latest(tmp_data_root)
    a = trade_cost_frac("intraday", 250_000.0, tmp_data_root, slip, 100, 0.02, 5e8, "open", "now")
    b = trade_cost_frac("delivery", 250_000.0, tmp_data_root, slip, 100, 0.02, 5e8, "open", "now")
    assert a < b


def test_nifty_table_roc_and_reindex():
    d = np.datetime64("2026-01-01") + np.arange(300)
    c = 100.0 * 1.001 ** np.arange(300)
    t = D.NiftyTable.from_close(np.r_[d, d[-1]], np.r_[c, 999.0])  # duplicate last date keeps the last
    assert len(t.date) == 300 and t.c[-1] == 999.0
    assert t.roc63[100] == pytest.approx((c[100] / c[37] - 1) * 100)
    got = t.reindex(np.array([d[0], d[5], np.datetime64("2030-01-01")]))
    assert got["c"][0] == c[0] and np.isnan(got["c"][2]) and np.isnan(got["sma200"][1])


def _panel(rows):
    return pl.DataFrame(
        rows,
        schema=[
            "sid",
            "symbol",
            "date",
            "series",
            "volume",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "adj_volume",
        ],
        orient="row",
    )


def test_daily_frame_reads_completed_adjusted_bars(tmp_data_root):
    d = [date(2026, 11, 16), date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)]
    rows = [
        (1, "AAA", x, "EQ", 100 + i, 10.0 + i, 11.0 + i, 9.0 + i, 10.5 + i, 200.0 + i)
        for i, x in enumerate(d)
    ]
    rows[1] = (1, "AAA", d[1], "EQ", 0, 11.0, 12.0, 10.0, 11.5, 0.0)  # no volume: dropped
    rows.append((2, "AAA", d[0], "BE", 5, 1.0, 1.0, 1.0, 1.0, 5.0))
    io.upsert(tmp_data_root, "panel_daily", _panel(rows))
    f = D.daily_frame(tmp_data_root, "AAA", end=date(2026, 11, 18))
    assert f.date.tolist() == [np.datetime64("2026-11-16"), np.datetime64("2026-11-18")]
    assert (
        f.close.tolist() == [10.5, 12.5]
        and f.volume.tolist() == [200.0, 202.0]
        and f.low.tolist() == [9.0, 11.0]
    )
    assert D.daily_frame(tmp_data_root, "ZZZ") is None
    cal = TradingCalendar.from_dates([date(2026, 11, 18), date(2026, 11, 19)])
    now = FixedClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    assert D.completed_end(now, cal, tmp_data_root) == date(2026, 11, 18)  # the 19th has not closed yet


def test_mirrored_daily_frame_swaps_high_and_low():
    f = D.DailyFrame(
        np.array(["2026-01-01"], "datetime64[D]"),
        np.array([10.0]),
        np.array([12.0]),
        np.array([9.0]),
        np.array([11.0]),
        np.array([5.0]),
    )
    m = f.mirrored()
    assert (m.open[0], m.high[0], m.low[0], m.close[0], m.volume[0]) == (-10.0, -9.0, -12.0, -11.0, 5.0)
