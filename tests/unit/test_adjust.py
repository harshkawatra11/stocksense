"""Price adjustment tests. The whole point of data/adjust.py is that a
confirmed split (verified live: ECLERX 1:2, PASHUPATI 1:10 -- bhavcopy's
own prev_close equals the prior RAW close across both, factor 1.0, no
adjustment) must produce a CONTINUOUS adjusted series -- no residual
jump on the ex-date -- while an unrelated symbol with no corporate
action is left untouched."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stocksense.data.adjust import adjusted_prices, adjustment_factors
from stocksense.data.store import Store


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _bhav_row(symbol, d, close, prev_close=None):
    return {
        "symbol": symbol, "series": "EQ", "date": d, "open": close, "high": close,
        "low": close, "close": close, "prev_close": prev_close if prev_close is not None else close,
        "volume": 1000.0, "turnover_inr": close * 1000.0, "era": "udiff",
    }


def _ca_row(symbol, ex_date, action_type, factor_price, dividend_amount=None):
    return {
        "symbol": symbol, "ex_date": ex_date, "action_type": action_type,
        "ratio_num": None, "ratio_den": None, "factor_price": factor_price,
        "dividend_amount": dividend_amount, "face_before": None, "face_after": None,
        "subject_raw": f"test {action_type}", "parse_status": "ok",
    }


def test_split_produces_continuous_adjusted_series_no_jump_on_ex_date(tmp_store) -> None:
    # A 1:1 split (ECLERX-style): raw close halves on the ex-date, but
    # NO real change in shareholder value occurred.
    rows = [
        _bhav_row("SPLITCO", date(2026, 3, 10), close=3151.8),  # day before ex-date
        _bhav_row("SPLITCO", date(2026, 3, 13), close=1576.6, prev_close=3151.8),  # ex-date, raw halved
        _bhav_row("SPLITCO", date(2026, 3, 16), close=1580.0),  # continues trading post-split
    ]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    tmp_store.write_corporate_actions(pd.DataFrame([_ca_row("SPLITCO", date(2026, 3, 13), "split", 0.5)]))

    out = adjusted_prices(tmp_store, ["SPLITCO"], date(2026, 3, 1), date(2026, 3, 31), basis="price")
    out = out.sort_values("date").reset_index(drop=True)

    pre_split_adj = out.iloc[0]["adj_close"]   # 2026-03-10, before ex-date -> gets multiplied by 0.5
    post_split_adj = out.iloc[1]["adj_close"]  # 2026-03-13, ex-date itself -> already at new scale, factor 1.0

    assert pre_split_adj == pytest.approx(3151.8 * 0.5)
    assert post_split_adj == pytest.approx(1576.6)
    # continuity: no residual one-day "return" from the split itself
    day_over_day_ret = abs(post_split_adj / pre_split_adj - 1)
    assert day_over_day_ret < 0.02


def test_symbol_with_no_corporate_action_is_unadjusted(tmp_store) -> None:
    rows = [_bhav_row("PLAINCO", date(2026, 1, 5), close=100.0), _bhav_row("PLAINCO", date(2026, 1, 6), close=101.0)]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    out = adjusted_prices(tmp_store, ["PLAINCO"], date(2026, 1, 1), date(2026, 1, 31), basis="price")
    assert (out["adj_close"] == out["close"]).all()


def test_adjustment_factors_cumulative_across_multiple_events(tmp_store) -> None:
    # Two splits on the same symbol: a price before BOTH ex-dates must
    # be multiplied by the product of both factors.
    rows = [
        _bhav_row("DOUBLE", date(2020, 1, 1), close=1000.0),
        _bhav_row("DOUBLE", date(2021, 1, 1), close=500.0, prev_close=1000.0),
        _bhav_row("DOUBLE", date(2022, 1, 1), close=250.0, prev_close=500.0),
    ]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    tmp_store.write_corporate_actions(pd.DataFrame([
        _ca_row("DOUBLE", date(2021, 1, 1), "split", 0.5),
        _ca_row("DOUBLE", date(2022, 1, 1), "split", 0.5),
    ]))

    out = adjusted_prices(tmp_store, ["DOUBLE"], date(2019, 1, 1), date(2022, 12, 31), basis="price").sort_values("date")
    first = out.iloc[0]["adj_close"]  # 2020-01-01: before both ex-dates
    assert first == pytest.approx(1000.0 * 0.5 * 0.5)


def test_dividend_only_affects_total_basis_not_price_basis(tmp_store) -> None:
    rows = [
        _bhav_row("DIVCO", date(2026, 1, 5), close=100.0),
        _bhav_row("DIVCO", date(2026, 1, 6), close=98.0, prev_close=100.0),  # ex-dividend drop
    ]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    tmp_store.write_corporate_actions(pd.DataFrame([_ca_row("DIVCO", date(2026, 1, 6), "dividend", 1.0, dividend_amount=2.0)]))

    price_basis = adjusted_prices(tmp_store, ["DIVCO"], date(2026, 1, 1), date(2026, 1, 31), basis="price")
    total_basis = adjusted_prices(tmp_store, ["DIVCO"], date(2026, 1, 1), date(2026, 1, 31), basis="total")

    assert (price_basis["adj_close"] == price_basis["close"]).all()
    pre_div_total = total_basis.sort_values("date").iloc[0]["adj_close"]
    assert pre_div_total < 100.0  # total-return basis backs out the dividend from the pre-ex-date price


def test_adjustment_factors_empty_when_no_actions(tmp_store) -> None:
    factors = adjustment_factors(tmp_store, ["ANYTHING"], basis="price")
    assert factors.empty


def test_adjustment_factors_rejects_invalid_basis(tmp_store) -> None:
    with pytest.raises(ValueError):
        adjustment_factors(tmp_store, ["X"], basis="bogus")
