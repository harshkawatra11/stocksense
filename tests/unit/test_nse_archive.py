"""NSE archive ingester tests. Network calls are mocked -- the fetchers
themselves were verified against the real live endpoints during
development (both format eras, delivery, F&O; see the session that
built this). What's under test here: era-switching on the exact
2024-07-08 boundary, canonical schema normalization, holiday-404
handling, and content-hash caching -- the properties that must hold
regardless of what NSE's servers happen to return on any given day."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stocksense.data.nse_archive import (
    UDIFF_CUTOVER,
    FetchError,
    _cached_or_fetch,
    fetch_cm_bhavcopy,
    fetch_delivery,
    fetch_fo_bhavcopy,
    fetch_range,
)


def _zip_csv(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.csv", df.to_csv(index=False))
    return buf.getvalue()


def _mock_response(status_code=200, content=b""):
    m = MagicMock()
    m.status_code = status_code
    m.content = content
    return m


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    import stocksense.data.nse_archive as nse_mod

    monkeypatch.setattr(nse_mod, "CACHE_DIR", tmp_path / "nse_archive")
    monkeypatch.setattr(nse_mod, "_POLITE_DELAY_S", 0.0)  # no need to actually sleep in tests
    return tmp_path


def test_udiff_cutover_is_the_documented_date() -> None:
    assert UDIFF_CUTOVER == date(2024, 7, 8)


@patch("stocksense.data.nse_archive.requests.get")
def test_fetch_cm_bhavcopy_uses_legacy_url_before_cutover(mock_get, cache_dir) -> None:
    legacy_df = pd.DataFrame({
        "SYMBOL": ["RELIANCE"], "SERIES": ["EQ"], "OPEN": [2500.0], "HIGH": [2510.0],
        "LOW": [2490.0], "CLOSE": [2505.0], "LAST": [2505.0], "PREVCLOSE": [2495.0],
        "TOTTRDQTY": [1000000], "TOTTRDVAL": [2500000000.0], "TIMESTAMP": ["04-JAN-2010"],
    })
    mock_get.return_value = _mock_response(200, _zip_csv(legacy_df))

    result = fetch_cm_bhavcopy(date(2010, 1, 4))
    assert result is not None
    assert result.iloc[0]["era"] == "legacy"
    assert result.iloc[0]["symbol"] == "RELIANCE"
    called_url = mock_get.call_args[0][0]
    assert "historical/EQUITIES" in called_url


@patch("stocksense.data.nse_archive.requests.get")
def test_fetch_cm_bhavcopy_uses_udiff_url_on_and_after_cutover(mock_get, cache_dir) -> None:
    udiff_df = pd.DataFrame({
        "TckrSymb": ["RELIANCE"], "SctySrs": ["EQ"], "OpnPric": [2500.0], "HghPric": [2510.0],
        "LwPric": [2490.0], "ClsPric": [2505.0], "PrvsClsgPric": [2495.0],
        "TtlTradgVol": [1000000], "TtlTrfVal": [2500000000.0],
    })
    mock_get.return_value = _mock_response(200, _zip_csv(udiff_df))

    result = fetch_cm_bhavcopy(UDIFF_CUTOVER)  # exactly on the boundary -> UDiFF, not legacy
    assert result is not None
    assert result.iloc[0]["era"] == "udiff"
    called_url = mock_get.call_args[0][0]
    assert "/content/cm/BhavCopy" in called_url


@patch("stocksense.data.nse_archive.requests.get")
def test_fetch_cm_bhavcopy_day_before_cutover_still_legacy(mock_get, cache_dir) -> None:
    legacy_df = pd.DataFrame({
        "SYMBOL": ["X"], "SERIES": ["EQ"], "OPEN": [1.0], "HIGH": [1.0], "LOW": [1.0],
        "CLOSE": [1.0], "LAST": [1.0], "PREVCLOSE": [1.0], "TOTTRDQTY": [1], "TOTTRDVAL": [1.0], "TIMESTAMP": ["x"],
    })
    mock_get.return_value = _mock_response(200, _zip_csv(legacy_df))

    result = fetch_cm_bhavcopy(date(2024, 7, 7))  # one day before cutover
    assert result.iloc[0]["era"] == "legacy"


@patch("stocksense.data.nse_archive.requests.get")
def test_both_eras_normalize_to_identical_canonical_columns(mock_get, cache_dir) -> None:
    legacy_df = pd.DataFrame({
        "SYMBOL": ["X"], "SERIES": ["EQ"], "OPEN": [1.0], "HIGH": [1.0], "LOW": [1.0],
        "CLOSE": [1.0], "LAST": [1.0], "PREVCLOSE": [1.0], "TOTTRDQTY": [1], "TOTTRDVAL": [1.0], "TIMESTAMP": ["x"],
    })
    mock_get.return_value = _mock_response(200, _zip_csv(legacy_df))
    legacy_result = fetch_cm_bhavcopy(date(2010, 1, 4))

    udiff_df = pd.DataFrame({
        "TckrSymb": ["X"], "SctySrs": ["EQ"], "OpnPric": [1.0], "HghPric": [1.0],
        "LwPric": [1.0], "ClsPric": [1.0], "PrvsClsgPric": [1.0], "TtlTradgVol": [1], "TtlTrfVal": [1.0],
    })
    mock_get.return_value = _mock_response(200, _zip_csv(udiff_df))
    udiff_result = fetch_cm_bhavcopy(date(2025, 1, 1))

    assert list(legacy_result.columns) == list(udiff_result.columns)


@patch("stocksense.data.nse_archive.requests.get")
def test_404_returns_none_not_an_error(mock_get, cache_dir) -> None:
    mock_get.return_value = _mock_response(404)
    result = fetch_cm_bhavcopy(date(2024, 1, 26))  # Republic Day, known holiday
    assert result is None


@patch("stocksense.data.nse_archive.requests.get")
def test_unexpected_status_raises_fetch_error(mock_get, cache_dir) -> None:
    mock_get.return_value = _mock_response(500)
    with pytest.raises(FetchError):
        fetch_cm_bhavcopy(date(2024, 1, 15))


@patch("stocksense.data.nse_archive.requests.get")
def test_content_hash_cache_avoids_second_network_call(mock_get, cache_dir) -> None:
    legacy_df = pd.DataFrame({
        "SYMBOL": ["X"], "SERIES": ["EQ"], "OPEN": [1.0], "HIGH": [1.0], "LOW": [1.0],
        "CLOSE": [1.0], "LAST": [1.0], "PREVCLOSE": [1.0], "TOTTRDQTY": [1], "TOTTRDVAL": [1.0], "TIMESTAMP": ["x"],
    })
    mock_get.return_value = _mock_response(200, _zip_csv(legacy_df))

    fetch_cm_bhavcopy(date(2010, 1, 4))
    assert mock_get.call_count == 1

    fetch_cm_bhavcopy(date(2010, 1, 4))  # same date again
    assert mock_get.call_count == 1  # not called a second time -- served from cache


@patch("stocksense.data.nse_archive.requests.get")
def test_fetch_delivery_extracts_deliv_qty_and_pct(mock_get, cache_dir) -> None:
    raw = pd.DataFrame({
        "SYMBOL": ["RELIANCE"], " SERIES": ["EQ"], "DELIV_QTY": [1730550], " DELIV_PER": [57.85],
    })
    buf = io.BytesIO()
    raw.to_csv(buf, index=False)
    mock_get.return_value = _mock_response(200, buf.getvalue())

    result = fetch_delivery(date(2023, 8, 7))
    assert result.iloc[0]["symbol"] == "RELIANCE"
    assert result.iloc[0]["delivery_pct"] == 57.85


@patch("stocksense.data.nse_archive.requests.get")
def test_fetch_fo_bhavcopy_legacy_extracts_open_interest(mock_get, cache_dir) -> None:
    raw = pd.DataFrame({
        "INSTRUMENT": ["FUTIDX"], "SYMBOL": ["BANKNIFTY"], "EXPIRY_DT": ["26-Feb-2015"],
        "STRIKE_PR": [0.0], "OPTION_TYP": ["XX"], "OPEN": [1.0], "HIGH": [1.0], "LOW": [1.0],
        "CLOSE": [1.0], "SETTLE_PR": [1.0], "CONTRACTS": [1], "VAL_INLAKH": [1.0],
        "OPEN_INT": [2215900], "CHG_IN_OI": [19275], "TIMESTAMP": ["16-Feb-2015"],
    })
    mock_get.return_value = _mock_response(200, _zip_csv(raw))

    result = fetch_fo_bhavcopy(date(2015, 2, 16))
    assert result.iloc[0]["open_interest"] == 2215900
    assert result.iloc[0]["era"] == "legacy"


@patch("stocksense.data.nse_archive.requests.get")
def test_fetch_range_skips_weekends_without_requesting(mock_get, cache_dir) -> None:
    mock_get.return_value = _mock_response(404)  # every weekday attempt is a "holiday"
    # 2024-01-06 (Sat) to 2024-01-07 (Sun) -- a pure weekend range
    results = list(fetch_range(date(2024, 1, 6), date(2024, 1, 7), kind="cm"))
    assert results == []
    mock_get.assert_not_called()


@patch("stocksense.data.nse_archive.requests.get")
def test_fetch_range_treats_network_error_as_none_not_a_crash(mock_get, cache_dir) -> None:
    import requests

    mock_get.side_effect = requests.exceptions.ConnectionError("connection reset")
    results = list(fetch_range(date(2024, 1, 15), date(2024, 1, 15), kind="cm"))
    assert len(results) == 1
    assert results[0] == (date(2024, 1, 15), None)


@patch("stocksense.data.nse_archive.requests.get")
def test_fetch_range_is_a_generator_yielding_incrementally(mock_get, cache_dir) -> None:
    """The load-bearing property behind resumability: fetch_range must
    yield each day as it's fetched, not accumulate everything and return
    once at the end -- a caller that writes to the database inside the
    loop can be killed mid-range and keep every day already yielded,
    which a list-returning version would not allow (a kill before the
    return statement loses everything, even days already fetched)."""
    import inspect

    assert inspect.isgeneratorfunction(fetch_range)

    mock_get.return_value = _mock_response(404)
    gen = fetch_range(date(2024, 1, 15), date(2024, 1, 19), kind="cm")
    first = next(gen)  # must be able to pull one result without exhausting the whole range
    assert first[0] == date(2024, 1, 15)
    assert mock_get.call_count == 1  # only the first day was actually fetched so far
