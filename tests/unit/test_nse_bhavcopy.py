"""Bhavcopy ingestion tests.

Network is mocked throughout -- the real endpoints were verified live during
design (UDiFF 2026-08-28 -> 3,612 rows; legacy 2020-01-01 -> 1,910 rows with
SERIES == "NA" present; holidays 404). What these tests pin down is the
behaviour that must hold regardless of what NSE returns on any given day, and
in particular the three traps that silently corrupt data rather than raising.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from stocksense.data import nse_bhavcopy as nb
from stocksense.data.store import Reader, Store

# --------------------------------------------------------------------- fixtures
# Two rows each, and CRUCIALLY one with SERIES == "NA" -- a real NSE series code
# for certain bond instruments, which pandas' default NA handling turns into NaN.

_LEGACY_CSV = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
    "RELIANCE,EQ,1500.00,1520.00,1495.00,1510.00,1510.00,1498.00,1000000,1510000000.00,01-JAN-2020,50000,INE002A01018,\n"
    "SOMEBOND,NA,101.00,101.50,100.50,101.20,101.20,101.00,500,50600.00,01-JAN-2020,5,INE123A01011,\n"
)

_UDIFF_CSV = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
    "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,"
    "LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,"
    "TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
    "2026-08-28,2026-08-28,CM,NSE,STK,1,INE002A01018,RELIANCE,EQ,,,,,RELIANCE,"
    "1400.00,1425.00,1395.00,1420.00,1420.00,1398.00,,,,,2000000,2840000000.00,60000,F1,1,,,,,\n"
    "2026-08-28,2026-08-28,CM,NSE,STK,2,INE123A01011,SOMEBOND,NA,,,,,SOMEBOND,"
    "101.00,101.50,100.50,101.20,101.20,101.00,,,,,500,50600.00,5,F1,1,,,,,\n"
)

_DELIVERY_CSV = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE,"
    " CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    "RELIANCE, EQ, 28-Aug-2026, 1398.00, 1400.00, 1425.00, 1395.00, 1420.00,"
    " 1420.00, 1412.00, 2000000, 28400.00, 60000, 900000, 45.00\n"
)


def _zip_bytes(name: str, body: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, body)
    return buf.getvalue()


@pytest.fixture()
def fake_net(monkeypatch):
    """Serve canned bytes per (kind, date); record what was requested."""
    calls: list[tuple[str, date]] = []
    served: dict[str, bytes | None] = {
        "cm_legacy": _zip_bytes("cm01JAN2020bhav.csv", _LEGACY_CSV),
        "cm_udiff": _zip_bytes("BhavCopy.csv", _UDIFF_CSV),
        "delivery": _DELIVERY_CSV.encode(),
    }

    def fake(cache_root, kind, d, url, ext):
        calls.append((kind, d))
        return served.get(kind)

    monkeypatch.setattr(nb, "_cached_or_fetch", fake)
    return calls, served


# ------------------------------------------------------------------ trap #1
def test_NA_is_preserved_as_a_real_series_code_legacy(fake_net, tmp_path):
    """THE trap. "NA" is a genuine NSE series code for certain bond instruments.

    pandas' default NA list contains the string "NA", so a plain read_csv turns
    those rows' series into NaN -- which in a previous build violated a NOT NULL
    constraint and killed an 11-year backfill on its first day. Verified against
    the real 2020-01-01 file during design: SERIES == "NA" is present in it.
    """
    df = nb.fetch_day(tmp_path, date(2020, 1, 1))
    assert set(df.series) == {"EQ", "NA"}
    assert df.series.isna().sum() == 0
    assert (df.series == "NA").sum() == 1


def test_NA_is_preserved_as_a_real_series_code_udiff(fake_net, tmp_path):
    df = nb.fetch_day(tmp_path, date(2026, 8, 28))
    assert set(df.series) == {"EQ", "NA"}
    assert df.series.isna().sum() == 0


# ------------------------------------------------------------------ trap #2
def test_both_eras_normalise_to_one_identical_schema(fake_net, tmp_path):
    """Two formats sharing no column names must land on one schema, with `era`
    recording the source so a format-specific bug stays traceable."""
    legacy = nb.fetch_day(tmp_path, date(2020, 1, 1))
    udiff = nb.fetch_day(tmp_path, date(2026, 8, 28))

    assert list(legacy.columns) == nb.CANON_COLS
    assert list(udiff.columns) == nb.CANON_COLS
    assert set(legacy.era) == {"legacy"}
    assert set(udiff.era) == {"udiff"}

    lr = legacy[legacy.symbol == "RELIANCE"].iloc[0]
    ur = udiff[udiff.symbol == "RELIANCE"].iloc[0]
    assert (lr.open, lr.close, lr.prev_close) == (1500.0, 1510.0, 1498.0)
    assert (ur.open, ur.close, ur.prev_close) == (1400.0, 1420.0, 1398.0)
    assert lr.volume == 1_000_000 and ur.volume == 2_000_000


def test_era_boundary_picks_the_right_parser(fake_net, tmp_path):
    """NSE switched to UDiFF on 2024-07-08. One day either side must route to a
    different URL family -- an off-by-one here silently 404s a whole era."""
    calls, _ = fake_net
    nb.fetch_day(tmp_path, nb.UDIFF_FROM)
    nb.fetch_day(tmp_path, date(2024, 7, 5))
    kinds = [k for k, _ in calls]
    assert kinds == ["cm_udiff", "cm_legacy"]


# ------------------------------------------------------------------ trap #3
def test_a_holiday_returns_none_rather_than_raising(monkeypatch, tmp_path):
    """NSE says "market holiday" with a 404. That is normal, not an error, and
    it must not abort a multi-year backfill."""
    monkeypatch.setattr(nb, "_cached_or_fetch", lambda *a, **k: None)
    assert nb.fetch_day(tmp_path, date(2026, 8, 15)) is None


def test_blank_symbol_footer_rows_are_dropped(monkeypatch, tmp_path):
    """Some legacy files carry a trailing summary row with a blank symbol, which
    crashed a previous backfill on a NOT NULL constraint."""
    body = _LEGACY_CSV + ",,,,,,,,,,,,,\n"
    monkeypatch.setattr(
        nb, "_cached_or_fetch", lambda *a, **k: _zip_bytes("cm.csv", body)
    )
    df = nb.fetch_day(tmp_path, date(2020, 1, 1))
    assert len(df) == 2
    assert (df.symbol == "").sum() == 0


# ----------------------------------------------------------------- delivery
def test_delivery_parses_padded_columns(fake_net, tmp_path):
    """sec_bhavdata_full is a plain CSV (not zipped) with space-padded headers."""
    df = nb.fetch_delivery_day(tmp_path, date(2026, 8, 28))
    assert list(df.columns) == ["symbol", "series", "date", "deliv_qty", "deliv_pct"]
    assert df.iloc[0].symbol == "RELIANCE"
    assert df.iloc[0].deliv_pct == 45.0


# ---------------------------------------------------------------- backfill
def test_backfill_is_resumable_and_skips_completed_days(fake_net, tmp_path):
    """Resumability is a PROPERTY, not a nicety: a previous build accumulated
    everything in memory and wrote at the end, and one interruption lost 1,652
    days of work. A second run must skip without issuing a single request."""
    calls, _ = fake_net
    db, pq = tmp_path / "hot.duckdb", tmp_path / "pq"

    with Store(db, pq) as s:
        first = nb.backfill(s, tmp_path, date(2026, 8, 26), date(2026, 8, 28), progress=False)
    assert first["days_ok"] == 3 and first["skipped"] == 0

    n_after_first = len(calls)
    with Store(db, pq) as s:
        second = nb.backfill(s, tmp_path, date(2026, 8, 26), date(2026, 8, 28), progress=False)
    assert second["skipped"] == 3
    assert second["days_ok"] == 0
    assert len(calls) == n_after_first, "resume re-fetched days it had already ingested"


def test_backfill_skips_weekends_without_requesting_them(fake_net, tmp_path):
    """NSE never trades a weekend. Requesting them wastes a third of the
    backfill's wall-clock on guaranteed 404s."""
    calls, _ = fake_net
    with Store(tmp_path / "h.duckdb", tmp_path / "pq") as s:
        # 2026-08-29 is a Saturday, 2026-08-30 a Sunday.
        nb.backfill(s, tmp_path, date(2026, 8, 29), date(2026, 8, 30), progress=False)
    assert calls == []


def test_a_failing_day_is_recorded_not_raised(monkeypatch, tmp_path):
    """One bad day must not abort a 16-year backfill, and an UNRECORDED failure
    is a silent hole in the spine -- so it is written to ingest_runs as failed
    and therefore retried on the next run."""

    def boom(cache_root, kind, d, url, ext):
        raise RuntimeError("NSE had a moment")

    monkeypatch.setattr(nb, "_cached_or_fetch", boom)
    db, pq = tmp_path / "hot.duckdb", tmp_path / "pq"
    with Store(db, pq) as s:
        stats = nb.backfill(s, tmp_path, date(2026, 8, 28), date(2026, 8, 28), progress=False)
        assert stats["days_failed"] == 1
        # failed days are NOT considered complete, so a re-run retries them
        assert s.completed_units("nse_bhavcopy") == set()

    with Reader(pq) as r:
        runs = r.ingest_runs()
        assert len(runs) == 1
        assert runs.iloc[0].status == "failed"
        assert "NSE had a moment" in runs.iloc[0].error


def test_backfill_writes_are_visible_to_a_lock_free_reader(fake_net, tmp_path):
    db, pq = tmp_path / "hot.duckdb", tmp_path / "pq"
    with Store(db, pq) as s:
        nb.backfill(s, tmp_path, date(2026, 8, 26), date(2026, 8, 28), progress=False)

    with Reader(pq) as r:
        eq = r.bhavcopy_eq()  # defaults to series='EQ'
        assert set(eq.symbol) == {"RELIANCE"}
        assert eq.date.nunique() == 3
        assert pd.api.types.is_datetime64_any_dtype(eq["date"])
        assert len(r.sql("SELECT * FROM {bhavcopy_delivery}")) == 3


# --------------------------------------------- the legacy era is not ONE format
_LEGACY_2010_CSV = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,\n"
    "20MICRONS,EQ,47.00,48.00,46.50,47.55,47.55,47.00,36282,1725210.00,04-JAN-2010,\n"
    "SOMEBOND,NA,101.00,101.50,100.50,101.20,101.20,101.00,500,50600.00,04-JAN-2010,\n"
)


def test_2010_era_without_TOTALTRADES_parses(monkeypatch, tmp_path):
    """Regression, found on the first REAL backfill run.

    The "legacy" era is not one format. Measured against live NSE files:
    2010-2011 have no TOTALTRADES and no ISIN column; 2012 onward do. Treating
    every mapped column as required raised
    `missing columns ['TOTALTRADES']` and failed EVERY day of 2010-2011 --
    two entire years silently absent from the spine.

    n_trades must therefore be optional and land as null, because its absence is
    a fact about NSE's file format, not a bug.
    """
    monkeypatch.setattr(
        nb, "_cached_or_fetch", lambda *a, **k: _zip_bytes("cm04JAN2010bhav.csv", _LEGACY_2010_CSV)
    )
    df = nb.fetch_day(tmp_path, date(2010, 1, 4))

    assert list(df.columns) == nb.CANON_COLS, "schema must match the other eras exactly"
    assert len(df) == 2
    assert df.n_trades.isna().all(), "absent column must be null, not zero or dropped"
    # everything else must still be populated
    r = df[df.symbol == "20MICRONS"].iloc[0]
    assert (r.close, r.volume, r.prev_close) == (47.55, 36282.0, 47.0)
    assert set(df.series) == {"EQ", "NA"}


def test_a_genuinely_missing_REQUIRED_column_still_raises(monkeypatch, tmp_path):
    """Optional-column tolerance must not become "accept anything". A missing
    CLOSE is a real format break and must fail loudly rather than write nulls."""
    broken = _LEGACY_2010_CSV.replace("CLOSE,LAST", "LAST")
    monkeypatch.setattr(nb, "_cached_or_fetch", lambda *a, **k: _zip_bytes("cm.csv", broken))
    with pytest.raises(ValueError, match="missing required columns"):
        nb.fetch_day(tmp_path, date(2010, 1, 4))


def test_ingest_telemetry_is_visible_to_readers_during_a_long_run(fake_net, tmp_path):
    """The pipeline monitor must not be blind while a backfill runs.

    `ingest_runs` lives in DuckDB, which lock-free readers only see after a
    publish. Publishing solely at the end would leave the health screen dark for
    the entire hour that is most worth watching -- so the backfill publishes
    periodically, mid-run.
    """
    db, pq = tmp_path / "hot.duckdb", tmp_path / "pq"
    store = Store(db, pq)
    try:
        nb.backfill(
            store, tmp_path, date(2026, 8, 3), date(2026, 8, 7),
            progress=False, publish_every=2,
        )
        # deliberately NOT closed/published -- mid-run state
        with Reader(pq) as r:
            runs = r.ingest_runs()
        assert len(runs) >= 4, "telemetry not visible mid-run; monitor would be blind"
        assert set(runs.status) <= {"ok", "empty", "failed"}
    finally:
        store.close()
