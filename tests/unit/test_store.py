"""Idempotency: re-ingesting the same date must not duplicate rows —
docs/05-nightly-pipeline.md's requirement that a re-run of any step
produce the same store state, not accumulate duplicates."""

from __future__ import annotations

import pandas as pd
import pytest

from stocksense.data.store import Store


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _sample_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "open": [100.0, 101.0, 50.0],
            "high": [102.0, 103.0, 51.0],
            "low": [99.0, 100.0, 49.0],
            "close": [101.0, 102.0, 50.5],
            "adj_close": [101.0, 102.0, 50.5],
            "volume": [1000.0, 1100.0, 500.0],
            "source": ["yfinance", "yfinance", "yfinance"],
        }
    )


def test_upsert_is_idempotent(tmp_store: Store) -> None:
    df = _sample_candles()
    n1 = tmp_store.upsert_candles(df)
    n2 = tmp_store.upsert_candles(df)  # re-run identical ingestion
    assert n1 == n2 == 3

    result = tmp_store.read_candles()
    assert len(result) == 3  # not 6 — no duplication


def test_upsert_updates_changed_values(tmp_store: Store) -> None:
    df = _sample_candles()
    tmp_store.upsert_candles(df)

    revised = df.copy()
    revised.loc[0, "close"] = 999.0  # simulate a corrected value

    tmp_store.upsert_candles(revised)
    result = tmp_store.read_candles(symbols=["AAA"])
    row = result[result["date"].astype(str) == "2024-01-01"]
    assert float(row["close"].iloc[0]) == 999.0
    assert len(result) == 2  # still just AAA's two rows, no duplicate
