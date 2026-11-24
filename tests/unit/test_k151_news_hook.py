"""PLAN-M M5.4: every fetched headline is appended to the news_headlines dataset (k151_nb_news input)."""

from __future__ import annotations

from datetime import datetime, timedelta

from stocksense.core.clock import IST, SteppingClock
from stocksense.data import io
from stocksense.marketlab import news as N
from tests.unit.test_ml_news import FakeSources, headline


def test_search_appends_headlines_once_even_when_refetched(tmp_data_root):
    clock = SteppingClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    src = FakeSources([headline("Profit jumps"), headline("Plain news")])
    svc = N.NewsService(src, clock, tmp_data_root)
    svc.search("energy stocks")
    clock.advance(timedelta(seconds=901))
    svc.search("energy stocks")  # refetched after the cache expired: same rows, no duplicates
    df = io.read(tmp_data_root, "news_headlines").collect()
    assert sorted(df["title"].to_list()) == ["Plain news", "Profit jumps"]
    assert set(df["query"].to_list()) == {"energy stocks"} and set(df["symbol"].to_list()) == {""}
    assert df["ts"].dtype == io.get_spec("news_headlines").schema["ts"]


def test_stock_headlines_carry_the_symbol(tmp_data_root):
    clock = SteppingClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    N.NewsService(FakeSources([headline("Reliance wins")]), clock, tmp_data_root).stock(
        "RELIANCE", "Reliance Industries Ltd"
    )
    df = io.read(tmp_data_root, "news_headlines").collect()
    assert df["symbol"].to_list() == ["RELIANCE"]


def test_without_settings_nothing_is_written_and_nothing_breaks():
    clock = SteppingClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    assert len(N.NewsService(FakeSources([headline("x")]), clock).search("q")) == 1
