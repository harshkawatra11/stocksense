from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from stocksense.core.clock import IST, FixedClock, SteppingClock
from stocksense.marketlab import news as N
from stocksense.marketlab.sources import SourceError


class FakeSources:
    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self.items = items if items is not None else []
        self.calls: list[tuple[str, int, int]] = []
        self.fail = False

    def google_news(self, q: str, days: int = 30, limit: int = 20) -> list[dict[str, Any]]:
        self.calls.append((q, days, limit))
        if self.fail:
            raise SourceError("google", "down")
        return [dict(i) for i in self.items]


def headline(title: str) -> dict[str, Any]:
    return {
        "time": "2026-11-18T05:00:00+00:00",
        "title": title,
        "source": "Wire",
        "link": "https://x.invalid/1",
    }


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Q2 results beat estimates", "RESULTS"),
        ("Net profit rises 12%", "RESULTS"),
        ("RBI policy keeps repo rate unchanged", "POLICY"),
        ("Government imposes duty on steel imports", "POLICY"),
        ("Company launches new product", "NEWS"),
        ("Q3 results and RBI policy", "RESULTS"),  # results wins when both match
    ],
)
def test_tag(title, expected):
    assert N.tag(title) == expected


def test_search_tags_and_caches_for_fifteen_minutes():
    clock = SteppingClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    src = FakeSources([headline("Profit jumps"), headline("Plain news")])
    svc = N.NewsService(src, clock)
    first = svc.search("energy stocks")
    assert [h["tag"] for h in first] == ["RESULTS", "NEWS"]
    svc.search("energy stocks")
    assert len(src.calls) == 1
    clock.advance(timedelta(seconds=901))
    svc.search("energy stocks")
    assert len(src.calls) == 2


def test_search_returns_the_stale_copy_when_the_source_fails():
    clock = SteppingClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    src = FakeSources([headline("Profit jumps")])
    svc = N.NewsService(src, clock)
    svc.search("q")
    clock.advance(timedelta(seconds=2000))
    src.fail = True
    assert [h["title"] for h in svc.search("q")] == ["Profit jumps"]
    assert svc.search("never fetched") == []


def test_sector_without_wait_reads_only_what_the_warmer_stored():
    clock = FixedClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    src = FakeSources([headline("Plain news")])
    svc = N.NewsService(src, clock)
    assert svc.sector("Energy", wait=False) == []
    assert src.calls == []
    svc.sector("Energy")
    assert src.calls[0][0] == N.QUERY["Energy"]
    assert len(svc.sector("Energy", wait=False)) == 1


def test_stock_query_strips_company_suffixes_and_uses_sixty_days():
    clock = FixedClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    src = FakeSources([])
    svc = N.NewsService(src, clock)
    svc.stock("RELIANCE", "Reliance Industries Limited")
    q, days, limit = src.calls[0]
    assert q == '"Reliance" OR RELIANCE' and days == 60 and limit == 12


def test_sector_without_a_query_uses_its_name():
    clock = FixedClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    src = FakeSources([])
    N.NewsService(src, clock).sector("Fancy")
    assert src.calls[0][0] == "Fancy stocks india"


def test_every_map_sector_has_a_query():
    from stocksense.marketlab.sectors import SECTORS

    assert {s["name"] for s in SECTORS} == set(N.QUERY)
