from __future__ import annotations

from datetime import datetime

import httpx
import pytest

from stocksense.core.clock import IST, SteppingClock
from stocksense.core.errors import DataNotAvailable
from stocksense.data.nse_http import NseClient


def _client(handler, clock=None):
    clock = clock or SteppingClock(datetime(2026, 11, 11, 10, 0, tzinfo=IST))
    return NseClient(clock, transport=httpx.MockTransport(handler)), clock


def test_archive_404_returns_none():
    c, _ = _client(lambda r: httpx.Response(404))
    assert c.get_archive("https://nsearchives.nseindia.com/x.csv") is None


@pytest.mark.parametrize("body", [b"<!DOCTYPE html><html></html>", b"<html><body>nf</body></html>"])
def test_archive_html_body_with_200_is_none(body):
    c, _ = _client(lambda r: httpx.Response(200, content=body))
    assert c.get_archive("https://nsearchives.nseindia.com/x.csv") is None


def test_archive_returns_bytes_and_headers():
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(200, content=b"SYMBOL,NAME\n")

    c, _ = _client(h)
    assert c.get_archive("https://nsearchives.nseindia.com/x.csv") == b"SYMBOL,NAME\n"
    hd = seen[0].headers
    assert hd["user-agent"].startswith("Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    assert hd["accept-language"] == "en-US,en;q=0.9"
    assert hd["accept"] == "*/*"
    assert hd["referer"] == "https://www.nseindia.com/"


def test_api_headers_and_params():
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(200, json={"CM": []})

    c, _ = _client(h)
    out = c.get_api("/api/holiday-master", {"type": "trading"}, "resources/holidays")
    assert out == {"CM": []}
    r = seen[0]
    assert str(r.url) == "https://www.nseindia.com/api/holiday-master?type=trading"
    assert r.headers["accept"] == "application/json, text/plain, */*"
    assert r.headers["referer"] == "https://www.nseindia.com/resources/holidays"


def test_api_403_then_200_after_exactly_two_warmup_gets():
    urls = []

    def h(req):
        urls.append(str(req.url))
        api = req.url.path.startswith("/api/")
        n_api = sum(u.startswith("https://www.nseindia.com/api/") for u in urls)
        if api and n_api == 1:
            return httpx.Response(403)
        return httpx.Response(200, json={"ok": 1}) if api else httpx.Response(200, text="home")

    c, _ = _client(h)
    assert c.get_api("/api/x", {}, "r") == {"ok": 1}
    assert urls == [
        "https://www.nseindia.com/api/x",
        "https://www.nseindia.com/",
        "https://www.nseindia.com/market-data/live-equity-market",
        "https://www.nseindia.com/api/x",
    ]


def test_api_still_403_after_two_warmups_raises():
    c, _ = _client(lambda r: httpx.Response(403))
    with pytest.raises(DataNotAvailable):
        c.get_api("/api/x", {}, "r")


def test_api_non_json_raises_data_not_available():
    c, _ = _client(lambda r: httpx.Response(200, text="<html>nope</html>"))
    with pytest.raises(DataNotAvailable):
        c.get_api("/api/x", {}, "r")


def test_500_500_200_succeeds_with_three_calls():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(500) if len(calls) < 3 else httpx.Response(200, content=b"ok")

    c, _ = _client(h)
    assert c.get_archive("https://nsearchives.nseindia.com/x") == b"ok"
    assert len(calls) == 3


def test_persistent_500_gives_up_after_five_attempts():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(500)

    c, _ = _client(h)
    with pytest.raises(DataNotAvailable):
        c.get_archive("https://nsearchives.nseindia.com/x")
    assert len(calls) == 5


def test_429_is_retried():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(429) if len(calls) == 1 else httpx.Response(200, content=b"ok")

    c, _ = _client(h)
    assert c.get_archive("https://nsearchives.nseindia.com/x") == b"ok"
    assert len(calls) == 2


def test_spacing_at_least_0_6_seconds_between_calls():
    clock = SteppingClock(datetime(2026, 11, 11, 10, 0, tzinfo=IST))
    stamps = []

    def h(req):
        stamps.append(clock.now())
        return httpx.Response(200, content=b"ok")

    c, _ = _client(h, clock)
    for _ in range(4):
        c.get_archive("https://nsearchives.nseindia.com/x")
    # 4 calls -> 3 gaps, each >= 0.6 s (and <= 0.8 s: 0.6 + jitter up to 0.2)
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:], strict=False)]
    assert len(gaps) == 3
    assert all(0.6 <= g <= 0.8 + 1e-9 for g in gaps)


def test_close_is_idempotent():
    c, _ = _client(lambda r: httpx.Response(200))
    c.close()
    c.close()
