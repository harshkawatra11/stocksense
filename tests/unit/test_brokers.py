from __future__ import annotations

import gzip
import json
from datetime import date, datetime

import httpx
import pytest

from stocksense.brokers.angel import AngelClient, candle_windows
from stocksense.brokers.instruments import InstrumentMap
from stocksense.brokers.upstox import UpstoxClient, request_windows
from stocksense.core.clock import IST, SteppingClock
from stocksense.core.config import Secrets
from stocksense.core.errors import BrokerError

T0 = datetime(2026, 11, 11, 10, 0, 7, tzinfo=IST)
TOTP_SECRET = "JBSWY3DPEHPK3PXP"


def _settings(tmp_data_root, **kw):
    sec = Secrets(
        _env_file=None,
        STOCKSENSE_ANGEL_API_KEY="k-secret",
        STOCKSENSE_ANGEL_CLIENT_CODE="C1",
        STOCKSENSE_ANGEL_PASSWORD="pin-secret",
        STOCKSENSE_ANGEL_TOTP_SECRET=TOTP_SECRET,
        **kw,
    )
    return tmp_data_root.model_copy(update={"secrets": sec})


class Server:
    def __init__(self, routes):
        self.routes = routes  # path suffix -> callable(request, n) -> Response
        self.calls: list[tuple[str, datetime]] = []
        self.clock: SteppingClock | None = None

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append((req.url.path, self.clock.now() if self.clock else T0))
        n = sum(1 for p, _ in self.calls if p == req.url.path)
        for suffix, fn in self.routes.items():
            if req.url.path.endswith(suffix):
                return fn(req, n)
        return httpx.Response(404, json={})


def _ok(data):
    return lambda req, n: httpx.Response(
        200, json={"status": True, "message": "SUCCESS", "errorcode": "", "data": data}
    )


LOGIN_OK = _ok({"jwtToken": "JWT", "refreshToken": "R", "feedToken": "F"})


def _client(settings, routes):
    clock = SteppingClock(T0)
    srv = Server(routes)
    srv.clock = clock
    return AngelClient(settings, clock, transport=httpx.MockTransport(srv)), srv, clock


def test_login_success_and_headers(tmp_data_root):
    seen = {}

    def login(req, n):
        seen["h"] = dict(req.headers)
        seen["b"] = json.loads(req.content)
        return LOGIN_OK(req, n)

    c, srv, _ = _client(_settings(tmp_data_root), {"loginByPassword": login})
    c.login()
    assert seen["h"]["x-privatekey"] == "k-secret" and seen["h"]["x-usertype"] == "USER"
    assert seen["b"]["clientcode"] == "C1" and len(seen["b"]["totp"]) == 6


def test_login_totp_error_waits_for_next_boundary_then_retries(tmp_data_root):
    def login(req, n):
        if n == 1:
            return httpx.Response(
                200, json={"status": False, "message": "Invalid totp", "errorcode": "AB1050"}
            )
        return LOGIN_OK(req, n)

    c, srv, clock = _client(_settings(tmp_data_root), {"loginByPassword": login})
    c.login()
    t1, t2 = srv.calls[0][1], srv.calls[1][1]
    # first call at 10:00:07 (>= 30 s boundary 10:00:30 + 1 s)
    assert (t2 - t1).total_seconds() >= 24.0
    assert t2.timestamp() % 30 >= 1.0 - 1e-6 and t2.timestamp() % 30 < 3.0


def test_funds_parse_strings_to_floats(tmp_data_root):
    rms = {"availablecash": "1234.50", "net": "2000", "utiliseddebits": "10.25"}
    c, _, _ = _client(_settings(tmp_data_root), {"loginByPassword": LOGIN_OK, "getRMS": _ok(rms)})
    f = c.funds()
    assert (f.available_cash, f.net, f.utilised) == (1234.5, 2000.0, 10.25)


def test_holdings_empty_and_positions_null(tmp_data_root):
    routes = {
        "loginByPassword": LOGIN_OK,
        "getAllHolding": _ok({"holdings": [], "totalholding": {}}),
        "getPosition": _ok(None),
    }
    c, _, _ = _client(_settings(tmp_data_root), routes)
    assert c.holdings() == [] and c.positions() == []


def test_holdings_parse(tmp_data_root):
    h = {"holdings": [{"tradingsymbol": "TCS-EQ", "quantity": 3, "averageprice": 3500.5, "ltp": 3600.0}]}
    c, _, _ = _client(_settings(tmp_data_root), {"loginByPassword": LOGIN_OK, "getAllHolding": _ok(h)})
    (x,) = c.holdings()
    assert (x.symbol, x.qty, x.avg_price, x.ltp) == ("TCS", 3.0, 3500.5, 3600.0)


FULL = {
    "tradingSymbol": "RELIANCE-EQ",
    "exchFeedTime": "11-Nov-2026 16:08:11",
    "ltp": 1335.4,
    "open": 1333.5,
    "high": 1349.4,
    "low": 1327.0,
    "close": 1324.4,
    "tradeVolume": 17554310,
    "upperCircuit": 1370.4,
    "lowerCircuit": 1290.6,
    "depth": {
        "buy": [{"price": 1335.3, "quantity": 100, "orders": 2}],
        "sell": [{"price": 1335.5, "quantity": 50, "orders": 1}],
    },
}


def test_quote_chunking_120_tokens_is_3_calls_one_second_apart(tmp_data_root):
    c, srv, _ = _client(
        _settings(tmp_data_root),
        {"loginByPassword": LOGIN_OK, "market/v1/quote/": _ok({"fetched": [FULL], "unfetched": []})},
    )
    qs = c.quote_full([str(i) for i in range(120)])
    qcalls = [t for p, t in srv.calls if p.endswith("quote/")]
    assert len(qcalls) == 3 and len(qs) == 3
    gaps = [(b - a).total_seconds() for a, b in zip(qcalls, qcalls[1:], strict=False)]
    assert all(g >= 1.0 for g in gaps)
    q = qs[0]
    assert q.symbol == "RELIANCE" and q.bid == 1335.3 and q.ask == 1335.5 and q.prev_close == 1324.4
    assert q.ts == datetime(2026, 11, 11, 16, 8, 11, tzinfo=IST) and q.source == "angel"
    assert q.upper_circuit == 1370.4


def test_ab1021_backoff_60_then_300_then_error(tmp_data_root):
    def limited(req, n):
        return httpx.Response(200, json={"status": False, "message": "Access denied", "errorcode": "AB1021"})

    c, srv, _ = _client(_settings(tmp_data_root), {"loginByPassword": LOGIN_OK, "getRMS": limited})
    with pytest.raises(BrokerError):
        c.funds()
    t = [t for p, t in srv.calls if p.endswith("getRMS")]
    assert len(t) == 3
    assert (t[1] - t[0]).total_seconds() >= 60 and (t[2] - t[1]).total_seconds() >= 300


def test_empty_body_is_treated_as_rate_limit_and_recovers(tmp_data_root):
    def flaky(req, n):
        return httpx.Response(200, content=b"") if n == 1 else _ok({"availablecash": "1"})(req, n)

    c, _, _ = _client(_settings(tmp_data_root), {"loginByPassword": LOGIN_OK, "getRMS": flaky})
    assert c.funds().available_cash == 1.0


def test_auth_error_relogs_in_once(tmp_data_root):
    def rms(req, n):
        if n == 1:
            return httpx.Response(
                200, json={"status": False, "message": "Invalid Token", "errorcode": "AG8001"}
            )
        return _ok({"availablecash": "5"})(req, n)

    c, srv, _ = _client(_settings(tmp_data_root), {"loginByPassword": LOGIN_OK, "getRMS": rms})
    assert c.funds().available_cash == 5.0
    assert sum(1 for p, _ in srv.calls if p.endswith("loginByPassword")) == 2


def test_candles_span_split_and_parse(tmp_data_root):
    rows = [["2026-11-09T09:20:00+05:30", 2, 3, 1, 2.5, 10], ["2026-11-09T09:15:00+05:30", 1, 2, 1, 1.5, 5]]
    c, srv, _ = _client(_settings(tmp_data_root), {"loginByPassword": LOGIN_OK, "getCandleData": _ok(rows)})
    df = c.candles(
        "2885",
        "FIVE_MINUTE",
        datetime(2026, 1, 1, 9, 15, tzinfo=IST),
        datetime(2026, 5, 1, 15, 30, tzinfo=IST),
    )
    # 120 days with a 100-day span -> 2 requests; duplicates across windows collapse
    assert sum(1 for p, _ in srv.calls if p.endswith("getCandleData")) == 2
    assert df.height == 2 and df["ts"][0] < df["ts"][1]
    assert len(candle_windows(datetime(2026, 1, 1), datetime(2026, 1, 31), 30)) == 1


def test_no_secret_in_logs(tmp_data_root, capfd, caplog):
    c, _, _ = _client(
        _settings(tmp_data_root), {"loginByPassword": LOGIN_OK, "getRMS": _ok({"availablecash": "1"})}
    )
    c.funds()
    out = capfd.readouterr()
    blob = out.out + out.err + caplog.text
    for secret in ("k-secret", "pin-secret", TOTP_SECRET, "JWT"):
        assert secret not in blob


def test_angel_credentials_missing_raises(tmp_data_root):
    c = AngelClient(tmp_data_root.model_copy(update={"secrets": Secrets(_env_file=None)}), SteppingClock(T0))
    with pytest.raises(BrokerError):
        c.login()


# ---- Upstox ---------------------------------------------------------------------------------------


def _up(tmp_data_root, handler, token="UP-TOKEN"):
    s = tmp_data_root.model_copy(
        update={"secrets": Secrets(_env_file=None, STOCKSENSE_UPSTOX_ANALYTICS_TOKEN=token)}
    )
    clock = SteppingClock(T0)
    return UpstoxClient(s, clock, transport=httpx.MockTransport(handler)), clock


def test_upstox_missing_token_raises(tmp_data_root):
    c = UpstoxClient(tmp_data_root.model_copy(update={"secrets": Secrets(_env_file=None)}), SteppingClock(T0))
    with pytest.raises(BrokerError, match="upstox token not set"):
        c.candles("NSE_EQ|X", "minutes", 1, date(2026, 1, 1), date(2026, 1, 2))


def test_upstox_70_days_of_1m_is_3_requests_ascending_and_spaced(tmp_data_root):
    calls = []

    def handler(req):
        calls.append((str(req.url.path), clock.now(), req.headers["authorization"]))
        # newest first, as the API returns it
        data = [
            ["2026-11-09T09:16:00+05:30", 2, 3, 1, 2.5, 7, 0],
            ["2026-11-09T09:15:00+05:30", 1, 2, 1, 1.5, 5, 0],
        ]
        return httpx.Response(200, json={"status": "success", "data": {"candles": data}})

    c, clock = _up(tmp_data_root, handler)
    df = c.candles("NSE_EQ|INE002A01018", "minutes", 1, date(2026, 1, 1), date(2026, 3, 11))  # 70 days
    assert len(calls) == 3
    assert [x[2] for x in calls] == ["Bearer UP-TOKEN"] * 3
    assert calls[0][0].endswith("/minutes/1/2026-01-31/2026-01-01")  # to_date first, then from_date
    assert all((b[1] - a[1]).total_seconds() >= 1.0 for a, b in zip(calls, calls[1:], strict=False))
    assert df["ts"].is_sorted() and df.height == 2


def test_upstox_request_windows():
    assert len(request_windows("minutes", 1, date(2026, 1, 1), date(2026, 3, 11))) == 3
    assert len(request_windows("minutes", 30, date(2026, 1, 1), date(2026, 3, 11))) == 1
    assert len(request_windows("days", 1, date(2012, 1, 1), date(2026, 1, 1))) == 2


def test_upstox_bad_range_error_surfaces_code(tmp_data_root):
    def handler(req):
        return httpx.Response(400, json={"status": "error", "errors": [{"errorCode": "UDAPI1148"}]})

    c, _ = _up(tmp_data_root, handler)
    with pytest.raises(BrokerError, match="UDAPI1148"):
        c.candles("k", "minutes", 1, date(2026, 1, 1), date(2026, 1, 2))


def test_upstox_quotes_parse(tmp_data_root):
    payload = {
        "status": "success",
        "data": {
            "NSE_EQ:RELIANCE": {
                "last_price": 1335.4,
                "ohlc": {"open": 1333.5, "high": 1349.4, "low": 1327.0, "close": 1324.4},
                "volume": 100,
                "depth": {
                    "buy": [{"price": 1335.3, "quantity": 5}],
                    "sell": [{"price": 1335.5, "quantity": 6}],
                },
                "timestamp": "2026-11-11T10:00:00.000+05:30",
            }
        },
    }
    c, _ = _up(tmp_data_root, lambda req: httpx.Response(200, json=payload))
    (q,) = c.quotes(["NSE_EQ|INE002A01018"])
    assert q.symbol == "RELIANCE" and q.bid == 1335.3 and q.ask == 1335.5 and q.source == "upstox"


# ---- instruments ----------------------------------------------------------------------------------


def test_instrument_map_parses_and_caches_daily(tmp_data_root):
    angel = [
        {"exch_seg": "NSE", "symbol": "TCS-EQ", "token": "11536"},
        {"exch_seg": "NSE", "symbol": "TCS-BE", "token": "1"},
        {"exch_seg": "BSE", "symbol": "TCS-EQ", "token": "2"},
    ]
    up = [
        {
            "segment": "NSE_EQ",
            "instrument_type": "EQ",
            "trading_symbol": "TCS",
            "instrument_key": "NSE_EQ|INE467B01029",
        },
        {"segment": "NSE_EQ", "instrument_type": "BE", "trading_symbol": "ZZ", "instrument_key": "NSE_EQ|ZZ"},
    ]
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        if "NSE.json.gz" in str(req.url):
            return httpx.Response(200, content=gzip.compress(json.dumps(up).encode()))
        return httpx.Response(200, content=json.dumps(angel).encode())

    m = InstrumentMap(tmp_data_root, SteppingClock(T0), transport=httpx.MockTransport(handler))
    assert m.angel_token("TCS") == "11536" and m.angel_token("NOPE") is None
    assert m.upstox_key("TCS") == "NSE_EQ|INE467B01029" and m.upstox_key("ZZ") is None
    assert n["c"] == 2
    m2 = InstrumentMap(tmp_data_root, SteppingClock(T0), transport=httpx.MockTransport(handler))
    assert m2.angel_token("TCS") == "11536"
    assert n["c"] == 2  # today's cache files are reused
