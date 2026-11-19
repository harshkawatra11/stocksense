"""Live wiring (G2-G1): live ids, the predict stage and the live score source on the synthetic market."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from stocksense.core.clock import IST, SteppingClock
from stocksense.core.errors import DataNotAvailable
from stocksense.core.state import State
from stocksense.data import io
from stocksense.features.build import load_features
from stocksense.learn.nightly import fit_production
from stocksense.live import wiring
from stocksense.models.base import ModelSpec
from stocksense.validation.walkforward import run_walk_forward
from tests.fixtures.g1_world import LOCKBOX_START, build_g1_world

SPEC = ModelSpec(family="s1-ridge", target="y_c2o_10", universe="u500", entry="close")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    w = build_g1_world(tmp_path_factory.mktemp("wiring"), planted_ic=0.08)
    clock = SteppingClock(datetime(2020, 3, 20, 18, 30, tzinfo=IST))
    st = State(w.settings, clock)
    st.migrate()
    res = run_walk_forward(
        SPEC,
        w.settings,
        st,
        folds=w.folds,
        calendar=w.calendar,
        registry=w.registry,
        seeds=(1, 2),
        inner_months=2,
        trial_family="g2-test",
    )
    as_of = w.calendar.prev_session(LOCKBOX_START)
    mid = fit_production(
        SPEC,
        w.settings,
        st,
        as_of,
        seeds=(1, 2),
        calendar=w.calendar,
        inner_months=2,
        allow_beyond_research=True,
    )
    st.kv_set("live_model:s1", mid)
    st.kv_set("live_wf_trial:s1", res.trial_id)
    feats = load_features(w.settings, "production", columns=["planted_c2o_10"]).collect()
    last = feats["date"].max()
    ctx = SimpleNamespace(settings=w.settings, state=st, clock=clock, calendar=w.calendar)
    return SimpleNamespace(w=w, st=st, ctx=ctx, mid=mid, last=last, trial=res.trial_id)


def test_live_ids_initialises_missing_keys_with_defaults(tmp_path):
    from tests.fixtures.synth import _settings_for

    st = State(_settings_for(tmp_path))
    st.migrate()
    ids = wiring.live_ids(st)
    assert ids["live_model:s1"] == "s1-gbm-y_c2o_10-120055de-20250930"
    assert ids["live_model:s2"] == "s2-gbm-y_oc-4fd8ff9b-20250930"
    assert ids["live_wf_trial:s1"] is None  # no finished walk-forward trial in a fresh state
    assert st.kv_get("live_model:s1") == ids["live_model:s1"]
    st.kv_set("live_model:s1", "other")
    assert wiring.live_ids(st)["live_model:s1"] == "other"  # existing values are kept


def test_predict_stage_writes_live_preds_for_every_u500_name(env):
    out = wiring.stage_predict(env.ctx, env.last)
    uni = io.read(env.w.settings, "panel_universe", start=env.last, end=env.last).collect()
    n = int(uni["u500"].sum())
    assert out == f"s1:{n} names"
    p = io.read(env.w.settings, "preds", start=env.last, end=env.last, sub=env.mid).collect()
    assert p.height == n and set(p["mode"]) == {"live"}


def test_live_score_source_ranks_and_gives_nonnegative_er(env):
    wiring.stage_predict(env.ctx, env.last)
    sc = wiring.LiveScoreSource(env.w.settings, env.st).scores(env.last)
    assert sc is not None and len(sc.score) > 50
    v = np.array(list(sc.score.values()))
    assert v.min() >= -0.5 - 1e-9 and v.max() <= 0.5 + 1e-9
    assert min(sc.er.values()) >= 0.0 and max(sc.er.values()) > 0.0
    assert (
        wiring.LiveScoreSource(env.w.settings, env.st).scores(env.w.calendar.prev_session(env.last)) is None
    )


def test_decile_er_map_option(env):
    sc = wiring.LiveScoreSource(env.w.settings, env.st, er_map="decile").scores(env.last)
    assert sc is not None and all(np.isfinite(x) and x >= 0.0 for x in sc.er.values())


def test_missing_features_raise_data_not_available(env):
    with pytest.raises(DataNotAvailable):
        wiring.predict_frame(env.w.settings, env.mid, env.last.replace(year=2031))


def test_stage_in_jobs_uses_the_hook_and_skips_without_one(env):
    from stocksense.live import jobs

    assert jobs._stage_predict(SimpleNamespace(hook=lambda n: None), env.last).startswith("skipped")
    hooks = {"predict": wiring.stage_predict}
    ctx = SimpleNamespace(**vars(env.ctx), hook=hooks.get)
    assert jobs._stage_predict(ctx, env.last).startswith("s1:")
    assert "predict" not in jobs.STUB_STAGES


# ---- G2-G2: ledger book, evening plan, 15:10 entries ---------------------------------------------------

import json  # noqa: E402

from stocksense.brokers.types import Quote  # noqa: E402
from stocksense.costs.india import CostBreakdown  # noqa: E402
from stocksense.live.jobs import JobContext  # noqa: E402
from stocksense.paper.ledger import NewOrder  # noqa: E402

CAP = 2_000_000.0


class FakeHub:
    def __init__(self, quotes):
        self.q, self.asked = quotes, []

    def get(self, symbols, held=None):
        self.asked.append(list(symbols))
        return {s: self.q[s] for s in symbols if s in self.q}


def _fee():
    return CostBreakdown(10.0, 0, 0, 0, 0, 0, 0, 10.0)


@pytest.fixture
def lctx(env, monkeypatch):
    monkeypatch.setattr("stocksense.portfolio.risk.UNIVERSE_MIN", 50)
    wiring.stage_predict(env.ctx, env.last)
    c = JobContext(
        env.w.settings, env.ctx.clock, env.st, env.w.calendar, hub=FakeHub({}),
        notifier=SimpleNamespace(send=lambda t: True),
    )  # fmt: skip
    for k, v in {"capital_ref_inr": repr(CAP), "capital_stale": "0", "capital_date": env.last.isoformat(),
                 "capital_read_at": env.ctx.clock.now().isoformat()}.items():  # fmt: skip
        env.st.kv_set(k, v)
    yield c
    con = env.st.connect()
    try:
        for t in ("decisions", "orders", "fills", "fees", "cash_events", "marks", "mark_prices"):
            con.execute(f"DELETE FROM {t}")
    finally:
        con.close()
    wiring._ACCESS.clear()


def _buy(c, env, sid, sym, qty, price, d):
    dec = c.ledger.record_decision("main", "s1", d, CAP, False, {}, {})
    c.ledger.add_orders(dec, "main", [NewOrder(sid, sym, "BUY", "delivery", "close", d, qty, price, "t")])
    oid = c.ledger.pending_orders(d, "close")[-1].order_id
    c.ledger.fill_order(oid, f"{d}T15:22:00+05:30", price, qty, 0.0, None, None, price, "t", _fee())


def _two_holdings(c, env, bad_entry=False):
    """The two best-scored names of `last` (so rank decay does not fire), entered 3 sessions earlier."""
    sc = wiring.LiveScoreSource(env.w.settings, env.st).scores(env.last)
    top = sorted(sc.score, key=lambda x: -sc.score[x])[:2]
    daily = io.read(
        env.w.settings, "panel_daily", start=env.last, end=env.last, columns=["sid", "symbol", "close"]
    )
    by_sid = {int(r["sid"]): r for r in daily.collect().to_dicts()}
    rows = [by_sid[x] for x in top]
    e0 = env.w.calendar.shift(env.last, -3)
    for i, r in enumerate(rows):
        px = float(r["close"]) * (10.0 if bad_entry and i == 0 else 1.0)
        _buy(c, env, int(r["sid"]), r["symbol"], 100, px, e0)
    return rows, e0


def test_ledger_book_reconstructs_holdings_and_held_sessions(env, lctx):
    rows, e0 = _two_holdings(lctx, env)
    b = wiring.ledger_book(lctx, env.last, capital=CAP)
    assert sorted(b.holdings) == sorted(int(r["sid"]) for r in rows)
    assert all(h.held == 3 and h.qty == 100 and h.vol_at_entry > 0 for h in b.holdings.values())
    assert 0.0 < b.cash < CAP and b.dd == 0.0 and not b.stopped


def test_plan_records_exits_as_next_open_sells_and_is_idempotent(env, lctx):
    rows, _ = _two_holdings(lctx, env, bad_entry=True)  # the first name entered at 10x: its stop is hit
    out = wiring.hook_s1_plan(lctx, env.last)
    assert out.startswith("exits=") and "planned_entries=" in out
    d1 = env.w.calendar.next_session(env.last)
    sells = lctx.ledger.pending_orders(d1, "open")
    assert all(o.side == "SELL" and o.book == "main" and o.qty == 100 for o in sells)
    stop = [o for o in sells if o.reason == "stop"]
    assert [o.sid for o in stop] == [int(rows[0]["sid"])]
    plan = json.loads(env.st.kv_get(f"s1_plan:{env.last.isoformat()}"))
    assert plan["cap_ref"] == CAP and plan["exits"][str(rows[0]["sid"])] == "stop"
    assert len(sells) == len(plan["exits"]) == len(plan["sells"])
    n_before = len(lctx.ledger.orders_for_date(d1))
    wiring.hook_s1_plan(lctx, env.last)
    assert len(lctx.ledger.orders_for_date(d1)) == n_before  # same ids: nothing new


def test_plan_without_capital_reference_is_skipped(env, lctx):
    env.st.kv_set("capital_ref_inr", "")
    con = env.st.connect()
    try:
        con.execute("DELETE FROM kv WHERE key='capital_ref_inr'")
    finally:
        con.close()
    assert wiring.hook_s1_plan(lctx, env.last).startswith("skipped")


def _stored_plan(env, cap_ref):
    plan = {
        "date": env.last.isoformat(), "cap_ref": cap_ref, "allow_entries": True, "note": "", "checks": [],
        "policy": wiring.live_policy(env.st),
        "entries": [
            {"sid": 1, "symbol": "AAA", "weight": 0.10, "score": 0.4, "rank": 1, "er": 0.01, "ref_close": 100.0},  # noqa: E501
            {"sid": 2, "symbol": "BBB", "weight": 0.10, "score": 0.3, "rank": 2, "er": 0.01, "ref_close": 50.0},  # noqa: E501
            {"sid": 3, "symbol": "CCC", "weight": 0.10, "score": 0.2, "rank": 3, "er": 0.01, "ref_close": 20.0},  # noqa: E501
        ],
    }  # fmt: skip
    env.st.kv_set(f"s1_plan:{env.last.isoformat()}", json.dumps(plan))


def _q(sym, ltp, upper=None):
    return Quote(symbol=sym, ts=datetime(2020, 3, 23, 15, 10, tzinfo=IST), ltp=ltp, open=ltp, high=ltp, low=ltp,  # noqa: E501
                 prev_close=ltp, volume=1.0, bid=ltp, ask=ltp, bid_qty=1, ask_qty=1,
                 upper_circuit=upper, lower_circuit=None, source="t")  # fmt: skip


def test_entries_are_resized_to_todays_capital_and_drop_a_name_at_the_upper_circuit(env, lctx):
    d = env.w.calendar.next_session(env.last)
    _stored_plan(env, CAP)
    lctx._hub = FakeHub({"AAA": _q("AAA", 101.0), "BBB": _q("BBB", 50.0, upper=50.0), "CCC": _q("CCC", 20.0)})
    env.st.kv_set("free_cash_frac:main", "1.0")
    cap = CAP * 1.03  # within the 5% tolerance: the stored plan is re-sized, not re-planned
    [(book, orders, payload)] = wiring.hook_s1_orders(lctx, d, cap)
    assert book == "main" and payload["replanned"] == ""
    got = {o.symbol: o.qty for o in orders}
    assert got == {"AAA": int(0.10 * cap / 101.0), "CCC": int(0.10 * cap / 20.0)}
    assert all(o.side == "BUY" and o.when_kind == "close" and o.trade_date == d for o in orders)
    assert payload["dropped"] == ["BBB:upper_circuit"] and len(payload["orders"]) == 2


def test_entries_cash_cap_leaves_the_cushion(env, lctx):
    d = env.w.calendar.next_session(env.last)
    _stored_plan(env, CAP)
    lctx._hub = FakeHub({"AAA": _q("AAA", 100.0)})
    env.st.kv_set("free_cash_frac:main", "0.05")  # only 5% of capital is free: less than the 10% weight
    [(_, orders, _p)] = wiring.hook_s1_orders(lctx, d, CAP)
    a = next(o for o in orders if o.symbol == "AAA")
    assert a.qty == int((0.05 * CAP - 30.0) // 100.0)


def test_blocked_plan_gives_no_orders_but_still_a_decision_payload(env, lctx):
    d = env.w.calendar.next_session(env.last)
    _stored_plan(env, CAP)
    p = json.loads(env.st.kv_get(f"s1_plan:{env.last.isoformat()}"))
    p["allow_entries"], p["note"] = False, "entries blocked by pre-trade checks"
    env.st.kv_set(f"s1_plan:{env.last.isoformat()}", json.dumps(p))
    [(book, orders, payload)] = wiring.hook_s1_orders(lctx, d, CAP)
    assert orders == [] and payload["allow_entries"] is False


def test_production_hooks_cover_the_live_hooks():
    assert {"predict", "s1_plan", "s1_orders", "corp_actions", "u500_symbols"} <= set(
        wiring.production_hooks(None)
    )


# ---- scenario 8 (PLAN-G2 card G2): evening plan -> 15:10 entries -> 15:22 fills -> a horizon-aged book --


class BarHub:
    """Quotes from the synthetic market's own daily bars (ltp = close of the clock's date)."""

    def __init__(self, market, clock):
        self.m = {(r["date"], r["symbol"]): r for r in market.eq.filter(pl.col("series") == "EQ").to_dicts()}
        self.clock = clock

    def get(self, symbols, held=None):
        d, out = self.clock.now().date(), {}
        for s in symbols:
            b = self.m.get((d, s))
            if b is None:
                continue
            out[s] = Quote(
                symbol=s, ts=self.clock.now(), ltp=round(b["close"] / 0.05) * 0.05, open=b["open"],
                high=b["high"], low=b["low"], prev_close=b["prev_close"], volume=1e5, bid=b["close"] - 0.05,
                ask=b["close"] + 0.05, bid_qty=500, ask_qty=500, upper_circuit=b["prev_close"] * 1.2,
                lower_circuit=b["prev_close"] * 0.8, source="bars",
            )  # fmt: skip
        return out


def test_scenario_plan_entries_fills_then_horizon_plan(env, monkeypatch):
    from stocksense.live.jobs import JOBS

    monkeypatch.setattr("stocksense.portfolio.risk.UNIVERSE_MIN", 50)
    monkeypatch.setattr("stocksense.portfolio.risk.MODEL_MAX_AGE_DAYS", 400)  # the fit is dated 2020-03-20
    cal = env.w.calendar
    d0 = cal.shift(env.last, -12)
    d1 = cal.next_session(d0)
    dx = cal.shift(d1, 9)
    clock = SteppingClock(datetime(d0.year, d0.month, d0.day, 18, 40, tzinfo=IST))
    ctx = JobContext(
        env.w.settings, clock, env.st, cal, hub=BarHub(env.w.market, clock),
        notifier=SimpleNamespace(send=lambda t: True), hooks=dict(wiring.production_hooks(env.w.settings)),
    )  # fmt: skip

    def cap_read(d):
        for k, v in {"capital_ref_inr": repr(CAP), "capital_stale": "0", "capital_date": d.isoformat(),
                     "capital_read_at": clock.now().isoformat()}.items():  # fmt: skip
            env.st.kv_set(k, v)

    try:
        # day 0 evening: predictions and a plan (nothing held yet)
        cap_read(d0)
        assert wiring.stage_predict(ctx, d0).startswith("s1:")
        out = wiring.hook_s1_plan(ctx, d0)
        assert out.startswith("exits=0 planned_entries=")
        n_entries = int(out.rsplit("=", 1)[1])
        assert n_entries > 0
        # day 1 15:10: entries from that plan; 15:22: the paper broker fills them at the close
        clock._dt = datetime(d1.year, d1.month, d1.day, 15, 10, tzinfo=IST)
        cap_read(d1)
        env.st.kv_set("free_cash_frac:main", "1.0")
        assert JOBS["s1_entries"](ctx, d1).startswith("orders=")
        buys = ctx.ledger.pending_orders(d1, "close", books=["main"])
        assert 0 < len(buys) <= n_entries and all(o.side == "BUY" and o.product == "delivery" for o in buys)
        clock._dt = datetime(d1.year, d1.month, d1.day, 15, 22, tzinfo=IST)
        res = ctx.paper().process_close(d1)
        assert len(res.filled) == len(buys), res
        held = {p.sid for p in ctx.ledger.positions("main")}
        assert len(held) == len(buys)
        # nine sessions later the held names are at the horizon: each is exited or extended
        clock._dt = datetime(dx.year, dx.month, dx.day, 18, 40, tzinfo=IST)
        cap_read(dx)
        wiring.stage_predict(ctx, dx)
        out = wiring.hook_s1_plan(ctx, dx)
        plan = json.loads(env.st.kv_get(f"s1_plan:{dx.isoformat()}"))
        exited = {int(s) for s in plan["exits"]}
        assert held <= exited | set(plan["extended"]), (held, plan["exits"], plan["extended"])
        sells = ctx.ledger.pending_orders(cal.next_session(dx), "open", books=["main"])
        assert {o.sid for o in sells} == exited and all(o.side == "SELL" for o in sells)
    finally:
        con = env.st.connect()
        try:
            for t in ("decisions", "orders", "fills", "fees", "cash_events", "marks", "mark_prices"):
                con.execute(f"DELETE FROM {t}")
        finally:
            con.close()
        wiring._ACCESS.clear()


def test_stored_plan_is_replanned_when_the_policy_or_capital_changed(env, lctx):
    d = env.w.calendar.next_session(env.last)
    _stored_plan(env, CAP)
    p = json.loads(env.st.kv_get(f"s1_plan:{env.last.isoformat()}"))
    p["policy"] = wiring.live_policy(env.st)
    env.st.kv_set(f"s1_plan:{env.last.isoformat()}", json.dumps(p))
    lctx._hub = FakeHub({})
    [(_, _o, payload)] = wiring.hook_s1_orders(lctx, d, CAP)
    assert payload["replanned"] == ""  # same capital, same policy: the stored plan stands
    env.st.kv_set("live_policy:s1", json.dumps({"er_map": "decile"}))
    try:
        [(_, _o, payload)] = wiring.hook_s1_orders(lctx, d, CAP)
        assert payload["replanned"] == "policy changed"
        assert json.loads(env.st.kv_get(f"s1_plan:{env.last.isoformat()}"))["policy"]["er_map"] == "decile"
        [(_, _o, payload)] = wiring.hook_s1_orders(lctx, d, CAP * 2)
        assert payload["replanned"] == "capital moved"
    finally:
        con = env.st.connect()
        try:
            con.execute("DELETE FROM kv WHERE key='live_policy:s1'")
        finally:
            con.close()

# ---- G2-G3: S2 measurement books -----------------------------------------------------------------------

S2_SPEC = ModelSpec(family="s2-ridge", target="y_oc", universe="u200", entry="open")


@pytest.fixture(scope="module")
def s2env(env):
    w = env.w
    res = run_walk_forward(
        S2_SPEC, w.settings, env.st, folds=w.folds, calendar=w.calendar, registry=w.registry, seeds=(1, 2),
        inner_months=2, trial_family="g2-test-s2",
    )  # fmt: skip
    as_of = w.calendar.prev_session(LOCKBOX_START)
    mid = fit_production(
        S2_SPEC, w.settings, env.st, as_of, seeds=(1, 2), calendar=w.calendar, inner_months=2,
        allow_beyond_research=True,
    )  # fmt: skip
    env.st.kv_set("live_model:s2", mid)
    env.st.kv_set("live_wf_trial:s2", res.trial_id)
    return env


def _preopen(env, d, n=12):
    uni = io.read(env.w.settings, "panel_universe", start=env.last, end=env.last).collect()
    sids = uni.filter(pl.col("u200").fill_null(False)).sort("sid")["sid"].head(n).to_list()
    daily = io.read(
        env.w.settings, "panel_daily", start=env.last, end=env.last, columns=["sid", "symbol", "close"]
    )
    cl = {int(r["sid"]): r for r in daily.collect().to_dicts()}
    ts = datetime(d.year, d.month, d.day, 9, 7, tzinfo=IST)
    rows = [
        {"symbol": cl[x]["symbol"], "ts": ts, "prev_close": cl[x]["close"], "indicative_price": cl[x]["close"] * 1.01,  # noqa: E501
         "final_price": None, "buy_qty": 100.0, "sell_qty": 50.0, "source": "test"}
        for x in sids
    ]  # fmt: skip
    return pl.DataFrame(rows, schema=io_schema("preopen"))


def io_schema(name):
    from stocksense.data.ingest_misc import _PRE_SCHEMA

    return _PRE_SCHEMA


def test_s2_orders_go_to_both_measurement_books_identically(s2env, lctx, monkeypatch):
    env = s2env
    d = env.w.calendar.next_session(env.last)
    io.upsert(env.w.settings, "preopen", _preopen(env, d), clock=env.ctx.clock)
    out = wiring.hook_s2_orders(lctx, d, CAP)
    assert [b for b, _o, _p in out] == ["auction", "auction_strict"]
    (_, o1, p1), (_, o2, p2) = out
    assert o1 == o2 and p1["note"].startswith("measurement only")
    assert len(o1) > 0, p1
    assert all(o.product == "intraday" and o.when_kind == "open" and o.trade_date == d for o in o1)
    assert all(o.symbol and o.qty > 0 and o.ref_price > 0 for o in o1)


def test_s2_no_orders_when_preopen_is_empty(s2env, lctx):
    env = s2env
    d = env.w.calendar.next_session(env.w.calendar.next_session(env.last))  # a day with no capture
    out = wiring.hook_s2_orders(lctx, d, CAP)
    assert all(o == [] for _b, o, _p in out) and len(out) == 2


def test_production_hooks_include_s2():
    assert "s2_orders" in wiring.production_hooks(None)


