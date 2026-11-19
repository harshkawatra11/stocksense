"""Production wiring of the daemon (Program G2, phase G): live scores, the S1 plan and entries, learning.

Everything here is paper only: orders are rows in the paper ledger, never broker calls. The hooks built by
`production_hooks` are what `live.jobs` calls through `ctx.hook(...)`; tests inject their own.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date, datetime, time

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.core.state import State
from stocksense.costs.india import round_trip_cost_frac
from stocksense.data import io
from stocksense.features.build import load_features
from stocksense.labels.forward import load_labels
from stocksense.learn.nightly import load_bag
from stocksense.models import store
from stocksense.models.base import ModelSpec, build_model
from stocksense.models.ensemble import expected_return_map, expected_return_map_decile, rank_scores
from stocksense.paper.broker import CorpAction
from stocksense.paper.ledger import NewOrder
from stocksense.portfolio.construct import (
    BUY_CUSHION_INR,
    DayScores,
    EnginePanelAccess,
    Holding,
    S1Book,
    S1Config,
    S1Plan,
    plan_s1,
)
from stocksense.portfolio.risk import RiskContext, entries_allowed, pre_trade_checks
from stocksense.validation.walkforward import END_COLUMN

log = get_logger(__name__)

S1_TARGET = "y_c2o_10"
DEFAULT_LIVE = {
    "live_model:s1": "s1-gbm-y_c2o_10-120055de-20250930",
    "live_model:s2": "s2-gbm-y_oc-4fd8ff9b-20250930",
}
ER_WINDOW_DATES = 520  # walk-forward dates kept for the expected-return map (its window is 504)


# ---- which model is live ------------------------------------------------------------------------------


def live_ids(state: State) -> dict[str, str | None]:
    """The kv ids of the live models and the walk-forward trial that calibrates S1's expected-return map.

    Missing keys are initialised with the production defaults (the S1 trial is looked up among the finished
    `s1-gbm` walk-forward trials of `y_c2o_10`; None when there is none)."""
    out: dict[str, str | None] = {}
    for key, default in DEFAULT_LIVE.items():
        cur = state.kv_get(key)
        if cur is None:
            state.kv_set(key, default)
            cur = default
        out[key] = cur
    wf = state.kv_get("live_wf_trial:s1")
    if wf is None:
        from stocksense.validation.g1 import surviving_models

        cands = surviving_models(state, "s1", S1_TARGET)
        gbm = [c for c in cands if c.family == "s1-gbm"] or cands
        if gbm:
            wf = gbm[0].trial_id
            state.kv_set("live_wf_trial:s1", wf)
    out["live_wf_trial:s1"] = wf
    wf2 = state.kv_get("live_wf_trial:s2")
    if wf2 is None:
        from stocksense.validation.g1 import surviving_models

        c2 = surviving_models(state, "s2", "y_oc")
        g2 = [c for c in c2 if c.family == "s2-gbm"] or c2
        if g2:
            wf2 = g2[0].trial_id
            state.kv_set("live_wf_trial:s2", wf2)
    out["live_wf_trial:s2"] = wf2
    return out


def live_policy(state: State) -> dict:
    """The live S1 policy (kv JSON `live_policy:s1`): k_max, overlay, S1Config overrides under `policy`,
    optional `er_map` ('linear' | 'decile'). Default: the g1-s1-y_c2o_10-cce12fb757 spec (k_max 10, no
    overlay) with every other option at its default."""
    raw = state.kv_get("live_policy:s1")
    base = {"k_max": 10, "overlay": False, "policy": {}, "er_map": "linear"}
    if raw:
        base.update(json.loads(raw))
    return base


# ---- the predict stage --------------------------------------------------------------------------------


def predict_frame(settings: Settings, model_id: str, d: date) -> pl.DataFrame:
    """date, sid, score of the production model `model_id` for the u500 members of d (one cross-section)."""
    mdir = store.model_dir(settings, model_id)
    names = list(json.loads((mdir / "feature_manifest.json").read_text(encoding="utf-8"))["features"])
    spec = ModelSpec.from_dict(json.loads((mdir / "spec.json").read_text(encoding="utf-8")))
    uni = io.read(settings, "panel_universe", start=d, end=d, columns=["date", "sid", "u500"]).collect()
    uni = uni.filter(pl.col("u500").fill_null(False))
    if uni.is_empty():
        raise DataNotAvailable(f"no u500 members for {d}")
    feats = load_features(settings, "production", start=d, end=d, columns=names).collect()
    if feats.is_empty():
        raise DataNotAvailable(f"no features for {d}")
    f = uni.select("date", "sid").join(feats, on=["date", "sid"], how="inner").sort("sid")
    if f.is_empty():
        raise DataNotAvailable(f"no feature rows for the u500 members of {d}")
    X = f.select([pl.col(n).cast(pl.Float32) for n in names]).to_numpy().astype(np.float32)
    if build_model(spec).needs_keys:  # sequence / cube models look rows up by (sid, day)
        from stocksense.models.seqnet import add_keys

        X = add_keys(X, f["sid"].to_numpy(), f["date"].to_numpy())
    score = load_bag(settings, model_id).predict(X)
    return pl.DataFrame({"date": f["date"], "sid": f["sid"], "score": score})


def stage_predict(ctx, d: date) -> str:  # type: ignore[no-untyped-def]
    ids = live_ids(ctx.state)
    mid = str(ids["live_model:s1"])
    frame = predict_frame(ctx.settings, mid, d)
    store.append_preds(ctx.settings, mid, frame, "live", clock=ctx.clock)
    return f"s1:{frame.height} names"


# ---- the live score source ----------------------------------------------------------------------------


class LiveScoreSource:
    """`ScoreSource` over the live predictions of `live_model:<strategy>`.

    score: the day's live score ranked to [-0.5, 0.5]. er: `max(0, b) * score` with b the through-the-origin
    slope of the raw label on the score over the trailing 504 dates whose label ended before d, computed on
    the union of the calibration walk-forward trial's stored predictions (ranked per date) and the live
    predictions; with `er_map == 'decile'` the isotonic decile map is used instead. Cached per date."""

    def __init__(
        self, settings: Settings, state: State, strategy: str = "s1", target: str = S1_TARGET,
        er_map: str | None = None,
    ) -> None:  # fmt: skip
        self.settings, self.state, self.strategy, self.target = settings, state, strategy, target
        self.er_map = er_map
        self._cache: dict[date, DayScores | None] = {}

    def _frames(self, d: date) -> tuple[pl.DataFrame, pl.DataFrame]:
        ids = live_ids(self.state)
        mid = ids[f"live_model:{self.strategy}"]
        live = (
            io.read(self.settings, "preds", end=d, columns=["date", "sid", "score"], sub=mid)
            .collect()
            .sort("date", "sid")
        )
        wf_id = ids.get(f"live_wf_trial:{self.strategy}")
        if wf_id:
            wf = io.read(self.settings, "preds", columns=["date", "sid", "score"], sub=wf_id).collect()
            keep = wf["date"].unique().sort().tail(ER_WINDOW_DATES)
            wf = wf.filter(pl.col("date").is_in(keep.implode()) & (pl.col("date") <= d))
            wf = wf.filter(~pl.col("date").is_in(live["date"].unique().implode()))
        else:
            wf = live.clear()
        return live, wf

    def scores(self, d: date) -> DayScores | None:
        if d in self._cache:
            return self._cache[d]
        try:
            live, wf = self._frames(d)
        except (DataNotAvailable, FileNotFoundError, pl.exceptions.PolarsError):
            self._cache[d] = None
            return None
        day = live.filter(pl.col("date") == d)
        if day.is_empty():
            self._cache[d] = None
            return None
        blend = rank_scores(
            pl.concat([wf.select("date", "sid", "score"), live.select("date", "sid", "score")])
        )
        lab = load_labels(
            self.settings,
            "production",
            start=blend["date"].min(),
            columns=[self.target, END_COLUMN[self.target]],
        ).collect()
        er_map = self.er_map or live_policy(self.state).get("er_map", "linear")
        if er_map == "decile":
            er = expected_return_map_decile(blend, lab, self.target)
        else:
            er = expected_return_map(blend, lab, self.target)
        er = er.filter(pl.col("date") == d)
        sids = er["sid"].to_list()
        out = DayScores(
            dict(zip(sids, er["score"].to_list(), strict=True)),
            dict(
                zip(sids, [max(0.0, float(x)) for x in er["exp_ret"].fill_null(0.0).to_list()], strict=True)
            ),
        )
        self._cache[d] = out
        return out


# ---- the S1 evening plan ------------------------------------------------------------------------------

BOOK = "main"
PLAN_CAPITAL_TOLERANCE = 0.05  # a stored plan is re-planned at 15:10 when capital moved by more than this
_ACCESS: dict[date, EnginePanelAccess] = {}


def live_panel_access(settings: Settings, calendar, d: date) -> EnginePanelAccess:  # type: ignore[no-untyped-def]
    """The S1 panel access over the 260 sessions up to d with the u500 hysteresis flag (cached per date)."""
    if d not in _ACCESS:
        from stocksense.validation.g1 import build_world

        _ACCESS.clear()
        world = build_world(settings, calendar, start=calendar.shift(d, -260), end=d, cut_events=False)
        _ACCESS[d] = world.s1_access(universe_flag=True)
    return _ACCESS[d]


def _vol_on(access: EnginePanelAccess, t: date, sid: int) -> float:
    sl = access.p.day(t)
    if sl is not None:
        for j in range(*sl):
            if int(access.p.sid[j]) == sid:
                return float(access.p.vol_21[j])
    return 0.02


def ledger_book(ctx, d: date, book: str = BOOK, capital: float | None = None) -> S1Book:  # type: ignore[no-untyped-def]
    """The planner's view of the paper ledger's delivery book as of the close of d."""
    from stocksense.live.jobs import capital_ref

    cal = ctx.calendar
    cap = capital if capital is not None else capital_ref(ctx)[0]
    if cap is None:
        raise DataNotAvailable("capital not read yet")
    pos = [p for p in ctx.ledger.positions(book) if p.product == "delivery" and p.qty > 0]
    con = ctx.state.connect()
    try:
        fills = con.execute(
            "SELECT o.sid, o.side, o.trade_date, f.price FROM fills f JOIN orders o ON o.order_id=f.order_id "
            "WHERE o.book=? AND o.product='delivery' ORDER BY o.trade_date, f.ts, f.fill_id",
            (book,),
        ).fetchall()
    finally:
        con.close()
    buys: dict[int, tuple[date, float]] = {}
    sells: dict[int, date] = {}
    for sid, side, td, price in fills:
        if side == "BUY":
            buys[int(sid)] = (date.fromisoformat(td), float(price))
        else:
            sells[int(sid)] = date.fromisoformat(td)
    idx_d = cal.session_index(d)
    access = live_panel_access(ctx.settings, cal, d) if pos else None
    holdings: dict[int, Holding] = {}
    for p in pos:
        ed, px = buys.get(p.sid, (d, p.last_price))
        vol = _vol_on(access, ed, p.sid) if access is not None else 0.02
        held = max(idx_d - cal.session_index(ed), 0) if cal.is_session(ed) else 0
        holdings[p.sid] = Holding(p.sid, int(round(p.qty)), px, held, vol if vol > 0 else 0.02)
    nav = ctx.ledger.nav_series(book)
    dd = 0.0
    if nav.height:
        peak = float(nav["nav"].max())
        dd = max(0.0, 1.0 - float(nav["nav"][-1]) / peak) if peak > 0 else 0.0
    recent = {
        s: idx_d - cal.session_index(td)
        for s, td in sells.items()
        if cal.is_session(td) and 0 <= idx_d - cal.session_index(td) <= 10
    }
    stopped = ctx.state.kv_get(f"book_stop:{book}") == "1"
    return S1Book(holdings, cap * ctx.ledger.cash_frac(book), dd, stopped, recent)


def _symbols(settings: Settings, d: date) -> dict[int, str]:
    rows = io.read(settings, "panel_daily", start=d, end=d, columns=["sid", "symbol"]).collect()
    return dict(zip(rows["sid"].to_list(), rows["symbol"].to_list(), strict=True))


def risk_context(ctx, d: date) -> RiskContext:  # type: ignore[no-untyped-def]
    """The inputs of the eight pre-trade checks as the evening of d sees them (see DECISIONS, G2-G2)."""
    s, cal, now = ctx.settings, ctx.calendar, ctx.clock.now()
    lo = cal.shift(d, -9)
    present = set(io.read(s, "panel_daily", start=lo, end=d, columns=["date"]).collect()["date"].to_list())
    missing = [x for x in cal.sessions(lo, d) if x not in present]
    feats = load_features(s, "production", start=lo, end=d, columns=[]).collect()
    f_last = feats["date"].max() if feats.height else None
    uni = io.read(s, "panel_universe", start=d, end=d, columns=["sid", "u500", "turnover_rank"]).collect()
    # the u500 hysteresis flag holds ~560 names (members stay until rank 600): the check is about a collapsed
    # or exploded universe, so it counts the flagged names within the nominal 500 (dry run 2026-11-19: 562)
    n_u500 = int((uni["u500"].fill_null(False) & (uni["turnover_rank"].fill_null(10**9) <= 500)).sum())
    mid = live_ids(ctx.state)["live_model:s1"]
    con = ctx.state.connect()
    try:
        row = con.execute("SELECT created_at FROM models WHERE model_id=?", (mid,)).fetchone()
    finally:
        con.close()
    fits = [datetime.fromisoformat(row[0])] if row and row[0] else []
    std_today = std_prev = null_frac = None
    live = io.read(s, "preds", end=d, columns=["date", "sid", "score"], sub=mid).collect()
    if live.height:
        stds = live.group_by("date").agg(pl.col("score").std().alias("sd")).sort("date")
        today = stds.filter(pl.col("date") == d)
        if today.height:
            std_today = float(today["sd"][0])
            prev = stds.filter(pl.col("date") < d).tail(60)["sd"].drop_nulls()
            std_prev = float(prev.mean()) if len(prev) else std_today  # no history yet: compare with itself
            day = live.filter(pl.col("date") == d)["score"]
            null_frac = float((~day.is_finite()).mean())  # type: ignore[arg-type]
    read_at = ctx.state.kv_get("capital_read_at")
    cap_at = datetime.fromisoformat(read_at) if read_at else None  # the check compares its date with today
    return RiskContext(
        now=now, last_completed_session=d, panel_last_date=max(present) if present else None,
        features_last_date=f_last, null_share_today={}, null_share_mean_60={}, model_fit_times=fits,
        n_u500=n_u500, score_std_today=std_today, score_std_mean_60=std_prev, score_null_frac=null_frac,
        capital_read_at=cap_at, last_quote_ts=None, market_open=False, missing_sessions_last10=missing,
    )  # fmt: skip


def compute_plan(ctx, d: date, cap: float, *, allow_entries: bool | None = None):  # type: ignore[no-untyped-def]
    """(S1Plan, context dict) for the close of d at capital `cap`; the book is read from the ledger."""
    from stocksense.costs.slippage import SlippageTable

    pol = live_policy(ctx.state)
    sc = LiveScoreSource(ctx.settings, ctx.state, er_map=pol.get("er_map")).scores(d)
    if sc is None:
        raise DataNotAvailable(f"no live scores for {d}")
    access = live_panel_access(ctx.settings, ctx.calendar, d)
    d1 = ctx.calendar.next_session(d)
    book = ledger_book(ctx, d, capital=cap)
    checks = pre_trade_checks(risk_context(ctx, d))
    if book.dd >= ctx.settings.risk.book_dd_stop and not book.stopped:
        ctx.state.kv_set(f"book_stop:{BOOK}", "1")
        book = S1Book(book.holdings, book.cash, book.dd, True, book.recent_sells)
    ok = entries_allowed(checks) if allow_entries is None else allow_entries
    cfg = S1Config.from_settings(
        ctx.settings, 10, SlippageTable.load_latest(ctx.settings), k_max=int(pol["k_max"]),
        overlay=bool(pol["overlay"]), **dict(pol.get("policy") or {}),
    )  # fmt: skip
    panel_t = access.day(d, d1)
    scale = float(ctx.state.kv_get(f"decay_scale:{BOOK}") or 1.0)
    plan = plan_s1(d, sc.score, sc.er, book, cap, panel_t, cfg, allow_entries=ok, gross_scale=scale)
    info = {
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
        "allow_entries": ok, "scores": sc, "panel": panel_t, "book": book,
    }  # fmt: skip
    return plan, info


def _plan_json(ctx, d: date, cap: float, stale: bool, plan: S1Plan, info: dict) -> dict:  # type: ignore[no-untyped-def]
    sc, pday = info["scores"], info["panel"]
    sym = _symbols(ctx.settings, d)
    ranked = sorted(sc.score, key=lambda x: (-sc.score[x], x))
    rank = {x: i + 1 for i, x in enumerate(ranked)}
    sells = [
        {"sid": o.sid, "symbol": sym.get(o.sid, str(o.sid)), "qty": int(o.qty), "reason": o.reason,
         "ref": pday.rows[o.sid].close if o.sid in pday.rows else None}
        for o in plan.orders if o.side == "SELL"
    ]  # fmt: skip
    entries = [
        {"sid": x, "symbol": sym.get(x, str(x)), "weight": w, "score": sc.score.get(x), "rank": rank.get(x),
         "er": sc.er.get(x), "ref_close": pday.rows[x].close}
        for x, w in sorted(plan.weights.items(), key=lambda kv: -sc.score.get(kv[0], 0.0))
    ]  # fmt: skip
    return {
        "date": d.isoformat(), "next_date": ctx.calendar.next_session(d).isoformat(), "cap_ref": cap,
        "stale": stale, "model": live_ids(ctx.state)["live_model:s1"], "note": plan.note, "k": plan.k,
        "k_raw": plan.k_raw, "gross_target": plan.gross_target, "allow_entries": info["allow_entries"],
        "checks": info["checks"], "exits": {str(k): v for k, v in plan.exits.items()},
        "extended": list(plan.extended), "sells": sells, "entries": entries,
        "policy": live_policy(ctx.state),
    }  # fmt: skip


def hook_s1_plan(ctx, d: date) -> str:  # type: ignore[no-untyped-def]
    """Evening: plan S1 for the next session, store it (kv s1_plan:<d>), record the exits as SELL open
    orders of the next session."""
    from stocksense.live.jobs import capital_ref

    cap, stale = capital_ref(ctx)
    if cap is None:
        return "skipped: no capital reference yet (entries are planned at 15:10)"
    plan, info = compute_plan(ctx, d, cap)
    body = _plan_json(ctx, d, cap, stale, plan, info)
    ctx.state.kv_set(f"s1_plan:{d.isoformat()}", json.dumps(body))
    d1 = ctx.calendar.next_session(d)
    sells = [
        NewOrder(
            x["sid"], x["symbol"], "SELL", "delivery", "open", d1, x["qty"], x["ref"] or 0.0, x["reason"]
        )
        for x in body["sells"]
    ]
    if sells:
        dec = ctx.ledger.record_decision(BOOK, "s1-exit", d1, cap, stale, {"checks": body["checks"]}, body)
        ctx.ledger.add_orders(dec, BOOK, sells)
    return f"exits={len(sells)} planned_entries={len(body['entries'])}"


def hook_s1_orders(ctx, d: date, cap: float):  # type: ignore[no-untyped-def]
    """15:10: today's BUY close orders from yesterday's plan, re-sized to today's capital and quotes."""
    from stocksense.live.jobs import capital_ref

    prev = ctx.calendar.prev_session(d)
    raw = ctx.state.kv_get(f"s1_plan:{prev.isoformat()}")
    plan = json.loads(raw) if raw else None
    replanned = ""
    if plan is None:
        replanned = "no stored plan"
    elif not plan.get("cap_ref") or abs(cap / plan["cap_ref"] - 1.0) > PLAN_CAPITAL_TOLERANCE:
        replanned = "capital moved"
    elif plan.get("policy") != live_policy(ctx.state):
        replanned = "policy changed"
    if replanned:
        p, info = compute_plan(ctx, prev, cap)
        plan = _plan_json(ctx, prev, cap, capital_ref(ctx)[1], p, info)
        ctx.state.kv_set(f"s1_plan:{prev.isoformat()}", json.dumps(plan))
    entries = plan["entries"] if plan["allow_entries"] else []
    quotes = ctx.hub.get([e["symbol"] for e in entries]) if entries else {}
    free = ctx.state.kv_get(f"free_cash_frac:{BOOK}")
    budget = cap * (float(free) if free is not None else ctx.ledger.cash_frac(BOOK))
    orders: list[NewOrder] = []
    detail: list[dict] = []
    dropped: list[str] = []
    for e in entries:  # already in descending score order
        q = quotes.get(e["symbol"])
        ltp = float(q.ltp) if q is not None and q.ltp else float(e["ref_close"])
        if q is not None and q.upper_circuit and ltp >= float(q.upper_circuit) - 0.05:
            dropped.append(f"{e['symbol']}:upper_circuit")
            continue
        want = int(math.floor(e["weight"] * cap / ltp + 1e-9))
        qty = min(want, int(math.floor(max(budget - BUY_CUSHION_INR, 0.0) / ltp)))
        if qty <= 0:
            dropped.append(f"{e['symbol']}:no_cash_or_qty")
            continue
        budget -= qty * ltp + BUY_CUSHION_INR
        orders.append(NewOrder(e["sid"], e["symbol"], "BUY", "delivery", "close", d, qty, ltp, "entry"))
        cost = round_trip_cost_frac("delivery", qty * ltp, ctx.settings.costs)
        detail.append({**e, "qty": qty, "ref_price": ltp, "cost_frac": cost})
    payload = {
        "plan_date": prev.isoformat(), "note": plan["note"], "replanned": replanned, "dropped": dropped,
        "allow_entries": plan["allow_entries"], "checks": plan["checks"], "orders": detail,
    }  # fmt: skip
    return [(BOOK, orders, payload)]


def corp_actions_hook(ctx, d: date) -> list[CorpAction]:  # type: ignore[no-untyped-def]
    """Ex-date `d` actions of the names held in any book (price factor and dividend per share)."""
    held = {p.symbol: p.sid for b in ctx.ledger.books() for p in ctx.ledger.positions(b)}
    if not held:
        return []
    ca = io.read(
        ctx.settings, "corporate_actions", columns=["symbol", "ex_date", "factor_price", "dividend_amount"]
    ).collect()  # fmt: skip
    ca = ca.filter((pl.col("ex_date") == d) & pl.col("symbol").is_in(list(held)))
    return [
        CorpAction(held[r["symbol"]], float(r["factor_price"] or 1.0), float(r["dividend_amount"] or 0.0))
        for r in ca.to_dicts()
    ]


# ---- S2 measurement books -----------------------------------------------------------------------------

S2_NOTE = "measurement only: S2 failed strict capture (RESULTS-G1)"
S2_BOOKS = ("auction", "auction_strict")
PREOPEN_END = time(9, 8)
_S2_ACCESS: dict[date, object] = {}


def _s2_access(settings: Settings, calendar, prev: date):  # type: ignore[no-untyped-def]
    if prev not in _S2_ACCESS:
        from stocksense.validation.g1 import build_world

        _S2_ACCESS.clear()
        world = build_world(settings, calendar, start=calendar.shift(prev, -60), end=prev, cut_events=False)
        _S2_ACCESS[prev] = world.s2_access()
    return _S2_ACCESS[prev]


def s2_score_frame(settings: Settings, model_id: str, prev: date) -> pl.DataFrame:
    """date, sid, score of the S2 production model on the u200 members of the decision date `prev`."""
    mdir = store.model_dir(settings, model_id)
    names = list(json.loads((mdir / "feature_manifest.json").read_text(encoding="utf-8"))["features"])
    uni = io.read(settings, "panel_universe", start=prev, end=prev, columns=["date", "sid", "u200"]).collect()
    uni = uni.filter(pl.col("u200").fill_null(False))
    feats = load_features(settings, "production", start=prev, end=prev, columns=names).collect()
    f = uni.select("date", "sid").join(feats, on=["date", "sid"], how="inner").sort("sid")
    if f.is_empty():
        raise DataNotAvailable(f"no S2 feature rows for {prev}")
    X = f.select([pl.col(n).cast(pl.Float32) for n in names]).to_numpy().astype(np.float32)
    return pl.DataFrame(
        {"date": f["date"], "sid": f["sid"], "score": load_bag(settings, model_id).predict(X)}
    )


def hook_s2_orders(ctx, d: date, cap: float):  # type: ignore[no-untyped-def]
    """09:07: S2 orders for the two measurement books (identical orders; never the main book)."""
    from stocksense.costs.slippage import SlippageTable
    from stocksense.portfolio.construct import AuctionRow, S2Config, plan_s2

    s = ctx.settings
    pre = io.read(s, "preopen", start=d, end=d).collect()
    if pre.height:
        pre = pre.filter((pl.col("ts").dt.date() == d) & (pl.col("ts").dt.time() <= PREOPEN_END))
    pre = pre.drop_nulls("indicative_price")
    if pre.is_empty():
        return [(b, [], {"note": S2_NOTE, "why": "no pre-open capture"}) for b in S2_BOOKS]
    prev = ctx.calendar.prev_session(d)
    ids = live_ids(ctx.state)
    mid = str(ids["live_model:s2"])
    frame = s2_score_frame(s, mid, prev)
    store.append_preds(s, mid, frame, "live", clock=ctx.clock)
    sc = LiveScoreSource(s, ctx.state, "s2", "y_oc", er_map="linear").scores(prev)
    if sc is None:
        raise DataNotAvailable(f"no S2 scores for {prev}")
    rows = _s2_access(s, ctx.calendar, prev).day(prev, d).rows  # type: ignore[attr-defined]
    sym = _symbols(s, prev)
    px = {
        r["symbol"]: float(r["indicative_price"])
        for r in pre.sort("ts").group_by("symbol", maintain_order=True).last().to_dicts()
    }
    auction = {}
    for sid, row in rows.items():
        p = px.get(sym.get(sid, ""))
        if p is None or p <= 0 or sid not in sc.score or row.close <= 0:
            continue
        auction[sid] = AuctionRow(
            p, p / row.close - 1.0, row.turnover_rank, row.med_turn_20, row.vol_21, row.in_fno, None
        )
    free = ctx.state.kv_get("free_cash_frac:auction")
    free_cash = cap * (float(free) if free is not None else ctx.ledger.cash_frac("auction"))
    cfg = S2Config.from_settings(s, SlippageTable.load_latest(s))
    plan = plan_s2(d, sc.score, sc.er, auction, free_cash, cap, cfg)
    orders = [
        NewOrder(
            o.sid,
            sym.get(o.sid, str(o.sid)),
            o.side,
            "intraday",
            "open",
            d,
            int(o.qty),
            auction[o.sid].open,
            o.reason,
        )
        for o in plan.orders
    ]
    payload = {"note": S2_NOTE, "plan_note": plan.note, "n_candidates": len(auction),
               "orders": [{"sid": o.sid, "symbol": o.symbol, "side": o.side, "qty": o.qty, "ref": o.ref_price,
                           "er": sc.er.get(o.sid)} for o in orders]}  # fmt: skip
    return [(b, orders, payload) for b in S2_BOOKS]


def production_hooks(settings: Settings) -> Mapping[str, object]:
    """The hooks `build_context` installs when none are injected."""
    from stocksense.live.jobs import u500_symbols

    _ = u500_symbols
    return {
        "predict": stage_predict,
        "s1_plan": hook_s1_plan,
        "s1_orders": hook_s1_orders,
        "s2_orders": hook_s2_orders,
        "corp_actions": corp_actions_hook,
        "u500_symbols": u500_symbols,
    }
