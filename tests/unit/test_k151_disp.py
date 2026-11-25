"""M5.0b: k151_dispersion (151 Trading Strategies s6.3): constituent choice, the book and the registration."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.features.derivs import R_FREE, bs_price
from stocksense.strategies import base
from stocksense.strategies.k151_disp import (
    LONG_STRADDLE,
    N_STAR,
    SHORT_STRADDLE,
    dispersion_daily,
    lowest_specific_variance,
    pick_constituents,
)
from stocksense.strategies.options_bt import EntryRule, run_combo

E1, E2, E3 = date(2026, 1, 29), date(2026, 2, 26), date(2026, 3, 26)
STRIKES = np.arange(60.0, 160.0, 1.0)


def _fo(sym: str, scale: float, trend: float) -> pl.DataFrame:
    rows, d, days = [], date(2026, 1, 20), []
    while d <= E3:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    for d in days:
        s = scale * (1.0 + trend * (d - date(2026, 1, 1)).days)
        for e in (E1, E2, E3, date(2026, 4, 30)):
            if e < d:
                continue
            t = (e - d).days / 365
            rows.append((sym, "IDF", e, 0.0, "", d, s, s))
            for k in STRIKES * scale / 100.0:
                for typ in ("CE", "PE"):
                    px = bs_price(s, float(k), t, R_FREE, 0.2, typ == "CE") if e > d else 9999.0
                    rows.append((sym, "IDO", e, float(k), typ, d, max(px, 0.05), s))
    cols = ["symbol", "instrument_type", "expiry_date", "strike_price", "option_type", "date", "settle_price"]
    return pl.DataFrame(rows, schema=[*cols, "underlying_price"], orient="row")


def test_lowest_specific_variance_drops_the_idiosyncratic_names():
    rng = np.random.default_rng(3)
    t, n = 252, 30
    f = rng.normal(0, 0.01, t)
    x = 1.0 * f[None, :] + rng.normal(0, 0.002, (n, t))
    x[:5] += rng.normal(0, 0.03, (5, t))  # five names with a large own-noise variance
    sel = lowest_specific_variance(x, np.full(n, 1 / n), 25)
    assert len(sel) == 25 and list(sel) == sorted(sel)
    assert not set(range(5)) & set(sel.tolist())


def test_pick_constituents_uses_only_the_previous_close():
    rng = np.random.default_rng(5)
    t, n = 300, 60
    dates = np.datetime64("2025-01-01") + np.arange(t)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (t, n)), axis=0))
    syms = np.array([f"S{i:02d}" for i in range(n)], dtype=object)
    med = np.tile(np.arange(n, 0, -1, dtype=float), (t, 1))  # S00 most traded
    fno = np.ones((t, n), bool)
    entry = date.fromisoformat(str(dates[280]))
    got = pick_constituents(close, dates, syms, med, fno, entry)
    assert len(got) == N_STAR and all(s in set(syms[:50]) for s in got)
    cut = close.copy()
    cut[280:] = np.nan  # nothing from the entry day on may matter
    assert pick_constituents(cut, dates, syms, med, fno, entry) == got
    assert pick_constituents(close, dates, syms, med, fno, date(2025, 3, 1)) == []  # too little history


def test_dispersion_book_is_index_short_plus_mean_stock_long():
    idx = run_combo(SHORT_STRADDLE, EntryRule(), None, "NIFTY", fo=_fo("NIFTY", 100.0, 0.0), lot=65.0)
    a = run_combo(LONG_STRADDLE, EntryRule(), None, "AAA", fo=_fo("AAA", 100.0, 0.002), lot=65.0)
    b = run_combo(LONG_STRADDLE, EntryRule(), None, "BBB", fo=_fo("BBB", 100.0, -0.002), lot=65.0)
    entry = idx.cycles["entry_date"][0]
    out = dispersion_daily(idx, {"AAA": a, "BBB": b}, {entry: ["AAA", "BBB"]})
    c = idx.cycles.row(0, named=True)
    win = out.filter((pl.col("date") >= c["entry_date"]) & (pl.col("date") <= c["exit_date"]))
    want = (
        idx.daily.filter(pl.col("date").is_between(c["entry_date"], c["exit_date"]))["ret"].to_numpy()
        + (
            a.daily.filter(pl.col("date").is_between(c["entry_date"], c["exit_date"]))["ret"].to_numpy()
            + b.daily.filter(pl.col("date").is_between(c["entry_date"], c["exit_date"]))["ret"].to_numpy()
        )
        / 2
    )
    assert win["ret"].to_numpy() == pytest.approx(want)
    assert win["cost"].sum() > idx.cycles["cost"][0]  # the stock straddles add their own charges
    other = out.filter(pl.col("date") > c["exit_date"])  # the second cycle has no picks: flat
    assert other["ret"].abs().sum() == 0.0


def test_registered_as_an_option_family_trial_with_a_cost_hook():
    spec = base.load_library()["k151_dispersion"]
    assert (spec.family, spec.kind, spec.status) == ("m-opt", "options", "ready")
    assert spec.timing is not None and callable(spec.timing.with_cost)  # type: ignore[attr-defined]
