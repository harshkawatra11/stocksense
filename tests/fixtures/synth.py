"""Synthetic NSE-like market with scripted corporate events (PLAN D.8, T1.9).

Raw prices are UNADJUSTED and tick-rounded; raw `prev_close` is the previous session's raw close of the
same company (as on NSE, PLAN B.3 Finding 1). Returns are fractions. All numpy, vectorised per stock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from stocksense.core.config import Paths, Settings, load_settings
from stocksense.data import io
from stocksense.data.catalog import CATALOG

_REPO = Path(__file__).resolve().parents[2]
_TICK = 0.05
_SATURDAY_AT = 120
_START = date(2018, 1, 1)
_INDUSTRIES = ["Banks", "Software", "Pharma", "Autos"]


@dataclass(frozen=True)
class SynthMarket:
    sessions: list[date]
    eq: pl.DataFrame
    delivery: pl.DataFrame
    corporate_actions: pl.DataFrame
    symbol_changes: pl.DataFrame
    listed: pl.DataFrame
    etf: pl.DataFrame
    sectors: pl.DataFrame
    truth: pl.DataFrame


# ---- helpers ----------------------------------------------------------------------------------


def _make_sessions(n_weekdays: int) -> list[date]:
    """n_weekdays weekdays from 2018-01-01 plus one Saturday session at index 120."""
    days: list[date] = []
    d = _START
    while len(days) < n_weekdays:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    if n_weekdays <= _SATURDAY_AT:
        return days
    # weekday 119 is a Friday (2018-01-01 is a Monday); the Saturday follows it
    sat = days[_SATURDAY_AT - 1] + timedelta(days=1)
    return [*days[:_SATURDAY_AT], sat, *days[_SATURDAY_AT:]]


def _tick(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.round(x / _TICK) * _TICK, _TICK)


def _frame(schema: dict[str, pl.DataType], data: dict[str, list]) -> pl.DataFrame:
    """Build a frame with exactly the catalog columns and dtypes (absent columns become nulls)."""
    n = max((len(v) for v in data.values()), default=0)
    cols = {c: pl.Series(c, data.get(c) or [None] * n, dtype=t) for c, t in schema.items()}
    return pl.DataFrame(cols)


def _isin(symbol: str) -> str:
    return "INE" + "".join(f"{ord(c) % 36:02d}"[:2] for c in symbol)[:6].ljust(6, "0") + "01010"


# ---- the market -------------------------------------------------------------------------------

# (symbol-index) -> scripted behaviour; indices beyond n or n_sessions are skipped.
_SPLITS = {3: (200, 0.2), 14: (220, 0.5)}  # price factor from this session on (14 has no CA row)
_BONUSES = {5: (250, 0.5), 6: (260, 0.5)}  # real price factor 0.5; S06's table factor is wrong
_DEMERGER = (16, 330, 0.7)


def make_market(n: int = 40, n_sessions: int = 600, seed: int = 7, planted_ic: float = 0.0) -> SynthMarket:
    """Synthetic raw NSE market: n stocks over n_sessions weekdays + one Saturday; returns are fractions."""
    sessions = _make_sessions(n_sessions)
    T = len(sessions)
    rng = np.random.default_rng(seed)

    m = rng.normal(0.0004, 0.01, T)
    beta = rng.uniform(0.5, 1.5, n)
    e = rng.normal(0.0, 0.02, (n, T))
    alpha = rng.normal(0.0, 1.0, (n, T))
    gap = rng.normal(0.0015, 0.008, (n, T))
    hi_n = np.abs(rng.normal(0.0, 0.004, (n, T)))
    lo_n = np.abs(rng.normal(0.0, 0.004, (n, T)))
    vol_noise = rng.normal(0.0, 0.5, (n, T))
    pct = np.clip(rng.normal(45.0, 15.0, (n, T)), 5.0, 95.0)
    p0 = np.exp(rng.uniform(np.log(100.0), np.log(2000.0), n))
    med_turnover = rng.permutation(np.logspace(np.log10(0.2e7), np.log10(200e7), n))

    r = beta[:, None] * m[None, :] + e
    if planted_ic:
        r[:, 1:] += planted_ic * 0.02 * alpha[:, :-1]

    def ok(i: int, *idx: int) -> bool:
        return i < n and all(k < T for k in idx)

    symbols = [f"S{i:02d}" for i in range(n)]
    if ok(17):
        symbols[17] = "NIFTYBEES"
    first = np.zeros(n, dtype=int)
    last = np.full(n, T - 1)
    if ok(19, 350):
        first[19] = 350
        p0[19] = 350.0
    if ok(11, 400):
        last[11] = 400
    if ok(21):
        p0[21] = 10.0
    if ok(9, 331):
        r[9, 310:331] = 0.0  # frozen price while suspended, so the path stays continuous
    if ok(15, 453):
        r[15, 450:453] = 0.05  # upper circuit

    eq_parts: list[pl.DataFrame] = []
    dl_parts: list[pl.DataFrame] = []
    truth_parts: list[pl.DataFrame] = []
    dates_arr = np.array(sessions, dtype="datetime64[D]")
    era = "legacy"

    for i in range(n):
        key = symbols[i]
        ret = r[i].copy()
        growth = 1.0 + ret
        growth[: first[i] + 1] = 1.0  # level is p0 on the first listed session
        q = p0[i] * np.cumprod(growth)  # continuous ex-dividend level in start-of-history units
        div_k = 150 if (i == 13 and ok(13, 150)) else None
        if div_k is not None:
            q[div_k:] *= (q[div_k] - 5.0) / q[div_k]
        if i == 21:
            q = q * (18.0 / q.max())
        scale = np.ones(T)
        vscale = np.ones(T)
        if i in _SPLITS and ok(i, _SPLITS[i][0]):
            k, f = _SPLITS[i]
            scale[k:] *= f
            vscale[k:] /= f
        if i in _BONUSES and ok(i, _BONUSES[i][0]):
            k, f = _BONUSES[i]
            scale[k:] *= f
            vscale[k:] /= f
        if i == _DEMERGER[0] and ok(i, _DEMERGER[1]):
            scale[_DEMERGER[1] :] *= _DEMERGER[2]

        close_x = q * scale  # exact raw close
        prev_x = np.concatenate(([close_x[0]], close_x[:-1]))
        open_x = prev_x * (scale / np.concatenate(([scale[0]], scale[:-1]))) * np.exp(gap[i])
        o = _tick(open_x)
        c = _tick(close_x)
        # locked at the upper circuit: o=h=l=c=prev rounded close * 1.05
        locked = np.zeros(T, dtype=bool)
        if i == 15 and ok(15, 453):
            for t in (450, 451, 452):
                c[t] = _tick(np.array([c[t - 1] * 1.05]))[0]
                o[t] = c[t]
                locked[t] = True
        h = np.maximum(_tick(np.maximum(o, c) * (1.0 + hi_n[i])), np.maximum(o, c))
        lo = np.minimum(_tick(np.minimum(o, c) * (1.0 - lo_n[i])), np.minimum(o, c))
        h[locked] = c[locked]
        lo[locked] = c[locked]

        present = np.zeros(T, dtype=bool)
        present[first[i] : last[i] + 1] = True
        if i == 9 and ok(9, 331):
            present[310:331] = False
        idx = np.flatnonzero(present)
        co, cc, ch, cl = o[idx], c[idx], h[idx], lo[idx]
        prev = np.concatenate(([co[0]], cc[:-1]))  # first row: prev_close = first open

        med_vol = med_turnover[i] / p0[i]
        vol = np.maximum(np.round(med_vol * vscale[idx] * np.exp(vol_noise[i, idx])), 1.0)
        vol[locked[idx]] = np.round(vol[locked[idx]] * 0.05) + 1.0
        series = np.full(len(idx), "EQ", dtype=object)
        if i == 23 and ok(23, 420):
            series[(idx >= 380) & (idx < 420)] = "BE"
        sym = np.full(len(idx), key, dtype=object)
        if i == 7 and ok(7, 300):
            sym[idx >= 300] = "S07N"
        if i == 8 and ok(8, 310):
            sym[idx >= 310] = "S08N"

        turnover = vol * (co + ch + cl + cc) / 4.0
        dates = [sessions[k] for k in idx]
        eq_parts.append(
            _frame(
                CATALOG["bhavcopy_eq"].schema,
                {
                    "symbol": sym.tolist(),
                    "series": series.tolist(),
                    "date": dates,
                    "open": co.tolist(),
                    "high": ch.tolist(),
                    "low": cl.tolist(),
                    "close": cc.tolist(),
                    "prev_close": prev.tolist(),
                    "last_price": cc.tolist(),
                    "volume": vol.astype(np.int64).tolist(),
                    "turnover_inr": turnover.tolist(),
                    "n_trades": (vol / 50.0).tolist(),
                    "era": [era] * len(idx),
                    "isin": [_isin(key)] * len(idx),
                    "name": [f"{key} Ltd"] * len(idx),
                    "avg_price": (turnover / vol).tolist(),
                },
            )
        )
        p = pct[i, idx]
        dl_parts.append(
            _frame(
                CATALOG["bhavcopy_delivery"].schema,
                {
                    "symbol": sym.tolist(),
                    "series": series.tolist(),
                    "date": dates,
                    "deliv_qty": (vol * p / 100.0).tolist(),
                    "deliv_pct": p.tolist(),
                },
            )
        )
        tr = ret[idx].astype(float).tolist()
        tr[0] = None  # first listed row has no previous close
        truth_parts.append(
            _frame(
                {"symbol_key": pl.Utf8, "date": pl.Date, "true_ret": pl.Float64, "alpha": pl.Float64},
                {
                    "symbol_key": [key] * len(idx),
                    "date": dates,
                    "true_ret": tr,
                    "alpha": alpha[i, idx].tolist(),
                },
            )
        )
        _ = dates_arr

    eq = pl.concat(eq_parts)
    delivery = pl.concat(dl_parts)
    truth = pl.concat(truth_parts)

    # ---- corporate actions, symbol changes, reference tables -----------------------------------
    ca: list[dict] = []

    def add(sym: str, k: int, typ: str, **kw) -> None:
        row = {
            "symbol": sym,
            "ex_date": sessions[k],
            "action_type": typ,
            "ratio_num": None,
            "ratio_den": None,
            "factor_price": 1.0,
            "dividend_amount": None,
            "face_before": None,
            "face_after": None,
            "subject_raw": typ,
            "parse_status": "ok",
        }
        row.update(kw)
        ca.append(row)

    if ok(3, 200):
        add(
            symbols[3],
            200,
            "split",
            factor_price=0.2,
            face_before=10.0,
            face_after=2.0,
            subject_raw="Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share",
        )
    if ok(5, 250):
        add(symbols[5], 250, "bonus", ratio_num=1.0, ratio_den=1.0, factor_price=0.5, subject_raw="Bonus 1:1")
    if ok(6, 260):
        add(
            symbols[6],
            260,
            "bonus",
            ratio_num=1.0,
            ratio_den=1.0,
            factor_price=0.667,
            subject_raw="Bonus 1:1",
        )
    if ok(13, 150):
        add(
            symbols[13],
            150,
            "dividend",
            dividend_amount=5.0,
            subject_raw="Dividend - Rs 5 Per Share",
        )
    if ok(16, 330):
        add(symbols[16], 330, "demerger", parse_status="unparsed", subject_raw="Demerger")
    if ok(25, 280):
        add(
            symbols[25],
            280,
            "rights",
            parse_status="unparsed",
            subject_raw="Rights 1:4 @ Premium Rs 40/-",
        )
    corporate_actions = _frame(
        CATALOG["corporate_actions"].schema,
        {c: [row.get(c) for row in ca] for c in CATALOG["corporate_actions"].schema},
    )

    sc_rows = []
    if ok(7, 300):
        sc_rows.append(("S07 Ltd", "S07", "S07N", sessions[300]))
    symbol_changes = _frame(
        CATALOG["symbol_changes"].schema,
        {
            "company": [x[0] for x in sc_rows],
            "old_symbol": [x[1] for x in sc_rows],
            "new_symbol": [x[2] for x in sc_rows],
            "change_date": [x[3] for x in sc_rows],
        },
    )

    as_of = sessions[-1]
    cur = [
        (i, ("S07N" if i == 7 and ok(7, 300) else "S08N" if i == 8 and ok(8, 310) else symbols[i]))
        for i in range(n)
    ]
    cur = [(i, s) for i, s in cur if not (i == 11 and ok(11, 400))]  # delisted names are not listed
    face = {i: 10.0 for i, _ in cur}
    if ok(3, 200):
        face[3] = 2.0
    if ok(14, 220):
        face[14] = 5.0
    listed = _frame(
        CATALOG["listed_equities"].schema,
        {
            "symbol": [s for _, s in cur],
            "name": [f"{symbols[i]} Ltd" for i, _ in cur],
            "series": ["EQ"] * len(cur),
            "listing_date": [sessions[int(first[i])] for i, _ in cur],
            "isin": [_isin(symbols[i]) for i, _ in cur],
            "face_value": [face[i] for i, _ in cur],
            "as_of": [as_of] * len(cur),
        },
    )
    etf_syms = [symbols[17]] if ok(17) else []
    etf = _frame(
        CATALOG["etf_list"].schema,
        {
            "symbol": etf_syms,
            "underlying": ["NIFTY 50"] * len(etf_syms),
            "name": ["Nifty BeES"] * len(etf_syms),
            "as_of": [as_of] * len(etf_syms),
        },
    )
    sec = [(i, s) for i, s in cur if not (i == 17 and ok(17))]
    sectors = _frame(
        CATALOG["sectors"].schema,
        {
            "symbol": [s for _, s in sec],
            "industry": [_INDUSTRIES[i % 4] for i, _ in sec],
            "company": [f"{symbols[i]} Ltd" for i, _ in sec],
            "isin": [_isin(symbols[i]) for i, _ in sec],
            "source": ["synth"] * len(sec),
            "as_of": [as_of] * len(sec),
        },
    )
    return SynthMarket(sessions, eq, delivery, corporate_actions, symbol_changes, listed, etf, sectors, truth)


# ---- tiny hand-written market -----------------------------------------------------------------

_TINY = {
    "AAA": (
        [100, 101, 103, 102, 104, 105, 104, 106, 107, 106, 108, 109],
        [101, 103, 102, 104, 105, 104, 106, 107, 106, 108, 109, 110],
    ),
    "BBB": (
        [50, 50, 49, 51, 52, 51, 50, 49, 50, 51, 52, 53],
        [50, 49, 51, 52, 51, 50, 49, 50, 51, 52, 53, 52],
    ),
    "CCC": (
        [200, 198, 202, 205, 203, 200, 204, 206, 205, 207, 210, 208],
        [199, 202, 205, 203, 200, 204, 206, 205, 207, 210, 208, 211],
    ),
}


def make_tiny_market() -> SynthMarket:
    """3 stocks x 12 sessions with hand-written prices (high = max(o,c)+1, low = min(o,c)-1)."""
    sessions = _make_sessions(12)
    eq_rows: dict[str, list] = {c: [] for c in CATALOG["bhavcopy_eq"].schema}
    dl_rows: dict[str, list] = {c: [] for c in CATALOG["bhavcopy_delivery"].schema}
    tr_rows: dict[str, list] = {"symbol_key": [], "date": [], "true_ret": [], "alpha": []}
    for sym, (opens, closes) in _TINY.items():
        for t in range(12):
            o, c = float(opens[t]), float(closes[t])
            prev = o if t == 0 else float(closes[t - 1])
            vol = 1000 * (t + 1)
            hi, lo = max(o, c) + 1.0, min(o, c) - 1.0
            for k, v in {
                "symbol": sym,
                "series": "EQ",
                "date": sessions[t],
                "open": o,
                "high": hi,
                "low": lo,
                "close": c,
                "prev_close": prev,
                "last_price": c,
                "volume": vol,
                "turnover_inr": vol * (o + hi + lo + c) / 4.0,
                "n_trades": vol / 50.0,
                "era": "legacy",
            }.items():
                eq_rows[k].append(v)
            for k, v in {
                "symbol": sym,
                "series": "EQ",
                "date": sessions[t],
                "deliv_qty": vol * 0.5,
                "deliv_pct": 50.0,
            }.items():
                dl_rows[k].append(v)
            tr_rows["symbol_key"].append(sym)
            tr_rows["date"].append(sessions[t])
            tr_rows["true_ret"].append(None if t == 0 else c / prev - 1.0)
            tr_rows["alpha"].append(0.0)
    as_of = sessions[-1]
    names = list(_TINY)
    return SynthMarket(
        sessions=sessions,
        eq=_frame(CATALOG["bhavcopy_eq"].schema, eq_rows),
        delivery=_frame(CATALOG["bhavcopy_delivery"].schema, dl_rows),
        corporate_actions=_frame(CATALOG["corporate_actions"].schema, {}),
        symbol_changes=_frame(CATALOG["symbol_changes"].schema, {}),
        listed=_frame(
            CATALOG["listed_equities"].schema,
            {
                "symbol": names,
                "name": [f"{s} Ltd" for s in names],
                "series": ["EQ"] * 3,
                "listing_date": [sessions[0]] * 3,
                "isin": [_isin(s) for s in names],
                "face_value": [10.0] * 3,
                "as_of": [as_of] * 3,
            },
        ),
        etf=_frame(CATALOG["etf_list"].schema, {}),
        sectors=_frame(
            CATALOG["sectors"].schema,
            {
                "symbol": names,
                "industry": [_INDUSTRIES[i % 4] for i in range(3)],
                "company": [f"{s} Ltd" for s in names],
                "isin": [_isin(s) for s in names],
                "source": ["synth"] * 3,
                "as_of": [as_of] * 3,
            },
        ),
        truth=_frame(
            {"symbol_key": pl.Utf8, "date": pl.Date, "true_ret": pl.Float64, "alpha": pl.Float64}, tr_rows
        ),
    )


# ---- persistence ------------------------------------------------------------------------------


def _settings_for(root: Path) -> Settings:
    s = load_settings(_REPO / "config", root / "no.env")
    return s.model_copy(update={"paths": Paths(data_root=root)})


def write_market(m: SynthMarket, root: Path) -> None:
    """Write every frame into the real partition layout under data root `root` (via data.io)."""
    s = _settings_for(Path(root))
    for name, df in [
        ("bhavcopy_eq", m.eq),
        ("bhavcopy_delivery", m.delivery),
        ("corporate_actions", m.corporate_actions),
        ("symbol_changes", m.symbol_changes),
        ("listed_equities", m.listed),
        ("etf_list", m.etf),
        ("sectors", m.sectors),
    ]:
        if df.height:
            io.upsert(s, name, df)
