"""Run Market Lab's own pure functions on the synthetic inputs and save their outputs (the golden files).

This is the ONLY sanctioned execution of Market Lab code (PLAN-M Part 0 rule 15). Run it with the clone's own
interpreter, never ours, and never install anything into that venv:

    C:\\Users\\harsh\\Downloads\\market-lab\\.venv\\Scripts\\python.exe -I scripts\\golden\\run_marketlab.py ^
        C:\\Users\\harsh\\Downloads\\market-lab tests\\fixtures\\marketlab

It reads inputs_*.json and writes golden_*.json in the same folder. Timestamps become epoch seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def frame(path: Path) -> pd.DataFrame:
    """DataFrame with a UTC DatetimeIndex and open/high/low/close/volume columns."""
    b = json.loads(path.read_text(encoding="utf-8"))
    idx = pd.to_datetime(b["ts"], unit="s", utc=True)
    return pd.DataFrame({k: b[k] for k in ("open", "high", "low", "close", "volume")}, index=idx)


def clean(x):
    """Recursively convert pandas/numpy objects to JSON types (timestamps to epoch seconds)."""
    if isinstance(x, pd.Timestamp):
        return int(x.timestamp())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return [clean(r) for r in x.to_dict("records")]
    if isinstance(x, pd.Series):
        return [clean(v) for v in x.tolist()]
    return x


def tape_snapshot(clone: Path, fx: Path, daily_frame: pd.DataFrame) -> dict:
    """Market Lab's feed.cycle and snapshot on a recorded moment (network and clock stubbed)."""
    import types

    from sm import feed

    inp = json.loads((fx / "inputs_tape.json").read_text(encoding="utf-8"))
    feed.STATE["meta"] = inp["meta"]
    feed.STATE["daily"] = inp["daily"]
    feed.STATE["industry"] = inp["industry"]
    feed.STATE["daily_idx"] = {k: daily_frame for k in feed.INDICES}
    for sym, g in inp["spark"].items():
        closes = pd.Series(g["close"], index=pd.to_datetime(g["ts"], unit="s", utc=True))
        feed.GOT[sym] = {"closes": closes, "ltp": g["ltp"], "prev": g["prev"], "time": g["time"]}
    feed.fetch = lambda ysyms: None  # prices are already in GOT
    feed.M.minute_now = lambda: inp["now_minute"]
    feed.M.market_status = lambda t=None: "open"
    feed.next_expiry = lambda key: None
    feed.time = types.SimpleNamespace(time=lambda: inp["now_epoch"], sleep=lambda s: None)
    feed.cycle(True, True)
    snap = feed.snapshot(True)
    snap.pop("time")
    return snap


def sector_goldens(clone: Path, fx: Path) -> dict:
    """Market Lab's horizon, sector_view, stock_view, search and tick on a recorded universe (no network)."""
    import time

    import app
    from sx import live as L
    from sx import market as M
    from sx import news as N
    from sx import universe as U

    inp = json.loads((fx / "inputs_sector.json").read_text(encoding="utf-8"))
    dates = pd.to_datetime(inp["dates"])
    hist = {}
    for sym, h in inp["hist"].items():
        hist[sym] = pd.DataFrame({k: h[k] for k in ("close", "high", "volume")}, index=dates[-h["n"] :])
    for name, h in inp["idx_hist"].items():
        hist["IDX:" + name] = pd.DataFrame(
            {"close": h["close"], "high": h["close"], "volume": 0.0}, index=dates
        )
    U.U["list"], U.U["cls"], U.U["hist"] = inp["list"], inp["cls"], hist
    L.S["indices"], L.S["stocks"], L.S["members"], L.S["spark"] = (
        inp["indices"],
        inp["stocks"],
        inp["members"],
        {},
    )
    L.S["day"], L.S["time"] = {"snaps": inp["snaps"]}, "2026-11-19T09:00:00+00:00"
    for sec in U.SECTORS:
        N._CACHE[(N.QUERY[sec["name"]], 30)] = (time.time(), inp["news"][sec["name"]])
    app._C["mk"] = (time.time(), inp["mk"])
    N.stock = lambda sym, company: inp["stock_news"]

    def quote(sym):
        if sym == "BNK02" or sym not in inp["quotes"]:
            raise RuntimeError("no quote")
        return inp["quotes"][sym]

    U.quote = quote
    M.market_status = lambda t=None: "open"
    out = {f"horizon_{r}": app.horizon(r) for r in ("today", "w1", "m1", "m3")}
    for name, rng in (
        ("Energy", "today"),
        ("Pharma", "w1"),
        ("Bank", "m3"),
        ("Fin Services", "today"),
        ("Healthcare", "today"),
    ):
        out[f"sector_{name.lower().replace(' ', '_')}_{rng}"] = app.sector_view(name, rng)
    for sym in ("ENG03", "BNK02", "ZZZ"):
        out[f"stock_{sym.lower()}"] = app.stock_view(sym)
    out["search_en"], out["search_bank"] = app.search("EN"), app.search("bank")
    out["tick"] = app.tick(["ENG01", "ENG02", "NOPE"])
    return out


def research_goldens(clone: Path, fx: Path) -> dict:
    """Market Lab's research-note build() for three stocks on recorded payloads (no network)."""
    import tempfile
    import types

    from lab import bars as B
    from lab import research as RS

    inp = json.loads((fx / "inputs_research.json").read_text(encoding="utf-8"))
    sec = json.loads((fx / "inputs_sector.json").read_text(encoding="utf-8"))
    dates = pd.to_datetime(sec["dates"])
    frames = {}
    for sym, h in sec["hist"].items():
        frames[sym] = pd.DataFrame(
            {
                "open": h["close"],
                "high": h["high"],
                "low": h["close"],
                "close": h["close"],
                "volume": h["volume"],
            },
            index=dates[-h["n"] :],
        )
    nx = sec["idx_hist"]["NIFTY 50"]["close"]
    frames["NIFTY"] = pd.DataFrame(
        {"open": nx, "high": nx, "low": nx, "close": nx, "volume": 1.0}, index=dates
    )
    B.load = lambda sym, tf="1d", *a, **k: frames.get(sym)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "classification.json").write_text(json.dumps(sec["cls"]), encoding="utf-8")

    def get_json(url, timeout=20):
        sym = url.rsplit("symbol=", 1)[1]
        if "getIntegratedFilingData" in url:
            return inp["filings"][sym]
        if inp["cmp"][sym] is None:
            raise RuntimeError("no filing")
        return inp["cmp"][sym]

    RS.M = types.SimpleNamespace(DATA=tmp, get_json=get_json)
    RS.U.quote = lambda sym: inp["quotes"][sym]
    RS.wiki = lambda name: inp["wiki"].get(name)
    RS.macro = lambda: inp["macro"]
    out = {}
    for sym in ("ENG03", "BNK02", "ENG14"):
        r = RS.build(sym)
        r.pop("built")
        out[sym] = r
    return out


def digest(a) -> dict:
    """Compact fingerprint of an array: length, NaN count, sums and 40 evenly spaced samples (NaN as None)."""
    x = np.asarray(a)
    x = x.astype(float)
    ok = np.isfinite(x)
    idx = np.unique(np.linspace(0, len(x) - 1, 40).round().astype(int)) if len(x) else []
    return {
        "n": int(len(x)),
        "nan": int((~ok).sum()),
        "sum": float(x[ok].sum()),
        "abs": float(np.abs(x[ok]).sum()),
        "at": [None if not ok[i] else float(x[i]) for i in idx],
    }


def digest_all(f: dict) -> dict:
    """Digest of every array in a feature dict (scalars and date objects are skipped)."""
    out = {}
    for k, v in f.items():
        if isinstance(v, np.ndarray) and v.dtype != object and v.dtype.kind in "fiub":
            out[k] = digest(v)
    return out


def features_goldens(clone: Path, fx: Path) -> dict:
    """Market Lab's daily and intraday feature builders, exits and trading windows on the synthetic bars."""
    from lab import engine as E
    from lab import features as F
    from lab import swing as S

    def ist(df):
        df = df.copy()
        df.index = df.index.tz_convert("Asia/Kolkata")
        return df

    d1, nf = ist(frame(fx / "inputs_1d_stock.json")), ist(frame(fx / "inputs_1d_nifty.json"))
    nc = nf["close"]
    nifty = pd.DataFrame(
        {"c": nc.values, "sma200": nc.rolling(200).mean().values, "roc63": nc.pct_change(63).values * 100},
        index=[x.date() for x in nf.index],
    )
    f1 = S.features(d1, nifty)
    out = {"d1": digest_all(f1)}
    follows = []
    for ie in range(300, f1["n"] - 2, 11):
        stop = float(f1["c"][ie - 1] - 2 * f1["atr"][ie - 1])
        for mode in S.EXITS:
            j, px = S.follow(f1, ie, stop, mode)
            follows.append([ie, stop, mode, int(j), float(px)])
    out["follow_1d"] = follows
    s5, ix5 = ist(frame(fx / "inputs_5m_stock.json")), ist(frame(fx / "inputs_5m_index.json"))
    dctx = ist(frame(fx / "inputs_1d_ctx.json"))
    ixf = F.build(ix5, None, 5, None, 1)
    ixm = F.build(F.mirror(ix5), None, 5, None, -1)
    ctx, ctxm = F.index_context(ixf), F.index_context(ixm)
    out["ix"] = digest_all(ixf)
    out["ix_ctx"] = {k: digest(v) for k, v in ctx.items()}
    f5 = F.build(s5, dctx, 5, ctx, 1)
    fm = F.build(F.mirror(s5), F.mirror(dctx), 5, ctxm, -1)
    out["m5"], out["m5_mirror"] = digest_all(f5), digest_all(fm)
    out["m5_nodaily"] = digest_all(F.build(s5, None, 5, None, 1))
    entry_ok, exit_idx = E.windows(f5)
    out["windows"] = {"entry_ok": digest(entry_ok), "exit_idx": digest(exit_idx)}
    out["trend"] = {k: digest(E.trend_mask(f5, k)) for k in ("D", "I", "DI", "DIX")}
    fi = []
    for i in range(80, f5["n"] - 80, 37):
        ie = i + 1
        if f5["day"][ie] != f5["day"][i] or exit_idx[f5["day"][i]] < ie:
            continue
        s, e = float(f5["l"][i] - 0.3 * f5["atr"][i]), float(f5["o"][ie])
        if e - s <= 0:
            continue
        o = E.follow(f5, ie, s, e, e - s, int(exit_idx[f5["day"][i]]))
        fi.append([ie, s, e, int(exit_idx[f5["day"][i]]), {k: float(v) for k, v in o.items()}])
    out["follow_intra"] = fi
    return out


def sig_rows(fn, f) -> dict:
    """Indices where a strategy fires and its stop there (NaN as None)."""
    sig, stop = fn(f)
    sig = np.asarray(sig, bool)
    idx = np.flatnonzero(sig)
    st = np.asarray(stop, float)[idx]
    return {"idx": [int(i) for i in idx], "stop": [None if not np.isfinite(x) else float(x) for x in st]}


def signals_goldens(clone: Path, fx: Path) -> dict:
    """Market Lab's 23 swing and 39 intraday strategy functions on the synthetic features."""
    from lab import features as F
    from lab import strategies as T
    from lab import swing as S

    def ist(df):
        df = df.copy()
        df.index = df.index.tz_convert("Asia/Kolkata")
        return df

    d1, nf = ist(frame(fx / "inputs_1d_stock.json")), ist(frame(fx / "inputs_1d_nifty.json"))
    nc = nf["close"]
    nifty = pd.DataFrame(
        {"c": nc.values, "sma200": nc.rolling(200).mean().values, "roc63": nc.pct_change(63).values * 100},
        index=[x.date() for x in nf.index],
    )
    f1 = S.features(d1, nifty)
    out = {"swing": {name: sig_rows(fn, f1) for name, (fn, _kind) in S.STRATS.items()}}
    out["swing_kinds"] = {name: kind for name, (_fn, kind) in S.STRATS.items()}
    s5, ix5 = ist(frame(fx / "inputs_5m_stock.json")), ist(frame(fx / "inputs_5m_index.json"))
    dctx = ist(frame(fx / "inputs_1d_ctx.json"))
    ixf = F.build(ix5, None, 5, None, 1)
    ixm = F.build(F.mirror(ix5), None, 5, None, -1)
    ctx, ctxm = F.index_context(ixf), F.index_context(ixm)
    f5 = F.build(s5, dctx, 5, ctx, 1)
    fm = F.build(F.mirror(s5), F.mirror(dctx), 5, ctxm, -1)
    out["intra"] = {name: sig_rows(fn, f5) for name, fn in T.STRATEGIES.items()}
    out["intra_mirror"] = {name: sig_rows(fn, fm) for name, fn in T.STRATEGIES.items()}
    return out


def main(clone: Path, fx: Path) -> None:
    sys.path.insert(0, str(clone))
    from sm import smc, structure  # noqa: PLC0415 (Market Lab's own modules)

    s5, i5 = frame(fx / "inputs_5m_stock.json"), frame(fx / "inputs_5m_index.json")
    s1, m15 = frame(fx / "inputs_1d_stock.json"), frame(fx / "inputs_15m_stock.json")
    d5 = frame(fx / "inputs_1d_from5m.json")
    out = {
        "smc_5m_stock": smc.read(s5, 2),
        "smc_5m_index": smc.read(i5, 2),
        "smc_1d_stock": smc.read(s1, 3),
        "smc_5m_stock_daily": smc.read(s5, 2, daily=d5),
    }
    for name, df, n in (("5m_stock", s5, 3), ("1d_stock", s1, 3), ("5m_index", i5, 3)):
        a = structure.analyse(df, n)
        out[f"analyse_{name}"] = {
            "swings": a["swings"],
            "events": a["events"],
            "bias": a["bias"],
            "last_event": a["last_event"],
            "recent_swings": a["recent_swings"],
            "zones": a["zones"],
            "regime": a["regime"],
            "price": a["price"],
            "atr": a["atr"],
            "rsi": a["rsi"],
            "df_atr_last": float(a["df"]["atr"].iloc[-1]),
            "df_rsi_last": float(a["df"]["rsi"].iloc[-1]),
            "df_ema200_last": float(a["df"]["ema200"].iloc[-1]),
        }
    out["session_levels_15m"] = structure.session_levels(m15)
    out["tape_snapshot"] = tape_snapshot(clone, fx, s1)
    out["sector"] = sector_goldens(clone, fx)
    out["research"] = research_goldens(clone, fx)
    out["features"] = features_goldens(clone, fx)
    out["signals"] = signals_goldens(clone, fx)
    for name, val in out.items():
        (fx / f"golden_{name}.json").write_text(json.dumps(clean(val)), encoding="utf-8")
        print(name, "None" if val is None else "ok")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
