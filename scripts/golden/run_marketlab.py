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
    for name, val in out.items():
        (fx / f"golden_{name}.json").write_text(json.dumps(clean(val)), encoding="utf-8")
        print(name, "None" if val is None else "ok")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
