"""Radar scan: every per-stock daily strategy on the u500 and F&O universe at a completed close (PLAN-M M5.6).

Ported from market-lab lab/swing_live.py scan / build_summary (MIT, ba3f771); changes: numpy and polars, all
registered daily strategies at their default exit, our costs, a process pool with top-level workers.

Everything here is exploratory. The scan reads only rows dated <= `asof`, so re-running it later on a longer
panel gives the same rows. Files: `data_store/lab/scan/<asof>.parquet` (the radar rows) and
`<asof>_stats.json` (per-strategy statistics, stock profile, the candidate and the radar for `summary`).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.costs.slippage import SlippageTable
from stocksense.data import io
from stocksense.strategies import base
from stocksense.strategies.costs import trade_cost_frac
from stocksense.strategies.data import NiftyTable, nifty_table
from stocksense.strategies.explore import (
    CANDIDATE,
    EXIT_NAMES,
    OOS_START,
    WARMUP,
    Prep,
    SymCtx,
    epoch_date,
    gated_verdicts,
    load_ctx,
    make_prep,
    trades_1d,
    universe_symbols,
)
from stocksense.strategies.profile import stock_profile

log = get_logger(__name__)

WORKERS = 6
CHUNK = 20
NOTIONAL = 100_000.0
RECENT_DAYS = 15
YEARS = (2022, 2023, 2024, 2025, 2026)
STATES = ("SIGNAL", "OPEN", "RECENT", "FORMING")
ROW_SCHEMA: dict[str, Any] = {
    "sym": pl.Utf8,
    "strat": pl.Utf8,
    "state": pl.Utf8,
    "t_in": pl.Int64,
    "p_in": pl.Float64,
    "stop": pl.Float64,
    "target": pl.Float64,
    "rs63": pl.Float64,
    "dist_pct": pl.Float64,
    "risk_pct": pl.Float64,
    "trigger": pl.Float64,
    "close": pl.Float64,
}
CAND_RULES = [
    "Universe: Nifty 500 + F&O stocks, delivery (CNC), longs only",
    "Market filter: buy only while Nifty closes above its 200-day average",
    "Stock trend: close > 200 SMA, 50 SMA > 200 SMA, 20 EMA > 50 SMA",
    "Setup: within 8% of the 52-week high, the last 10 days' range under 3.6 x ATR (tightening)",
    "Trigger: close above the 10-day high on volume >= 1.3 x the 50-day average; buy next day's open",
    "Stop: under the 10-day low (-0.3 ATR); target 3 x risk; exit after 40 trading days at the latest",
    "Up to 10 positions of 10% each; when several trigger, take the strongest 63-day relative strength first",
]


# ---- memory ------------------------------------------------------------------------------------


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process in MB (Windows peak working set; ru_maxrss elsewhere)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        c = _Counters()
        c.cb = ctypes.sizeof(c)
        if not ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return c.PeakWorkingSetSize / 2**20
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ---- files -------------------------------------------------------------------------------------


def scan_dir(settings: Settings) -> Path:
    """Folder of scan outputs: data_store/lab/scan."""
    return Paths(settings).lab / "scan"


def scan_path(settings: Settings, asof: date) -> Path:
    """The radar parquet of one scan date."""
    return scan_dir(settings) / f"{asof.isoformat()}.parquet"


def stats_path(settings: Settings, asof: date) -> Path:
    """The statistics JSON of one scan date."""
    return scan_dir(settings) / f"{asof.isoformat()}_stats.json"


def latest_asof(settings: Settings) -> date | None:
    """Newest scan date that has a statistics file."""
    days = sorted(p.name[:10] for p in scan_dir(settings).glob("*_stats.json"))
    return date.fromisoformat(days[-1]) if days else None


def scan_specs() -> list[base.StrategySpec]:
    """Per-stock daily strategies the scan runs, each at its default exit."""
    return [s for s in base.load_library().values() if s.timeframe == "1d" and s.signal is not None]


# ---- worker ------------------------------------------------------------------------------------


def _row(sym: str, strat: str, state: str, **kw: float | int | None) -> dict[str, Any]:
    r: dict[str, Any] = {k: None for k in ROW_SCHEMA}
    r.update(sym=sym, strat=strat, state=state, **kw)
    return r


def _scan_chunk(
    args: tuple[Settings, dict[str, SymCtx], NiftyTable, date, list[str]],
) -> dict[str, Any]:
    """Scan one chunk of symbols whose bars the parent already loaded (top-level so a spawned worker can
    import it; workers never read the panel, which keeps their memory small)."""
    settings, ctxs, nifty, asof, keys = args
    reg = base.load_library()
    syms = sorted(ctxs)
    slip = SlippageTable.load_latest(settings)
    stop_day = np.datetime64(asof)
    cut = asof - timedelta(days=RECENT_DAYS)
    rows: list[dict[str, Any]] = []
    stats: dict[str, list[tuple[float, float, float, float]]] = {k: [] for k in keys}
    cand: list[dict[str, Any]] = []
    done = 0
    for sym in syms:
        ctx = ctxs.get(sym)
        if ctx is None or len(ctx.frame) < WARMUP or ctx.frame.date[-1] != stop_day:
            continue
        prep: Prep = make_prep(settings, ctx, nifty, slip)
        f, close = prep.f, float(ctx.frame.close[-1])
        days = ctx.frame.date.astype("datetime64[s]").astype(np.int64)
        done += 1
        for key in keys:
            spec = reg[key]
            try:
                sig, stop = spec.signal(f)  # type: ignore[misc]
            except (KeyError, ValueError, IndexError) as e:
                log.warning("strategy_error", strat=key, sym=sym, error=type(e).__name__)
                continue
            trades = trades_1d(prep, spec, sig, stop, spec.default_exit, warmup=WARMUP, notional=NOTIONAL)
            live = False
            for t in trades:
                if t.status == "SIGNAL":
                    live = True
                    rows.append(
                        _row(
                            sym,
                            key,
                            "SIGNAL",
                            t_in=t.t_in,
                            p_in=t.p_in,
                            stop=t.stop,
                            target=t.target,
                            rs63=t.rs63,
                            dist_pct=0.0,
                            risk_pct=(t.p_in - t.stop) / t.p_in * 100,
                            trigger=t.p_in,
                            close=close,
                        )
                    )
                elif t.status == "OPEN":
                    live = True
                    if t.mkt_up:
                        rows.append(
                            _row(
                                sym,
                                key,
                                "OPEN",
                                t_in=t.t_in,
                                p_in=t.p_in,
                                stop=t.stop,
                                target=t.target,
                                rs63=t.rs63,
                                dist_pct=(close / t.p_in - 1) * 100,
                                risk_pct=(t.p_in - t.stop) / t.p_in * 100,
                                close=close,
                            )
                        )
                else:
                    stats[key].append((t.R, t.net_pct, t.days, t.t_in // 86400))  # type: ignore[arg-type]
                    if epoch_date(t.t_in) >= cut:
                        rows.append(
                            _row(
                                sym,
                                key,
                                "RECENT",
                                t_in=t.t_in,
                                p_in=t.p_in,
                                stop=t.stop,
                                target=t.target,
                                rs63=t.rs63,
                                dist_pct=t.net_pct,
                                risk_pct=(t.p_in - t.stop) / t.p_in * 100,
                                close=close,
                            )
                        )
                    if key == CANDIDATE["strat"] and t.mkt_up:
                        i = int(np.searchsorted(days, t.t_sig))
                        med = ctx.med_turn[i]
                        cand.append(
                            {
                                "sym": sym,
                                "net": t.net_pct,
                                "risk": (t.p_in - t.stop) / t.p_in * 100,
                                "rs63": t.rs63,
                                "val_cr": float(med) / 1e7 if np.isfinite(med) else None,
                            }
                        )
            if spec.forming is not None and not live and f["n"] >= 10:
                st = spec.forming(f)
                if st and st.get("state") == "FORMING":
                    rows.append(
                        _row(
                            sym,
                            key,
                            "FORMING",
                            p_in=st["trigger"],
                            stop=st["stop"],
                            target=st["target"],
                            rs63=st["rs63"],
                            dist_pct=st["dist_pct"],
                            risk_pct=st["risk_pct"],
                            trigger=st["trigger"],
                            close=st["close"],
                        )
                    )
    return {
        "pid": os.getpid(),
        "peak_mb": peak_rss_mb(),
        "n_symbols": done,
        "rows": rows,
        "stats": {k: np.array(v, dtype=float).reshape(-1, 4) for k, v in stats.items()},
        "cand": cand,
    }


# ---- parent ------------------------------------------------------------------------------------


def _clean(x: Any) -> Any:
    """JSON-safe copy: NaN and inf become None, numpy scalars become Python numbers."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _market_drift(settings: Settings, asof: date) -> float:
    """Mean daily return (fraction) of the equal-weight market up to asof; 0 when not stored."""
    df = io.read(settings, "panel_market", start=None, end=asof, columns=["date", "ew500_ret"]).collect()
    v = df["ew500_ret"].drop_nulls()
    return float(v.mean()) if v.len() else 0.0  # type: ignore[arg-type]


def _strategy_rows(
    stats: dict[str, np.ndarray], mu: float, oos_start: date, verdicts: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Market Lab's `strategies` / `removed` from per-strategy trade arrays (R, net %, days, t_in day)."""
    reg = base.load_library()
    oos_day = int(np.datetime64(oos_start, "D").astype(np.int64))
    working: dict[str, dict[str, Any]] = {}
    removed: list[str] = []
    for key, a in stats.items():
        if not len(a):
            continue
        spec = reg[key]
        net, days, tin = a[:, 1], a[:, 2], a[:, 3]
        exc = net - days * mu * 100
        oos = tin >= oos_day
        year = (tin.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970).astype(int)
        row = {
            "key": key,
            "exit": spec.default_exit,
            "exit_name": EXIT_NAMES[spec.default_exit],
            "n": int(len(a)),
            "win": float((net > 0).mean() * 100),
            "net": float(net.mean()),
            "excess": float(exc.mean()),
            "days": float(days.mean()),
            "IS": float(net[~oos].mean()) if (~oos).any() else None,
            "OOS": float(net[oos].mean()) if oos.any() else None,
            "OOS_exc": float(exc[oos].mean()) if oos.any() else None,
            "years": {int(y): float(net[year == y].mean()) for y in YEARS if (year == y).any()},
            "gated": verdicts.get(key),
        }
        ok = row["excess"] >= 0.25 and (row["OOS_exc"] or 0) > 0 and row["net"] > 0
        if ok:
            working[spec.name] = row
        else:
            removed.append(spec.name)
    return working, sorted(removed)


def _classification(settings: Settings) -> dict[str, dict[str, Any]]:
    df = io.read(settings, "sectors", columns=["symbol", "industry", "as_of"]).collect()
    if not df.height:
        return {}
    df = df.sort("as_of").unique(subset=["symbol"], keep="last")
    return {r["symbol"]: {"sector": r["industry"]} for r in df.iter_rows(named=True)}


def run_scan(
    settings: Settings,
    asof: date,
    *,
    workers: int = WORKERS,
    symbols: list[str] | None = None,
    chunk: int = CHUNK,
    clock: Clock | None = None,
) -> Path:
    """Scan the u500 and F&O universe at `asof`; write the radar parquet and the statistics JSON.

    `workers` <= 1 runs in this process. Returns the parquet path. Re-running overwrites with the same rows.
    """
    clk = clock or SystemClock()
    reg_keys = [s.key for s in scan_specs()]
    syms = symbols if symbols is not None else universe_symbols(settings, asof)
    if not syms:
        raise DataNotAvailable(f"no universe symbols at {asof}")
    ctxs = load_ctx(settings, syms, asof)
    nifty = nifty_table(settings, end=asof)
    names = sorted(ctxs)
    jobs = [
        (settings, {k: ctxs[k] for k in names[i : i + chunk]}, nifty, asof, reg_keys)
        for i in range(0, len(names), chunk)
    ]
    if workers <= 1:
        results = [_scan_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_scan_chunk, jobs))
    parent_peak = peak_rss_mb() or 0.0
    peaks: dict[int, float] = {}  # each worker's own peak working set (the largest it reported)
    for r in results:
        peaks[r["pid"]] = max(peaks.get(r["pid"], 0.0), r["peak_mb"] or 0.0)
    peak_total = parent_peak + sum(v for p, v in peaks.items() if p != os.getpid())

    rows = [r for res in results for r in res["rows"]]
    df = pl.DataFrame({k: [r[k] for r in rows] for k in ROW_SCHEMA}, schema=ROW_SCHEMA)
    df = df.sort("state", "strat", "sym", "t_in", nulls_last=True)
    out = scan_path(settings, asof)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    df.write_parquet(tmp)
    tmp.replace(out)

    stats = {
        k: np.concatenate([res["stats"][k] for res in results if len(res["stats"][k])] or [np.empty((0, 4))])
        for k in reg_keys
    }
    cand = [c for res in results for c in res["cand"]]
    verdicts = gated_verdicts(settings)
    mu = _market_drift(settings, asof)
    strategies, removed = _strategy_rows(stats, mu, OOS_START, verdicts)
    profile, best = stock_profile(cand, _classification(settings))
    nt = nifty
    nc, n200 = (float(nt.c[-1]), float(nt.sma200[-1])) if len(nt.c) else (float("nan"), float("nan"))
    ca = stats.get(CANDIDATE["strat"], np.empty((0, 4)))
    r_c = ca[:, 0] if len(ca) else np.array([])
    payload = {
        "asof": asof.isoformat(),
        "built": clk.now().isoformat(timespec="seconds"),
        "built_ts": int(clk.now().timestamp()),
        "peak_rss_mb": round(peak_total, 1),
        "stocks": sum(r["n_symbols"] for r in results),
        "trades": int(sum(len(a) for a in stats.values())),
        "cost_pct": round(
            100
            * trade_cost_frac(
                "delivery",
                NOTIONAL,
                settings,
                SlippageTable.load_latest(settings),
                200,
                0.02,
                NOTIONAL * 100,
                "open",
                "now",
            ),
            3,
        ),
        "market_drift_daily_pct": mu * 100,
        "nifty": {"close": nc, "sma200": n200, "mkt_up": bool(nc > n200)},
        "strategies": strategies,
        "removed": removed,
        "profile": profile,
        "best_stocks": best,
        "candidate_stats": {
            "n": int(len(r_c)),
            "win": float((r_c > 0).mean() * 100) if len(r_c) else None,
            "avgR": float(r_c.mean()) if len(r_c) else None,
            "totR": float(r_c.sum()) if len(r_c) else 0.0,
            "net_pct": float(ca[:, 1].sum()) if len(ca) else 0.0,
        },
    }
    sp = stats_path(settings, asof)
    sp.write_text(json.dumps(_clean(payload)), encoding="utf-8")
    log.info(
        "scan_done", asof=str(asof), rows=df.height, stocks=payload["stocks"], peak_mb=payload["peak_rss_mb"]
    )
    return out


# ---- summary -----------------------------------------------------------------------------------


def _live(settings: Settings, asof: date, stats: dict[str, Any]) -> dict[str, Any]:
    """The radar from the scan parquet, in Market Lab's `live` tree."""
    names = {k: s.name for k, s in base.load_library().items()}
    df = (
        pl.read_parquet(scan_path(settings, asof))
        if scan_path(settings, asof).is_file()
        else pl.DataFrame(schema=ROW_SCHEMA)
    )

    def pick(state: str) -> list[dict[str, Any]]:
        out = []
        for r in df.filter(pl.col("state") == state).iter_rows(named=True):
            r = _clean(r)
            r["strat_name"] = names.get(r["strat"], r["strat"])
            out.append(r)
        return out

    def by(rows: list[dict[str, Any]], key: str, rev: bool) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda x: (x[key] is None, -(x[key] or 0) if rev else (x[key] or 0)))

    nf = stats.get("nifty", {})
    return {
        "time": stats.get("built_ts"),
        "asof": asof.isoformat(),
        "nifty": _r1(nf.get("close")),
        "nifty_200": _r1(nf.get("sma200")),
        "mkt_up": nf.get("mkt_up"),
        "signals": by(pick("SIGNAL"), "rs63", True),
        "open": by(pick("OPEN"), "t_in", True),
        "forming": by(pick("FORMING"), "dist_pct", False),
        "recent": by(pick("RECENT"), "t_in", True),
    }


def _r1(x: float | None) -> float | None:
    return None if x is None else round(float(x), 1)


def summary(settings: Settings) -> dict[str, Any]:
    """Market Lab's /api/lab/swing key tree from the latest scan (exploratory; `gated` per strategy).

    Keys: built, asof, cost_pct, stocks, trades, strategies, removed, profile, best_stocks, candidate, live,
    list, exits. Without a scan it returns {"error": ...}.
    """
    asof = latest_asof(settings)
    if asof is None:
        return {"error": "run: stocksense lab scan"}
    stats = json.loads(stats_path(settings, asof).read_text(encoding="utf-8"))
    verdicts = gated_verdicts(settings)
    reg = base.load_library()
    passing = sorted(k for k, v in verdicts.items() if v == "PASS" and k in reg)
    cs = stats["candidate_stats"]
    if passing:
        spec = reg[passing[0]]
        cand = {
            "strat": spec.name,
            "exit": spec.default_exit,
            "k": 10,
            "label": f"{spec.name} (gate v2 PASS)",
            "rules": [],
            "gated": "PASS",
            "exploratory": False,
            "metrics": None,
        }
    else:
        spec = reg[CANDIDATE["strat"]]
        cand = {
            "strat": spec.name,
            "exit": CANDIDATE["exit"],
            "k": 10,
            "label": "VCP tightening break · Nifty above 200-DMA · 3R / 40 days (exploratory candidate)",
            "rules": CAND_RULES,
            "gated": verdicts.get(spec.key),
            "exploratory": True,
            "metrics": cs,
        }
    return {
        "built": stats["built"],
        "asof": stats["asof"],
        "cost_pct": stats.get("cost_pct"),
        "stocks": stats["stocks"],
        "trades": stats["trades"],
        "peak_rss_mb": stats.get("peak_rss_mb"),
        "strategies": stats["strategies"],
        "removed": stats["removed"],
        "profile": stats["profile"],
        "best_stocks": stats["best_stocks"],
        "candidate": cand,
        "live": _live(settings, asof, stats),
        "list": [s.name for s in scan_specs()],
        "exits": EXIT_NAMES,
        "exploratory": True,
    }
