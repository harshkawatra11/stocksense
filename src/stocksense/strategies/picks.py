"""Daily Picks: today's stocks from the strategies whose backtests survived the gate (PLAN-M M5.9).

A strategy may pick only if its latest `reports/m/strategies/<trial>/eval.json` says PASS on all nine gate-v2 lines.
Each gated strategy gets a weight `w_s = DSR_s * min(1, trades_oos_s / 100)` (the DSR is already penalised by the
total number of trials); a stock's evidence is the sum of the weights of the gated strategies that fire on it.
Signals come from that day's scan (`data_store/lab/scan/<asof>.parquet`, state SIGNAL, completed bars only); a gated
cross-sectional strategy fires on the ten best-scored names of its universe. Exploratory candidates (strategies that
have not passed) are listed apart under a fixed label and are never mixed into the gated list.

Paper only: sizes are paper quantities from a capital the caller read from the broker at decision time (CLAUDE.md
rule 6); the capital itself is never stored or returned. The result of one `asof` is written once to
`data_store/lab/picks/<asof>.json` (CLAUDE.md rule 7) and every gated pick is journalled as `picks:<trial>`.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.costs.india import round_trip_cost_frac
from stocksense.data import io
from stocksense.strategies import base, journal, scan
from stocksense.validation.trials import n_trials

log = get_logger(__name__)

LABEL = "EXPLORATORY — not validated"
FOOTER = "Backtests decide which strategies may pick; past results do not guarantee future returns."
DAILY_FAMILIES = ("m-swing", "m-xs")
OTHER_FAMILIES = ("m-intra", "m-opt")
XS_TOP = 10
TRADES_FULL_WEIGHT = 100
REFERENCE_KEY = "500000"
JOB = "picks"
XsScores = Callable[[str, date], list[str]]
Closes = Callable[[date, list[str]], dict[str, float]]


@dataclass(frozen=True)
class Pick:
    """One gated pick. `stop` and `target` are None when the best firing strategy has none (a cross-sectional
    score); then the paper quantity is the equal-weight slice only."""

    sym: str
    side: int
    entry_ref: float
    stop: float | None
    target: float | None
    strategies: tuple[str, ...]
    evidence: float
    best_trial: str
    net_sharpe_5l: float
    dsr: float
    pbo: float
    trades_oos: int
    qty: int
    notional_inr: float
    risk_inr: float
    reason: str


@dataclass(frozen=True)
class Gated:
    """A strategy that passed all nine gate-v2 lines, with the numbers of its best passing trial."""

    key: str
    name: str
    family: str
    trial_id: str
    dsr: float
    pbo: float
    net_sharpe_5l: float
    trades_oos: int

    @property
    def weight(self) -> float:
        """w_s = DSR x min(1, trades / 100)."""
        return self.dsr * min(1.0, self.trades_oos / TRADES_FULL_WEIGHT)


# ---- files -----------------------------------------------------------------------------------------------


def picks_dir(settings: Settings) -> Path:
    """`data_store/lab/picks`."""
    return Paths(settings).lab / "picks"


def picks_path(settings: Settings, asof: date) -> Path:
    """The stored picks of one date."""
    return picks_dir(settings) / f"{asof.isoformat()}.json"


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, date):
        return x.isoformat()
    return x


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_clean(payload), indent=1), encoding="utf-8", newline="\n")
    tmp.replace(path)


# ---- the gated set ---------------------------------------------------------------------------------------


def _line(ev: dict[str, Any], name: str) -> float | None:
    for ln in (ev.get("gate") or {}).get("lines", []):
        if ln.get("name") == name:
            return ln.get("value")
    return None


def _passes_all_nine(ev: dict[str, Any]) -> bool:
    g = ev.get("gate") or {}
    lines = g.get("lines") or []
    return (
        ev.get("verdict") == "PASS"
        and str(g.get("kind", "")).endswith("v2")
        and len(lines) == 9
        and all(ln.get("passed") for ln in lines)
    )


def passing_strategies(settings: Settings) -> dict[str, Gated]:
    """Strategy key -> its best passing trial, from every eval.json that passes all nine v2 lines."""
    root = Paths(settings).reports / "m" / "strategies"
    best: dict[str, Gated] = {}
    reg = base.load_library()
    for p in sorted(root.glob("*/eval.json")) if root.is_dir() else []:
        try:
            ev = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("eval_unreadable", path=str(p), error=type(e).__name__)
            continue
        if not _passes_all_nine(ev):
            continue
        key = str(ev.get("strategy") or ev.get("strat"))
        by_cap = ev.get("net_sharpe_by_capital") or {}
        sharpe = by_cap.get(REFERENCE_KEY, ev.get("net_sharpe"))
        g = Gated(
            key,
            reg[key].name if key in reg else key,
            str(ev.get("family")),
            str(ev["trial_id"]),
            float(_line(ev, "dsr") or 0.0),
            float(_line(ev, "pbo") if _line(ev, "pbo") is not None else 1.0),
            float(sharpe if sharpe is not None else 0.0),
            int(ev.get("trades") or 0),
        )
        if key not in best or g.net_sharpe_5l > best[key].net_sharpe_5l:
            best[key] = g
    return best


# ---- sizing ----------------------------------------------------------------------------------------------


def size_position(
    capital_inr: float | None, risk_per_trade: float, max_picks: int, entry: float, stop: float | None
) -> tuple[int, float, float]:
    """(quantity, notional INR, risk INR): floor(min(capital x risk / (entry - stop), capital / max_picks / entry)).

    Without a stop only the equal-weight slice applies. 0 when the capital is unknown or the entry is invalid.
    """
    if capital_inr is None or capital_inr <= 0 or entry <= 0:
        return 0, 0.0, 0.0
    by_slice = capital_inr / max_picks / entry
    if stop is not None and entry - stop > 0:
        qty = math.floor(min(capital_inr * risk_per_trade / (entry - stop), by_slice))
    else:
        qty = math.floor(by_slice)
    qty = max(qty, 0)
    risk = qty * (entry - stop) if stop is not None and entry > stop else 0.0
    return qty, qty * entry, risk


# ---- signals ---------------------------------------------------------------------------------------------


def latest_scan_asof(settings: Settings) -> date | None:
    """Newest date that has a radar parquet (`data_store/lab/scan/<asof>.parquet`)."""
    days = sorted(p.stem for p in scan.scan_dir(settings).glob("????-??-??.parquet"))
    return date.fromisoformat(days[-1]) if days else None


def _scan_signals(settings: Settings, asof: date) -> pl.DataFrame:
    p = scan.scan_path(settings, asof)
    if not p.is_file():
        raise DataNotAvailable(f"no scan for {asof}: run stocksense lab scan --asof {asof}")
    return pl.read_parquet(p).filter(pl.col("state") == "SIGNAL")


def _xs_top(key: str, asof: date, settings: Settings) -> list[str]:
    """The `XS_TOP` best-scored symbols of a cross-sectional strategy on `asof`, restricted to the u500 then."""
    spec = base.load_library()[key]
    if spec.score is None:
        return []
    sc = spec.score(settings, asof, asof)
    if not sc.height:
        return []
    sc = sc.filter(pl.col("date") == asof).drop_nulls("score")
    uni = (
        io.read(settings, "panel_universe", start=asof, end=asof, columns=["date", "sid", "u500"])
        .filter(pl.col("u500"))
        .collect()
    )
    top = sc.join(uni.select("sid"), on="sid").sort("score", descending=True).head(XS_TOP)
    px = (
        io.read(settings, "panel_daily", start=asof, end=asof, columns=["sid", "symbol", "series"])
        .filter(pl.col("series") == "EQ")
        .collect()
    )
    names = dict(zip(px["sid"].to_list(), px["symbol"].to_list(), strict=True))
    return [names[s] for s in top["sid"].to_list() if s in names]


def _closes(settings: Settings, asof: date, syms: list[str]) -> dict[str, float]:
    if not syms:
        return {}
    df = (
        io.read(
            settings, "panel_daily", start=asof, end=asof, columns=["symbol", "series", "date", "adj_close"]
        )
        .filter(pl.col("symbol").is_in(syms) & (pl.col("series") == "EQ"))
        .collect()
    )
    return {
        r["symbol"]: float(r["adj_close"]) for r in df.iter_rows(named=True) if r["adj_close"] is not None
    }


def _exploratory(
    settings: Settings, asof: date, sig: pl.DataFrame, gated: dict[str, Gated]
) -> list[dict[str, Any]]:
    """The ten best signals of strategies that have not passed, ranked by the scan's full-universe excess return."""
    rows = sig.filter(~pl.col("strat").is_in(list(gated)))
    if not rows.height:
        return []
    reg = base.load_library()
    excess: dict[str, float] = {}
    sp = scan.stats_path(settings, asof)
    if sp.is_file():
        stats = json.loads(sp.read_text(encoding="utf-8")).get("strategies") or {}
        excess = {s["key"]: float(s["excess"]) for s in stats.values() if s.get("excess") is not None}
    out = []
    for r in rows.iter_rows(named=True):
        spec = reg.get(r["strat"])
        out.append(
            {
                "sym": r["sym"], "strat": r["strat"], "name": spec.name if spec else r["strat"], "side": 1,
                "entry_ref": r["p_in"], "stop": r["stop"], "target": r["target"], "rs63": r["rs63"],
                "excess": excess.get(r["strat"]), "label": LABEL, "exploratory": True,
            }
        )  # fmt: skip
    out.sort(key=lambda d: (d["excess"] is None, -(d["excess"] or 0.0), -(d["rs63"] or 0.0), d["sym"]))
    return out[:10]


# ---- the picks -------------------------------------------------------------------------------------------


def build_picks(
    settings: Settings,
    state: State,
    asof: date,
    capital_inr: float | None,
    max_picks: int = 10,
    risk_per_trade: float = 0.01,
    *,
    clock: Clock | None = None,
    force: bool = False,
    persist: bool = True,
    xs_scores: XsScores | None = None,
    closes: Closes | None = None,
) -> dict[str, Any]:
    """Daily Picks at the close of `asof`: {asof, gated: [Pick], exploratory: [...], other_gated, n_strategies_passing,
    n_trials_total, note, footer}. The gated list uses only strategies whose latest gate-v2 report passes all
    nine lines; the exploratory list is labelled and never merged into it.

    `capital_inr` is read from the broker by the caller at decision time (None when unavailable: quantities
    are then 0). The stored file of `asof` is returned as is unless `force`; with `persist=False` nothing is
    written or journalled (a preview, used by the API when no file exists yet).
    """
    path = picks_path(settings, asof)
    if persist and path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    clk = clock or SystemClock()
    sig = _scan_signals(settings, asof)
    passing = passing_strategies(settings)
    daily = {k: g for k, g in passing.items() if g.family in DAILY_FAMILIES}
    other = [g for g in passing.values() if g.family in OTHER_FAMILIES]
    total = n_trials(state)
    fire: dict[str, list[tuple[Gated, dict[str, Any] | None]]] = {}
    for r in sig.filter(
        pl.col("strat").is_in([k for k, g in daily.items() if g.family == "m-swing"])
    ).iter_rows(named=True):
        fire.setdefault(r["sym"], []).append((daily[r["strat"]], r))
    xs_keys = [k for k, g in daily.items() if g.family == "m-xs"]
    xs_syms: dict[str, list[str]] = {}
    for k in xs_keys:
        try:
            top = (xs_scores or (lambda key, d: _xs_top(key, d, settings)))(k, asof)
        except (DataNotAvailable, KeyError, ValueError) as e:
            log.warning("xs_picks_failed", strat=k, error=type(e).__name__)
            continue
        xs_syms[k] = top[:XS_TOP]
        for s in xs_syms[k]:
            fire.setdefault(s, []).append((daily[k], None))
    need = [s for s, v in fire.items() if all(row is None for _, row in v)]
    px = (closes or (lambda d, syms: _closes(settings, d, syms)))(asof, need)
    picks: list[Pick] = []
    for sym, firing in fire.items():
        best_g = max((g for g, _ in firing), key=lambda g: (g.weight, g.net_sharpe_5l))
        with_stop = [
            (g, r) for g, r in sorted(firing, key=lambda t: (-t[0].weight, -t[0].net_sharpe_5l)) if r
        ]
        row = with_stop[0][1] if with_stop else None
        entry = float(row["p_in"]) if row else px.get(sym)
        if entry is None:
            continue
        stop = float(row["stop"]) if row and row["stop"] is not None else None
        target = float(row["target"]) if row and row["target"] is not None else None
        qty, notional, risk = size_position(capital_inr, risk_per_trade, max_picks, entry, stop)
        names = [g.name for g, _ in sorted(firing, key=lambda t: -t[0].weight)]
        picks.append(
            Pick(
                sym, 1, entry, stop, target, tuple(g.key for g, _ in firing),
                sum(g.weight for g, _ in firing), best_g.trial_id, best_g.net_sharpe_5l, best_g.dsr, best_g.pbo,
                best_g.trades_oos, qty, notional, risk,
                f"{' + '.join(names[:3])}; {len(firing)} gated {'strategy' if len(firing) == 1 else 'strategies'}",
            )
        )  # fmt: skip
    picks.sort(key=lambda p: (-p.evidence, -p.net_sharpe_5l, p.sym))
    picks = picks[:max_picks]
    if not passing:
        note = (
            f"No strategy has passed the gate yet (n_trials_total = {total}). "
            "Exploratory candidates below are not evidence."
        )
    else:
        note = f"{len(passing)} strateg{'y' if len(passing) == 1 else 'ies'} passing the gate (n_trials_total = {total})."
    if capital_inr is None and picks:
        note += " Capital is unavailable, so the paper quantities are 0."
    result: dict[str, Any] = {
        "asof": asof,
        "gated": [asdict(p) for p in picks],
        "exploratory": _exploratory(settings, asof, sig, passing),
        "other_gated": [
            {
                "strategy": g.key,
                "name": g.name,
                "family": g.family,
                "trial": g.trial_id,
                "dsr": g.dsr,
                "pbo": g.pbo,
                "net_sharpe_5l": g.net_sharpe_5l,
                "trades_oos": g.trades_oos,
                "weight": g.weight,
                "note": "no live signal source for this family yet",
            }
            for g in other
        ],  # fmt: skip
        "n_strategies_passing": len(passing),
        "n_trials_total": total,
        "note": note,
        "footer": FOOTER,
        "paper": True,
    }
    if not persist:
        return json.loads(json.dumps(_clean(result)))
    for p in picks:
        if p.stop is None:  # a score has no stop: the journal's exit rules need one
            continue
        journal.record_signals(
            state,
            [{"state": "SIGNAL", "sym": p.sym, "p_in": p.entry_ref, "stop": p.stop, "target": p.target,
              "side": p.side, "reason": p.reason}],
            clk,
            trade_date=asof,
            strat=f"{JOB}:{p.best_trial}",
        )  # fmt: skip
    _write(path, result)
    return json.loads(path.read_text(encoding="utf-8"))


def picks_step(
    settings: Settings,
    state: State,
    clock: Clock,
    asof: date,
    capital_inr: float | None,
    **kw: Any,
) -> dict[str, Any]:
    """The catch-up step "picks" (after "scan" and "flow_eod"; not wired here): builds `asof`'s picks once.

    Idempotent by (job "picks", trading date): a date whose last attempt is `done` is skipped, a date with no
    scan is `skipped`. Returns {"status": "done" | "skipped", ...}. The caller reads `capital_inr` from the broker.
    """
    if state.job_status(JOB, asof) == "done":
        return {"status": "skipped", "reason": "already done", "asof": asof.isoformat()}
    attempt = state.job_start(JOB, asof)
    try:
        res = build_picks(settings, state, asof, capital_inr, clock=clock, **kw)
    except DataNotAvailable as e:
        state.job_finish(JOB, asof, attempt, "skipped", str(e))
        return {"status": "skipped", "reason": str(e), "asof": asof.isoformat()}
    except BaseException as e:
        state.job_finish(JOB, asof, attempt, "failed", type(e).__name__)
        raise
    state.job_finish(
        JOB, asof, attempt, "done", f"{len(res['gated'])} gated, {len(res['exploratory'])} exploratory"
    )
    return {"status": "done", "asof": asof.isoformat(), "n_gated": len(res["gated"])}


# ---- track record ----------------------------------------------------------------------------------------


def history(settings: Settings, clock: Clock, days: int = 30) -> list[dict[str, Any]]:
    """Gated picks of the last `days` days with their realised outcome from the journal (closed rows: R, net %).

    `net_pct` is the price move from the reference close to the exit, less the statutory delivery round trip on
    100,000 INR (no slippage: the journal has no fills); `r_mult` is that net move over the initial risk.
    """
    cut = clock.today() - timedelta(days=days)
    jr = {
        (r["trade_date"], r["strat"], r["sym"]): r
        for r in journal.read(settings, days=days + 10, clock=clock)["trades"]
    }
    cost = round_trip_cost_frac("delivery", 100_000.0, settings.costs)
    out: list[dict[str, Any]] = []
    root = picks_dir(settings)
    for p in sorted(root.glob("*.json"), reverse=True) if root.is_dir() else []:
        d = date.fromisoformat(p.stem)
        if d < cut:
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        for pick in doc.get("gated", []):
            j = jr.get((p.stem, f"{JOB}:{pick['best_trial']}", pick["sym"]))
            row: dict[str, Any] = {
                "asof": p.stem, "sym": pick["sym"], "strategies": pick["strategies"], "best_trial": pick["best_trial"],
                "entry_ref": pick["entry_ref"], "stop": pick["stop"], "target": pick["target"],
                "status": j["status"] if j else None, "close_price": None, "r_mult": None, "net_pct": None,
            }  # fmt: skip
            if j and j["status"] == "closed" and j["close_price"] is not None:
                gross = j["close_price"] / j["ref_price"] - 1.0
                row["close_price"] = j["close_price"]
                row["net_pct"] = 100 * (gross - cost)
                risk = (j["ref_price"] - j["stop"]) / j["ref_price"] if j["stop"] else None
                row["r_mult"] = (gross - cost) / risk if risk and risk > 0 else None
            out.append(row)
    return out
