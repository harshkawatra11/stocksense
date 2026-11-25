"""*151 Trading Strategies* (Kakushadze and Serur 2018) section 2 option combos on NIFTY and BANKNIFTY
monthly options, plus the section 7.4/7.5 index-option trades (PLAN-M M5.4c, family m-opt).

Every combo is entered every monthly cycle (the paper's unconditional description); only the VRP pair
adds a condition. The legs, strikes and costs are run by `options_bt.run_combo`. Section 2.52 (the box)
is an informational arbitrage check and has no trial; k151_dispersion lives in `k151_disp.py`.
The spec `timing` of an m-opt strategy returns `(date, ret)`: net daily return as a fraction of each
cycle's entry spot (zero on flat days), the m-opt convention (return on underlying notional, cash benchmark).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.strategies.base import StrategySpec, register
from stocksense.strategies.options_bt import ComboResult, EntryRule, Leg, run_combo

F, C, P = "FUT", "CE", "PE"
RV_WINDOW = 21


def L(  # noqa: N802
    kind: str, side: int, m: float = 0.0, qty: float = 1.0, expiry: str = "near", x: float | None = None
) -> Leg:
    """Shorthand leg constructor for the table below."""
    return Leg(kind, side, m, qty, expiry, x)  # type: ignore[arg-type]


# (suffix, paper section, payoff equation, legs)
COMBOS: tuple[tuple[str, str, str, tuple[Leg, ...]], ...] = (
    ("covered_call", "2.2", "1", (L(F, 1), L(C, -1, 1))),
    ("covered_put", "2.3", "5", (L(F, -1), L(P, -1, -1))),
    ("protective_put", "2.4", "9", (L(F, 1), L(P, 1))),
    ("protective_call", "2.5", "13", (L(F, -1), L(C, 1))),
    ("bull_call_spread", "2.6", "17", (L(C, 1), L(C, -1, 1))),
    ("bull_put_spread", "2.7", "21", (L(P, 1, -2), L(P, -1, -1))),
    ("bear_call_spread", "2.8", "25", (L(C, 1, 2), L(C, -1, 1))),
    ("bear_put_spread", "2.9", "29", (L(P, 1), L(P, -1, -1))),
    ("long_synth_fwd", "2.10", "33", (L(C, 1), L(P, -1))),
    ("short_synth_fwd", "2.11", "37", (L(P, 1), L(C, -1))),
    ("long_combo", "2.12", "41", (L(C, 1, 1), L(P, -1, -1))),
    ("short_combo", "2.13", "47", (L(P, 1, -1), L(C, -1, 1))),
    ("bull_call_ladder", "2.14", "53", (L(C, 1), L(C, -1, 1), L(C, -1, 2))),
    ("bull_put_ladder", "2.15", "58", (L(P, -1), L(P, 1, -1), L(P, 1, -2))),
    ("bear_call_ladder", "2.16", "63", (L(C, -1), L(C, 1, 1), L(C, 1, 2))),
    ("bear_put_ladder", "2.17", "68", (L(P, 1), L(P, -1, -1), L(P, -1, -2))),
    ("calendar_call", "2.18", "73-74", (L(C, -1), L(C, 1, expiry="next"))),
    ("calendar_put", "2.19", "75-76", (L(P, -1), L(P, 1, expiry="next"))),
    ("diagonal_call", "2.20", "77-78", (L(C, 1, -2, expiry="next"), L(C, -1, 1))),
    ("diagonal_put", "2.21", "79-80", (L(P, 1, 2, expiry="next"), L(P, -1, -1))),
    ("long_straddle", "2.22", "81", (L(C, 1), L(P, 1))),
    ("long_strangle", "2.23", "86", (L(C, 1, 1), L(P, 1, -1))),
    ("long_guts", "2.24", "91", (L(C, 1, -1), L(P, 1, 1))),
    ("short_straddle", "2.25", "96", (L(C, -1), L(P, -1))),
    ("short_strangle", "2.26", "101", (L(C, -1, 1), L(P, -1, -1))),
    ("short_guts", "2.27", "106", (L(C, -1, -1), L(P, -1, 1))),
    ("long_call_synth_straddle", "2.28", "111", (L(F, -1), L(C, 1, qty=2))),
    ("long_put_synth_straddle", "2.29", "116", (L(F, 1), L(P, 1, qty=2))),
    ("short_call_synth_straddle", "2.30", "121", (L(F, 1), L(C, -1, qty=2))),
    ("short_put_synth_straddle", "2.31", "126", (L(F, -1), L(P, -1, qty=2))),
    ("covered_short_straddle", "2.32", "131", (L(F, 1), L(C, -1), L(P, -1))),
    ("covered_short_strangle", "2.33", "135", (L(F, 1), L(C, -1, 1), L(P, -1, -1))),
    ("strap", "2.34", "138", (L(C, 1, qty=2), L(P, 1))),
    ("strip", "2.35", "143", (L(C, 1), L(P, 1, qty=2))),
    ("call_ratio_backspread", "2.36", "148", (L(C, -1), L(C, 1, 1, qty=2))),
    ("put_ratio_backspread", "2.37", "153", (L(P, -1), L(P, 1, -1, qty=2))),
    ("ratio_call_spread", "2.38", "158", (L(C, 1, -1), L(C, -1, qty=2))),
    ("ratio_put_spread", "2.39", "163", (L(P, 1, 1), L(P, -1, qty=2))),
    ("long_call_butterfly", "2.40", "168", (L(C, 1, 1), L(C, -1, qty=2), L(C, 1, -1))),
    ("mod_call_butterfly", "2.40.1", "173", (L(C, 1, 1), L(C, -1, qty=2), L(C, 1, -2))),
    ("long_put_butterfly", "2.41", "177", (L(P, 1, -1), L(P, -1, qty=2), L(P, 1, 1))),
    ("mod_put_butterfly", "2.41.1", "182", (L(P, 1, -2), L(P, -1, qty=2), L(P, 1, 1))),
    ("short_call_butterfly", "2.42", "186", (L(C, -1, -1), L(C, 1, qty=2), L(C, -1, 1))),
    ("short_put_butterfly", "2.43", "191", (L(P, -1, 1), L(P, 1, qty=2), L(P, -1, -1))),
    ("long_iron_butterfly", "2.44", "196", (L(P, 1, -1), L(P, -1), L(C, -1), L(C, 1, 1))),
    ("short_iron_butterfly", "2.45", "201", (L(P, -1, -1), L(P, 1), L(C, 1), L(C, -1, 1))),
    ("long_call_condor", "2.46", "206", (L(C, 1, -1.5), L(C, -1, -0.5), L(C, -1, 0.5), L(C, 1, 1.5))),
    ("long_put_condor", "2.47", "211", (L(P, 1, -1.5), L(P, -1, -0.5), L(P, -1, 0.5), L(P, 1, 1.5))),
    ("short_call_condor", "2.48", "216", (L(C, -1, -1.5), L(C, 1, -0.5), L(C, 1, 0.5), L(C, -1, 1.5))),
    ("short_put_condor", "2.49", "221", (L(P, -1, -1.5), L(P, 1, -0.5), L(P, 1, 0.5), L(P, -1, 1.5))),
    ("long_iron_condor", "2.50", "226", (L(P, 1, -1.5), L(P, -1, -0.5), L(C, -1, 0.5), L(C, 1, 1.5))),
    ("short_iron_condor", "2.51", "231", (L(P, -1, -1.5), L(P, 1, -0.5), L(C, 1, 0.5), L(C, -1, 1.5))),
    ("collar", "2.53", "238", (L(F, 1), L(P, 1, -1), L(C, -1, 1))),
    ("bull_short_seagull", "2.54", "242", (L(P, -1, -1), L(C, 1), L(C, -1, 1))),
    ("bear_long_seagull", "2.55", "248", (L(P, 1, -1), L(C, -1), L(C, 1, 1))),
    ("bear_short_seagull", "2.56", "254", (L(P, -1, -1), L(P, 1), L(C, -1, 1))),
    ("bull_long_seagull", "2.57", "260", (L(P, 1, -1), L(P, -1), L(C, 1, 1))),
)
UNDERLYINGS = ("NIFTY", "BANKNIFTY")
STRADDLE = (L(C, -1), L(P, -1))
RISK_REVERSAL = (L(C, 1, x=0.05), L(P, -1, x=-0.05))


def realised_vol_pct(close: np.ndarray, i: int, window: int = RV_WINDOW) -> float | None:
    """Annualised % std (ddof 1) of the `window` daily log returns ending at index i; None if too short."""
    if i < window:
        return None
    r = np.diff(np.log(close[i - window : i + 1]))
    return float(np.std(r, ddof=1) * np.sqrt(252.0) * 100.0)


def vrp_condition(dates: np.ndarray, close: np.ndarray, vix: dict[date, float]) -> Callable[[date], bool]:
    """True when INDIA VIX(d) minus NIFTY 21-session realised vol (data up to d) is positive."""
    idx = {d: i for i, d in enumerate(dates)}

    def cond(d: date) -> bool:
        i, v = idx.get(d), vix.get(d)
        if i is None or v is None:
            return False
        rv = realised_vol_pct(close, i)
        return rv is not None and v - rv > 0

    return cond


def _vrp_rule(settings: Settings, hedge: bool) -> EntryRule:
    ix = (
        io.read(settings, "index_daily")
        .filter(pl.col("index_name") == "NIFTY 50")
        .select("date", "close")
        .collect()
        .sort("date")
        .unique("date", keep="last", maintain_order=True)
    )
    vx = io.read(settings, "india_vix").select("date", "close").collect()
    cond = vrp_condition(
        np.array(ix["date"].to_list()), ix["close"].to_numpy().astype(float), dict(vx.iter_rows())
    )
    return EntryRule(condition=cond, delta_hedge=hedge)


def option_run(settings: Settings, legs: tuple[Leg, ...], und: str, vrp: str | None = None) -> ComboResult:
    """The full combo run (cycles and daily series) of `legs` on `und`, optionally with the VRP condition."""
    rule = EntryRule() if vrp is None else _vrp_rule(settings, hedge=vrp == "dh")
    return run_combo(legs, rule, settings, und)


def _clip(df: pl.DataFrame, start: date | None, end: date | None) -> pl.DataFrame:
    if start is not None:
        df = df.filter(pl.col("date") >= start)
    if end is not None:
        df = df.filter(pl.col("date") <= end)
    return df


def option_returns(
    settings: Settings,
    legs: tuple[Leg, ...],
    und: str,
    start: date | None,
    end: date | None,
    vrp: str | None = None,
) -> pl.DataFrame:
    """Daily `(date, ret)` of a combo on `und` (net of costs, fraction of entry spot), optionally clipped."""
    return _clip(option_run(settings, legs, und, vrp).daily, start, end)


def option_returns_cost(
    settings: Settings,
    legs: tuple[Leg, ...],
    und: str,
    start: date | None,
    end: date | None,
    vrp: str | None = None,
) -> pl.DataFrame:
    """Daily `(date, ret, cost)`: `ret` as in `option_returns`; `cost` is the cycle's round-trip charge
    (a fraction of entry spot, already inside `ret`) on its exit date and 0 elsewhere (the 2x-cost line)."""
    res = option_run(settings, legs, und, vrp)
    cost = res.cycles.select(pl.col("exit_date").alias("date"), "cost")
    out = res.daily.join(cost, on="date", how="left").with_columns(pl.col("cost").fill_null(0.0))
    return _clip(out, start, end)


def _reg(
    key: str, name: str, ref: str, desc: str, legs: tuple[Leg, ...], und: str, vrp: str | None = None
) -> None:
    def timing(settings: Settings, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        return option_returns(settings, legs, und, start, end, vrp)

    def with_cost(settings: Settings, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        return option_returns_cost(settings, legs, und, start, end, vrp)

    timing.with_cost = with_cost  # type: ignore[attr-defined]

    register(
        StrategySpec(
            key=key,
            name=name,
            source="k151",
            tier="none",
            family="m-opt",
            timeframe="1d",
            kind="options",
            direction="both",
            exits=("expiry",),
            default_exit="expiry",
            signal=None,
            timing=timing,
            description=desc,
            reference=ref,
        )
    )


for _suffix, _sec, _eq, _legs in COMBOS:
    _title = _suffix.replace("_", " ")
    for _und in UNDERLYINGS:
        _reg(
            f"k151_opt_{_und.lower()}_{_suffix}",
            f"151 s{_sec} {_title} ({_und})",
            f"Kakushadze and Serur (2018) s{_sec}, eq. {_eq}",
            f"{_title} on {_und} monthly options, entered each cycle at settle, held to expiry.",
            _legs,
            _und,
        )

_REF = "Kakushadze and Serur (2018) "
_reg(
    "k151_vrp_straddle",
    "151 s7.4 VRP short straddle (NIFTY)",
    _REF + "s7.4, eq. 96",
    "Short ATM straddle only when INDIA VIX minus 21-session realised vol is positive.",
    STRADDLE,
    "NIFTY",
    "plain",
)
_reg(
    "k151_vrp_straddle_dh",
    "151 s7.4.1 VRP short straddle, delta hedged (NIFTY)",
    _REF + "s7.4.1",
    "The VRP straddle plus a daily delta hedge in NIFTY near futures (BS delta from each leg IV).",
    STRADDLE,
    "NIFTY",
    "dh",
)
_reg(
    "k151_risk_reversal",
    "151 s7.5 risk reversal (NIFTY)",
    _REF + "s7.5, eq. 41",
    "Long CE at +5% and short PE at -5% of spot, every monthly cycle.",
    RISK_REVERSAL,
    "NIFTY",
)
