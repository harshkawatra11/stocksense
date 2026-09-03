"""Corporate-action-adjusted prices.

NSE bhavcopy carries RAW prices. A 1:10 split prints as a 90% one-day collapse,
and a 1:1 bonus as a 50% one. Feed that to a feature engine and it learns that
splits are catastrophes; feed it to a backtest and it books a fictional loss.

## The arithmetic

For a symbol with ex-dates e1 < e2 < ... < en and price factors f1 ... fn, the
cumulative back-adjustment for a price observed on date d is

    cum(d) = product of { fi : ei > d }

i.e. multiply by every factor whose ex-date is still in the future. Prices at or
after the last ex-date are untouched (cum = 1.0), so the series is anchored to
today's scale and never needs restating as new actions arrive.

Worked: a 1:1 bonus (f = 0.5) on ex-date E. The raw print is 1000 before E and
500 on E. cum(d < E) = 0.5 gives 1000 -> 500; cum(d >= E) = 1.0 leaves 500. The
adjusted series is continuous, which is the whole point.

## Two things this module refuses to do, both learned the hard way

**1. It does not use bhavcopy's own `prev_close` to compute returns.** NSE does
not adjust that column, so on a split ex-date `open / prev_close - 1` is a
fabricated -50%. `with_prev_adjusted_close` derives the previous close from the
ADJUSTED series instead.

**2. It does not flag anomalies by comparing `adj_close` to raw `close`.** That
check is only meaningful when the source is already split-adjusted. Applied to
bhavcopy it fires on every genuine corporate action, and in a previous build it
quarantined RELIANCE, TCS, INFY, HDFCBANK and ~600 other names -- the most
liquid ones, because large caps are the most likely to have split in 16 years.
That does not merely lose data; it inverts survivorship bias into something
worse, selecting for "never had a corporate action".

The correct check, implemented here, is: an adjusted-price jump with **no
matching corporate-action record**, measured only across genuinely CONSECUTIVE
trading days.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# A day-over-day move beyond this, on consecutive trading days, with no
# corporate action to explain it, is treated as a data error rather than a
# market event. Indian circuit bands are 2/5/10/20%, so 35% cannot be a single
# session's legitimate move on a liquid name.
DEFAULT_JUMP_THRESHOLD = 0.35

# How far either side of a price jump to look for an explaining ex-date. NSE's
# announced ex-date and the day the price actually gaps can differ by a day or
# two around holidays.
DEFAULT_EXPLAIN_WINDOW_DAYS = 7

PRICE_COLUMNS = ("open", "high", "low", "close")


def cumulative_factors(
    corporate_actions: pd.DataFrame,
    basis: str = "price",
) -> pd.DataFrame:
    """Per-symbol step function of cumulative adjustment factors.

    Args:
        corporate_actions: as produced by `corporate_actions.parse_records`.
        basis: "price" uses splits/bonuses only. "total" additionally reinvests
            dividends, giving a total-return series.

    Returns:
        Frame [symbol, ex_date, cum_factor] where `cum_factor` applies to every
        price STRICTLY BEFORE that ex_date. Empty if nothing adjusts.
    """
    if basis not in ("price", "total"):
        raise ValueError("basis must be 'price' or 'total'")
    if corporate_actions.empty:
        return pd.DataFrame(columns=["symbol", "ex_date", "cum_factor"])

    ca = corporate_actions.copy()
    ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    ca["factor_price"] = pd.to_numeric(ca["factor_price"], errors="coerce").fillna(1.0)

    if basis == "total":
        # A dividend does not change the share count, so factor_price is 1.0 and
        # it is invisible on a price-return basis. On a total-return basis the
        # cash is reinvested, which is a separate multiplicative factor.
        # NOTE: this needs the pre-ex close to be exact; `dividend_total_factors`
        # computes it. Here we only carry the split/bonus part, and the dividend
        # part is merged in by `adjusted_prices` where prices are available.
        pass

    ca = ca[ca["factor_price"] != 1.0]
    if ca.empty:
        return pd.DataFrame(columns=["symbol", "ex_date", "cum_factor"])

    # One row per (symbol, ex_date): several actions can share an ex-date and
    # they compose multiplicatively.
    per_date = (
        ca.groupby(["symbol", "ex_date"], as_index=False)["factor_price"]
        .prod()
        .sort_values(["symbol", "ex_date"])
    )

    # cum(d) = product of factors with ex_date > d. Walking each symbol's events
    # from the LAST backwards, the running product at event i is exactly the
    # factor that applies to any date before ei.
    per_date["cum_factor"] = (
        per_date.iloc[::-1]
        .groupby("symbol")["factor_price"]
        .cumprod()
        .iloc[::-1]
    )
    return per_date[["symbol", "ex_date", "cum_factor"]].reset_index(drop=True)


def apply_factors(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Attach `cum_factor` to each price row and produce adjusted columns.

    Uses a FORWARD as-of join with exact matches excluded, which is precisely
    "the nearest ex-date strictly after this price date". Rows after every
    ex-date match nothing and take 1.0.
    """
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"])

    if factors.empty:
        out["cum_factor"] = 1.0
    else:
        left = out.sort_values("date")
        right = factors.sort_values("ex_date")
        merged = pd.merge_asof(
            left,
            right.rename(columns={"ex_date": "date"}),
            on="date",
            by="symbol",
            direction="forward",
            allow_exact_matches=False,
        )
        merged["cum_factor"] = merged["cum_factor"].fillna(1.0)
        out = merged

    for col in PRICE_COLUMNS:
        if col in out.columns:
            out[f"adj_{col}"] = out[col] * out["cum_factor"]
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def adjusted_prices(
    reader,
    symbols: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    basis: str = "price",
) -> pd.DataFrame:
    """Adjusted OHLC for a symbol/date range.

    ALL FOUR price columns are adjusted, not just close. This matters
    specifically for the overnight/intraday decomposition: `adj_open` against the
    previous session's `adj_close` is the only way to measure an overnight return
    across a split ex-date without inventing a 50% gap.

    Raw `open/high/low/close` are retained alongside, because candlestick shape
    features (body %, wick %, true range) describe the bar as actually printed.
    """
    prices = reader.bhavcopy_eq(symbols=symbols, start=start, end=end)
    if prices.empty:
        return prices

    ca = reader.corporate_actions(symbols=symbols)
    factors = cumulative_factors(ca, basis=basis)
    out = apply_factors(prices, factors)

    if basis == "total":
        out = _apply_dividend_reinvestment(out, ca)
    return out


def _apply_dividend_reinvestment(prices: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    """Fold dividends into the adjusted series for a total-return basis.

    A dividend of D against a pre-ex close of C is a factor of (C - D) / C
    applied to everything before the ex-date. C is taken from the ADJUSTED close
    on the last trading day before the ex-date, so splits and dividends compose
    correctly in either order.

    Dividends are the weakest link in this whole module -- the amounts are free
    text, and "special" versus "ordinary" is ambiguous -- which is exactly why
    the price basis is computed alongside and is the default.
    """
    if ca.empty or prices.empty:
        return prices

    divs = ca[(ca["action_type"] == "dividend") & ca["dividend_amount"].notna()].copy()
    if divs.empty:
        return prices

    divs["ex_date"] = pd.to_datetime(divs["ex_date"])
    divs = divs.groupby(["symbol", "ex_date"], as_index=False)["dividend_amount"].sum()

    # Pre-ex adjusted close: the last adjusted close strictly before the ex-date.
    prior = pd.merge_asof(
        divs.sort_values("ex_date").rename(columns={"ex_date": "date"}),
        prices.sort_values("date")[["symbol", "date", "adj_close"]],
        on="date",
        by="symbol",
        direction="backward",
        allow_exact_matches=False,
    ).rename(columns={"date": "ex_date", "adj_close": "pre_ex_close"})

    prior = prior[prior["pre_ex_close"].notna() & (prior["pre_ex_close"] > 0)]
    if prior.empty:
        return prices

    prior["factor_price"] = (
        (prior["pre_ex_close"] - prior["dividend_amount"]) / prior["pre_ex_close"]
    ).clip(lower=0.0)
    prior = prior[prior["factor_price"] != 1.0]
    if prior.empty:
        return prices

    div_factors = cumulative_factors(
        prior.assign(parse_status="ok")[["symbol", "ex_date", "factor_price"]], basis="price"
    )
    out = prices.drop(columns=[c for c in prices.columns if c.startswith("adj_")] + ["cum_factor"])
    combined = apply_factors(out, div_factors)
    # Compose with the split/bonus factors already applied.
    combined["cum_factor"] = combined["cum_factor"] * prices["cum_factor"].to_numpy()
    for col in PRICE_COLUMNS:
        if col in combined.columns:
            combined[f"adj_{col}"] = combined[col] * combined["cum_factor"]
    return combined


def with_prev_adjusted_close(
    prices: pd.DataFrame,
    calendar: list[date] | pd.Series | None = None,
) -> pd.DataFrame:
    """Add `prev_adj_close` and `prev_gap_sessions`.

    THE trap this exists to avoid: `groupby(symbol).shift(1)` is NOT "yesterday".
    An illiquid NSE name can skip days entirely, so its previous ROW may be weeks
    earlier -- and treating that as a one-day return fabricates enormous moves.
    A previous build hit this with INDOTECH, whose apparent jump was a multi-week
    gap.

    `prev_gap_sessions` is the number of TRADING sessions between the two rows,
    computed against the real market calendar. Consumers wanting a genuine
    overnight return must filter to `prev_gap_sessions == 1`.
    """
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    if calendar is None:
        sessions = pd.Index(sorted(out["date"].unique()))
    else:
        sessions = pd.Index(sorted(pd.to_datetime(pd.Series(list(calendar))).unique()))

    session_no = pd.Series(np.arange(len(sessions)), index=sessions)
    out["_sess"] = out["date"].map(session_no)

    grp = out.groupby("symbol", sort=False)
    out["prev_adj_close"] = grp["adj_close"].shift(1)
    out["prev_gap_sessions"] = out["_sess"] - grp["_sess"].shift(1)
    return out.drop(columns=["_sess"])


def flag_unexplained_jumps(
    prices: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
    explain_window_days: int = DEFAULT_EXPLAIN_WINDOW_DAYS,
    calendar: list[date] | None = None,
) -> pd.DataFrame:
    """Adjusted-price jumps with no corporate action to explain them.

    Deliberately NOT `adj_close vs close` -- see the module docstring. A jump WITH
    a matching record is the adjustment layer working correctly and passes
    through untouched; a jump with NO record is either an unparsed action (a
    demerger, a rights issue) or a genuine data error, and is what we want.

    Restricted to CONSECUTIVE trading sessions, because a large move across a
    three-week gap in an illiquid name is not a one-day jump.
    """
    if prices.empty:
        return pd.DataFrame(columns=["symbol", "date", "ret", "adj_close"])

    df = with_prev_adjusted_close(prices, calendar)
    df = df[(df["prev_gap_sessions"] == 1) & df["prev_adj_close"].gt(0)]
    if df.empty:
        return pd.DataFrame(columns=["symbol", "date", "ret", "adj_close"])

    df = df.assign(ret=df["adj_close"] / df["prev_adj_close"] - 1.0)
    candidates = df[df["ret"].abs() > jump_threshold]
    if candidates.empty:
        return pd.DataFrame(columns=["symbol", "date", "ret", "adj_close"])

    if corporate_actions.empty:
        return candidates[["symbol", "date", "ret", "adj_close"]].reset_index(drop=True)

    ca = corporate_actions.copy()
    ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    window = pd.Timedelta(days=explain_window_days)

    explained = np.zeros(len(candidates), dtype=bool)
    by_symbol = {s: g["ex_date"].to_numpy() for s, g in ca.groupby("symbol")}
    for i, (sym, d) in enumerate(zip(candidates["symbol"], candidates["date"], strict=True)):
        ex_dates = by_symbol.get(sym)
        if ex_dates is None or len(ex_dates) == 0:
            continue
        explained[i] = bool(np.any(np.abs(ex_dates - np.datetime64(d)) <= window))

    return (
        candidates.loc[~explained, ["symbol", "date", "ret", "adj_close"]]
        .reset_index(drop=True)
        .sort_values(["symbol", "date"])
    )


def quarantine_unexplained(
    prices: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    **kwargs,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop every row of any symbol carrying an unexplained jump.

    Coarse but safe: a symbol with one unexplained discontinuity is not
    trustworthy anywhere in its history, because we do not know what else the
    same missing action distorted.
    """
    bad = flag_unexplained_jumps(prices, corporate_actions, **kwargs)
    symbols = sorted(bad["symbol"].unique().tolist()) if not bad.empty else []
    clean = prices[~prices["symbol"].isin(symbols)].copy() if symbols else prices.copy()
    return clean, symbols
