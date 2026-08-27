"""
Phase J2.b: the paper trading engine. One rule, stated once because it
governs everything below: this account only acts at the SAME cadence
the gate actually validated -- every `horizon_bars` trading days, via
the predictions ledger's own distinct as_of_dates as the calendar
(exactly optimizer.rebalance.recommend_todays_actions's own rebalance-
point derivation, reused here rather than re-implemented, so a paper
account can never quietly re-rank daily and generate churn nothing in
research/verdict_bhavcopy_rerun.md ever paid for).

Mirrors evaluation.backtest.simulate_portfolio's own loop shape (target
weights -> no-trade-band -> turnover cost -> drifted weights forward by
REALIZED return) rather than inventing a second accounting method --
the paper book's measured alpha must be computable the same way the
backtest's was, or the two numbers are not comparable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from stocksense.optimizer.rebalance import RebalanceAction, recommend_rebalance
from stocksense.portfolio.construct import target_weights_top_n


@dataclass(frozen=True)
class RebalanceRun:
    rebalance_date: str
    actions: list[RebalanceAction]
    period_gross_return: float
    period_charges: float
    period_net_return: float
    benchmark_return: float
    nav_units_after: float


def _rebalance_points(as_of_dates: list, horizon_bars: int) -> list:
    """Identical derivation to optimizer.rebalance.recommend_todays_
    actions: index 0, then every date at least horizon_bars entries
    later than the previous rebalance point -- but returning the FULL
    list, not just the latest, since a paper account must process every
    point it hasn't seen yet, not only "today's"."""
    if not as_of_dates:
        return []
    points = [as_of_dates[0]]
    last_idx = 0
    for i, d in enumerate(as_of_dates[1:], start=1):
        if i - last_idx >= horizon_bars:
            points.append(d)
            last_idx = i
    return points


def _weights_at(preds: pd.DataFrame, as_of_date, top_n: int) -> pd.Series:
    day = preds[preds["as_of_date"] == as_of_date].drop_duplicates(subset=["symbol"], keep="last")
    scores = day.set_index("symbol")["score"]
    return target_weights_top_n(scores, top_n)


def _close_prices(store, symbols: list[str], as_of_date) -> pd.Series:
    """The actual tradeable close on `as_of_date` for `symbols`, read
    directly from bhavcopy_eq -- the raw print, not adjusted, since a
    paper fill pays the real quoted price. Symbols this table has no row
    for on this exact date come back absent from the Series (NOT zero),
    so callers can tell 'no price' from 'price is zero' apart."""
    if not symbols:
        return pd.Series(dtype=float)
    placeholders = ", ".join(["?"] * len(symbols))
    df = store.con.execute(
        f"SELECT symbol, close FROM bhavcopy_eq WHERE symbol IN ({placeholders}) AND date = ? AND series = 'EQ'",
        [*symbols, as_of_date],
    ).fetchdf()
    return df.set_index("symbol")["close"]


def run_pending_rebalances(store, account, segment: str = "equity_delivery") -> list[RebalanceRun]:
    """Processes every rebalance point the ledger has produced since
    this account was last stepped forward -- idempotent by construction
    (store.rebalance_dates_recorded is the account's own high-water
    mark), so calling this nightly alongside the reconcile loop just
    picks up wherever it left off. Returns one RebalanceRun per newly
    processed point, empty if there was nothing new."""
    preds = store.read_predictions()
    preds = preds[(preds["model_version"] == account.model_id) & (preds["horizon_bars"] == account.horizon_bars)]
    if preds.empty:
        return []

    as_of_dates = sorted(preds["as_of_date"].unique())
    points = _rebalance_points(as_of_dates, account.horizon_bars)
    already_done = set(store.rebalance_dates_recorded(account.account_id))
    pending = [p for p in points if p not in already_done]
    if not pending:
        return []

    nav_row = store.read_paper_daily_nav(account.account_id)
    nav_units = float(nav_row["nav_units"].iloc[-1]) if not nav_row.empty else 1.0
    cum_return = float(nav_row["cum_return"].iloc[-1]) if not nav_row.empty else 0.0
    benchmark_nav_units = float(nav_row["benchmark_nav_units"].iloc[-1]) if not nav_row.empty else 1.0
    benchmark_cum_return = float(nav_row["benchmark_cum_return"].iloc[-1]) if not nav_row.empty else 0.0

    open_positions = store.read_paper_positions(account.account_id, status="open")
    current_weights = (
        pd.Series(open_positions["weight"].values, index=open_positions["symbol"].values)
        if not open_positions.empty else pd.Series(dtype=float)
    )
    # entry price for the CURRENTLY open leg of each held symbol, used to
    # compute this period's realized return on it
    entry_prices = (
        pd.Series(open_positions["entry_price"].values, index=open_positions["symbol"].values)
        if not open_positions.empty else pd.Series(dtype=float)
    )

    runs: list[RebalanceRun] = []
    all_points_sorted = points  # for benchmark universe at each date
    for point in sorted(pending, key=lambda p: as_of_dates.index(p)):
        target = _weights_at(preds, point, account.top_n)
        actions = recommend_rebalance(
            target, current_weights, portfolio_value_inr=1.0, band=account.no_trade_band, segment=segment,
        )

        symbols_needed = sorted(set(target.index) | set(current_weights.index))
        prices_now = _close_prices(store, symbols_needed, point)

        # Realized return on whatever was ALREADY held coming into this
        # rebalance, priced from its own entry to today -- this is the
        # period return the NAV actually earns; the fresh entries below
        # start earning from THIS point forward, not this period.
        period_gross = 0.0
        for symbol, w in current_weights.items():
            if symbol not in prices_now.index or symbol not in entry_prices.index:
                continue
            entry_p = entry_prices[symbol]
            if entry_p is None or entry_p <= 0:
                continue
            period_gross += float(w) * (float(prices_now[symbol]) / float(entry_p) - 1.0)

        period_charges = sum(a.estimated_cost_inr for a in actions)  # notional=1.0, so this is a fraction
        period_net = period_gross - period_charges

        # Benchmark: equal-weight mean return of the SAME cross-section
        # the model was scored on this date -- the same definition
        # labels.add_relative_forward_return anchors "relative" against,
        # so paper alpha and backtest alpha are the same quantity.
        bench_syms = [s for s in target.index if s in prices_now.index]
        # approximate benchmark return using price change since the
        # PRIOR rebalance point (or since account inception for the
        # first point) -- the same period this rebalance's return covers
        prior_point = all_points_sorted[all_points_sorted.index(point) - 1] if all_points_sorted.index(point) > 0 else None
        if prior_point is not None and bench_syms:
            prior_prices = _close_prices(store, bench_syms, prior_point)
            common = [s for s in bench_syms if s in prior_prices.index and prior_prices[s] > 0]
            bench_ret = float(pd.Series({s: prices_now[s] / prior_prices[s] - 1.0 for s in common}).mean()) if common else 0.0
        else:
            bench_ret = 0.0

        nav_units = nav_units * (1.0 + period_net)
        cum_return = nav_units - 1.0
        benchmark_nav_units = benchmark_nav_units * (1.0 + bench_ret)
        benchmark_cum_return = benchmark_nav_units - 1.0

        now = datetime.now(timezone.utc)
        order_rows = []
        for a in actions:
            price = float(prices_now[a.symbol]) if a.symbol in prices_now.index else None
            fill_status = "filled" if price is not None else "rejected"
            rejection_reason = None if price is not None else "no_price_on_rebalance_date"
            order_rows.append({
                "order_id": str(uuid.uuid4())[:12], "account_id": account.account_id,
                "rebalance_date": point, "symbol": a.symbol, "action": a.action,
                "current_weight": a.current_weight, "target_weight": a.target_weight,
                "weight_delta": a.weight_delta, "fill_rule": account.fill_rule,
                "fill_price": price, "fill_status": fill_status, "rejection_reason": rejection_reason,
                "charges_fraction": a.estimated_cost_inr, "model_id": account.model_id, "created_at": now,
            })
        store.write_paper_orders(pd.DataFrame(order_rows))

        position_rows = []
        for a in actions:
            price = float(prices_now[a.symbol]) if a.symbol in prices_now.index else None
            if a.action in ("exit", "trim") and a.symbol in entry_prices.index:
                open_date = open_positions.loc[open_positions["symbol"] == a.symbol, "open_date"].iloc[0]
                entry_p = float(entry_prices[a.symbol])
                exit_gross = (price / entry_p - 1.0) if (price is not None and entry_p > 0) else None
                position_rows.append({
                    "account_id": account.account_id, "symbol": a.symbol, "open_date": open_date,
                    "close_date": point if a.action == "exit" else None,
                    "weight": a.target_weight if a.action == "trim" else a.current_weight,
                    "entry_price": entry_p, "exit_price": price if a.action == "exit" else None,
                    "gross_return": exit_gross if a.action == "exit" else None,
                    "charges_fraction": a.estimated_cost_inr if a.action == "exit" else None,
                    "net_return": (exit_gross - a.estimated_cost_inr) if (a.action == "exit" and exit_gross is not None) else None,
                    "status": "closed" if a.action == "exit" else "open",
                    "open_order_id": order_rows[0]["order_id"], "close_order_id": None,
                })
            elif a.action in ("enter", "add") and price is not None:
                open_date = point if a.action == "enter" else open_positions.loc[open_positions["symbol"] == a.symbol, "open_date"].iloc[0]
                position_rows.append({
                    "account_id": account.account_id, "symbol": a.symbol, "open_date": open_date,
                    "close_date": None, "weight": a.target_weight, "entry_price": price if a.action == "enter" else float(entry_prices.get(a.symbol, price)),
                    "exit_price": None, "gross_return": None, "charges_fraction": None, "net_return": None,
                    "status": "open", "open_order_id": order_rows[0]["order_id"], "close_order_id": None,
                })
        if position_rows:
            store.upsert_paper_positions(pd.DataFrame(position_rows))

        store.upsert_paper_daily_nav({
            "account_id": account.account_id, "date": point, "nav_units": nav_units,
            "daily_return": period_net, "cum_return": cum_return,
            "benchmark_nav_units": benchmark_nav_units, "benchmark_daily_return": bench_ret,
            "benchmark_cum_return": benchmark_cum_return, "n_positions": int((target > 0).sum()),
            "drawdown": None,
        })

        runs.append(RebalanceRun(
            rebalance_date=str(point), actions=actions, period_gross_return=period_gross,
            period_charges=period_charges, period_net_return=period_net, benchmark_return=bench_ret,
            nav_units_after=nav_units,
        ))

        # roll forward for the next pending point in this same call
        current_weights = target
        entry_prices = pd.Series({s: (float(prices_now[s]) if s in prices_now.index else entry_prices.get(s)) for s in target.index})
        open_positions = store.read_paper_positions(account.account_id, status="open")

    return runs
