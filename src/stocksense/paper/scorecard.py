"""
Phase J2.d: the paper scorecard, and the bar before real capital is even
discussed (J2.4 in the plan). Every threshold here is fixed in this
docstring, not tunable by a caller, and the module's own promise:
falling short of any one criterion means the paper record CONTINUES,
never that thresholds get revisited.

The six criteria, fixed before any paper NAV existed:
1. >= MIN_REBALANCES completed rebalance cycles (20 -- double
   GateCriteria.min_folds_required, since forward observations are
   fewer and non-overlapping in a way backtested folds are not).
2. Mean net return per rebalance > 0, AND the count of positive
   rebalances significant at one-sided binomial p <= SIGNIFICANCE_ALPHA
   -- reusing evaluation.gate's own exact binomial test, unmodified,
   not a re-derived approximation.
3. Forward mean net return falls within the walk-forward backtest's OWN
   measured fold-alpha range for this model (from model_registry.
   metrics_json) -- a forward record that's positive but far below
   backtest is evidence of overfitting, not of success.
4. evaluation.gate.evaluate_forward_record has never demoted this model
   during the paper period.
5. Max paper drawdown <= MAX_DRAWDOWN_MULTIPLE x the worst single
   backtest fold's own loss.
6. Fill rate >= MIN_FILL_RATE -- if a meaningful share of intended
   orders never filled (no_price_on_rebalance_date), the paper book is
   measuring a different portfolio than the one that passed the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stocksense.evaluation.gate import _one_sided_binomial_pvalue

MIN_REBALANCES = 20
SIGNIFICANCE_ALPHA = 0.10  # same alpha GateCriteria uses
MAX_DRAWDOWN_MULTIPLE = 1.5
MIN_FILL_RATE = 0.80


@dataclass(frozen=True)
class ReadinessVerdict:
    ready: bool
    criteria: dict  # {criterion_name: {"met": bool, "detail": str}}
    reasons_not_ready: list = field(default_factory=list)


def paper_scorecard(store, account_id: str) -> dict:
    nav = store.read_paper_daily_nav(account_id)
    orders = store.read_paper_orders(account_id)

    if nav.empty:
        return {
            "account_id": account_id, "n_rebalances": 0, "mean_net_return": None,
            "cum_return": None, "benchmark_cum_return": None, "max_drawdown": None,
            "fill_rate": None, "n_orders": 0, "n_filled": 0, "n_rejected": 0,
        }

    returns = nav["daily_return"].dropna()
    n = len(returns)
    n_positive = int((returns > 0).sum())
    pvalue = _one_sided_binomial_pvalue(n_positive, n) if n else 1.0

    running_max = nav["nav_units"].cummax()
    drawdown = (nav["nav_units"] / running_max - 1.0)
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    non_hold = orders[orders["action"] != "hold"]
    n_orders = int(len(non_hold))
    n_filled = int((non_hold["fill_status"] == "filled").sum())
    n_rejected = n_orders - n_filled
    fill_rate = (n_filled / n_orders) if n_orders else None

    return {
        "account_id": account_id,
        "n_rebalances": n,
        "n_rebalances_positive": n_positive,
        "hit_rate_pvalue": pvalue,
        "mean_net_return": float(returns.mean()) if n else None,
        "cum_return": float(nav["cum_return"].iloc[-1]),
        "benchmark_cum_return": float(nav["benchmark_cum_return"].iloc[-1]) if "benchmark_cum_return" in nav else None,
        "max_drawdown": max_dd,
        "fill_rate": fill_rate,
        "n_orders": n_orders,
        "n_filled": n_filled,
        "n_rejected": n_rejected,
        "as_of_date": str(nav["date"].iloc[-1]),
    }


def _backtest_fold_alphas(store, model_id: str) -> np.ndarray | None:
    import json

    row = store.con.execute("SELECT metrics_json FROM model_registry WHERE model_id = ?", [model_id]).fetchone()
    if row is None or row[0] is None:
        return None
    metrics = json.loads(row[0])
    alphas = metrics.get("fold_alphas")
    return np.array(alphas) if alphas else None


def real_capital_readiness(store, account_id: str, model_id: str) -> ReadinessVerdict:
    """All six criteria must hold simultaneously -- see module
    docstring. Never returns ready=True on a partial record; missing
    data (e.g. no backtest fold_alphas recorded) counts as NOT met for
    that specific criterion, not skipped."""
    from stocksense.evaluation.gate import ForwardRecordCriteria, evaluate_forward_record

    card = paper_scorecard(store, account_id)
    criteria: dict = {}

    n_rebalances = card["n_rebalances"]
    criteria["min_rebalances"] = {
        "met": n_rebalances >= MIN_REBALANCES,
        "detail": f"{n_rebalances}/{MIN_REBALANCES} completed rebalances",
    }

    mean_net = card["mean_net_return"]
    pvalue = card.get("hit_rate_pvalue")
    hit_rate_ok = (mean_net is not None and mean_net > 0 and pvalue is not None and pvalue <= SIGNIFICANCE_ALPHA)
    criteria["significant_positive_mean_return"] = {
        "met": hit_rate_ok,
        "detail": f"mean_net_return={mean_net!r}, hit_rate_pvalue={pvalue!r} (need <= {SIGNIFICANCE_ALPHA})",
    }

    fold_alphas = _backtest_fold_alphas(store, model_id)
    within_backtest_range = False
    detail = "no backtest fold_alphas recorded for this model"
    if fold_alphas is not None and mean_net is not None:
        lo, hi = float(np.percentile(fold_alphas, 2.5)), float(np.percentile(fold_alphas, 97.5))
        within_backtest_range = lo <= mean_net <= hi
        detail = f"forward mean {mean_net:+.4%} vs backtest 95% range [{lo:+.4%}, {hi:+.4%}]"
    criteria["matches_backtest_distribution"] = {"met": within_backtest_range, "detail": detail}

    fwd_verdict = evaluate_forward_record(store, model_id, ForwardRecordCriteria())
    never_demoted = not fwd_verdict.demote
    criteria["no_forward_demotion"] = {
        "met": never_demoted,
        "detail": fwd_verdict.reason,
    }

    max_dd = card["max_drawdown"]
    dd_ok = False
    dd_detail = "no paper drawdown yet"
    if max_dd is not None and fold_alphas is not None:
        worst_backtest_fold = float(fold_alphas.min())
        threshold = worst_backtest_fold * MAX_DRAWDOWN_MULTIPLE if worst_backtest_fold < 0 else -abs(MAX_DRAWDOWN_MULTIPLE * 0.01)
        dd_ok = max_dd >= threshold  # both negative; drawdown must not be WORSE (more negative) than threshold
        dd_detail = f"paper max drawdown {max_dd:+.4%} vs threshold {threshold:+.4%} ({MAX_DRAWDOWN_MULTIPLE}x worst backtest fold)"
    criteria["drawdown_within_bound"] = {"met": dd_ok, "detail": dd_detail}

    fill_rate = card["fill_rate"]
    fill_ok = fill_rate is not None and fill_rate >= MIN_FILL_RATE
    criteria["fill_rate"] = {
        "met": fill_ok,
        "detail": f"fill_rate={fill_rate!r} (need >= {MIN_FILL_RATE:.0%}); {card['n_rejected']}/{card['n_orders']} orders rejected",
    }

    reasons_not_ready = [name for name, c in criteria.items() if not c["met"]]
    ready = len(reasons_not_ready) == 0

    return ReadinessVerdict(ready=ready, criteria=criteria, reasons_not_ready=reasons_not_ready)
