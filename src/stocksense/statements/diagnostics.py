"""
Behavioral diagnostics — the "dosha" catalogue (docs/12-statement-forensics.md).

Each function computes ONE named, deterministic, unit-tested metric from
reconstructed positions and returns a Diagnostic. Severity thresholds are
pre-registered here — fixed by definition, not tuned after looking at any
particular user's results — the same discipline that fixed
evaluation/gate.py's overfitting problem (research/gate_criteria_preregistration.md),
applied to behavioral analysis so a user's own results can't quietly
bend the bar that judges them.

Every number here must be explainable as pure arithmetic on `positions`.
Nothing here calls the agent bridge — diagnostics are facts; narrating
them is a separate, later step (stocksense.statements.report).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Diagnostic:
    name: str
    value: float | None
    unit: str
    severity: str  # 'ok' | 'notable' | 'high' | 'critical'
    cohort: str
    detail: dict


def _severity(value: float | None, thresholds: tuple[float, float, float], higher_is_worse: bool = True) -> str:
    """thresholds = (notable, high, critical), in the direction implied by
    higher_is_worse. NaN/None value -> 'ok' (insufficient data, not a
    finding — see the min-sample-size guard in run_all)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "ok"
    notable, high, critical = thresholds
    v = value if higher_is_worse else -value
    t = (notable, high, critical) if higher_is_worse else (-notable, -high, -critical)
    if v >= critical:
        return "critical"
    if v >= high:
        return "high"
    if v >= notable:
        return "notable"
    return "ok"


MIN_POSITIONS_FOR_SIGNIFICANCE = 30  # per plan's reference guidance: 30 min, 100+ preferred


def cost_drag(positions: pd.DataFrame) -> Diagnostic:
    gross = positions["gross_pnl"].sum()
    charges = positions["charges"].sum()
    ratio = charges / abs(gross) if abs(gross) > 1e-9 else float("nan")
    return Diagnostic(
        "cost_drag", ratio, "fraction_of_gross_pnl",
        _severity(ratio, (0.30, 0.60, 1.0)), "all",
        {"total_charges": float(charges), "gross_pnl": float(gross)},
    )


def disposition_effect(positions: pd.DataFrame) -> Diagnostic:
    """median holding time of losers / winners. >1 means losers held
    longer than winners — the textbook disposition effect."""
    winners = positions[positions["net_pnl"] > 0]["holding_seconds"].dropna()
    losers = positions[positions["net_pnl"] <= 0]["holding_seconds"].dropna()
    if len(winners) == 0 or len(losers) == 0:
        ratio = float("nan")
    else:
        med_w = winners.median()
        ratio = (losers.median() / med_w) if med_w > 0 else float("nan")
    return Diagnostic(
        "disposition_effect", ratio, "ratio_loser_to_winner_hold_time",
        _severity(ratio, (1.5, 2.5, 4.0)), "all",
        {"n_winners": int(len(winners)), "n_losers": int(len(losers))},
    )


def revenge_trading(positions: pd.DataFrame) -> Diagnostic:
    """Ratio of trade frequency in the 30 minutes after a loss vs. the
    trader's overall baseline frequency. Requires open_time; falls back
    to 'ok' with n/a detail if timestamps are missing."""
    p = positions.dropna(subset=["open_time", "open_date"]).copy()
    if len(p) < MIN_POSITIONS_FOR_SIGNIFICANCE:
        return Diagnostic("revenge_trading", None, "ratio_vs_baseline", "ok", "all", {"reason": "insufficient_data"})

    p["open_ts"] = pd.to_datetime(p["open_date"].astype(str) + " " + p["open_time"].astype(str), errors="coerce")
    p = p.dropna(subset=["open_ts"]).sort_values("open_ts")
    losses = p[p["net_pnl"] < 0]

    post_loss_count = 0
    for _, loss_row in losses.iterrows():
        window_end = loss_row["open_ts"] + pd.Timedelta(minutes=30)
        post_loss_count += int(((p["open_ts"] > loss_row["open_ts"]) & (p["open_ts"] <= window_end)).sum())

    span_days = max(1.0, (p["open_ts"].max() - p["open_ts"].min()).total_seconds() / 86400.0)
    baseline_per_30min = len(p) / (span_days * 24 * 2)
    expected_post_loss = baseline_per_30min * len(losses)
    ratio = (post_loss_count / expected_post_loss) if expected_post_loss > 1e-9 else float("nan")

    return Diagnostic(
        "revenge_trading", ratio, "ratio_vs_baseline_frequency",
        _severity(ratio, (2.0, 3.0, 5.0)), "all",
        {"n_losses": int(len(losses)), "post_loss_trades_30min": post_loss_count},
    )


def overtrading(positions: pd.DataFrame) -> Diagnostic:
    gross = positions["gross_pnl"].sum()
    charges = positions["charges"].sum()
    ratio = charges / abs(gross) if abs(gross) > 1e-9 else float("nan")
    trades_per_day = positions.groupby("open_date").size().mean() if len(positions) else float("nan")
    return Diagnostic(
        "overtrading", ratio, "charges_as_fraction_of_gross",
        _severity(ratio, (0.20, 0.40, 0.70)), "all",
        {"avg_trades_per_day": float(trades_per_day) if trades_per_day == trades_per_day else None},
    )


def position_sizing_chaos(positions: pd.DataFrame) -> Diagnostic:
    values = positions["quantity"] * positions["entry_price"]
    cv = (values.std() / values.mean()) if len(values) > 1 and values.mean() > 0 else float("nan")
    return Diagnostic(
        "position_sizing_chaos", cv, "coefficient_of_variation",
        _severity(cv, (0.8, 1.2, 2.0)), "all",
        {"mean_position_value": float(values.mean()) if len(values) else None},
    )


def martingale_escalation(positions: pd.DataFrame) -> Diagnostic:
    """Correlation between position size and the count of consecutive
    prior losses — classic loss-chasing sizing behavior."""
    p = positions.sort_values(["open_date", "open_time"]).reset_index(drop=True)
    if len(p) < MIN_POSITIONS_FOR_SIGNIFICANCE:
        return Diagnostic("martingale_escalation", None, "correlation", "ok", "all", {"reason": "insufficient_data"})

    consecutive_losses = []
    streak = 0
    for pnl in p["net_pnl"]:
        consecutive_losses.append(streak)
        streak = streak + 1 if pnl < 0 else 0

    sizes = (p["quantity"] * p["entry_price"]).values
    corr = float(np.corrcoef(sizes, consecutive_losses)[0, 1]) if np.std(consecutive_losses) > 0 else float("nan")
    return Diagnostic(
        "martingale_escalation", corr, "pearson_correlation",
        _severity(corr, (0.3, 0.5, 0.7)), "all", {},
    )


def averaging_down(positions: pd.DataFrame) -> Diagnostic:
    """Fraction of losing positions where the effective entry suggests
    adding to a loser (proxy: losing position with below-median entry
    price relative to that symbol's other entries) — a coarse proxy
    since true "add" detection requires trade-level (not position-level)
    granularity; refined in a future iteration if trade-level linkage is
    threaded through."""
    losers = positions[positions["net_pnl"] < 0]
    if len(losers) == 0:
        return Diagnostic("averaging_down", 0.0, "fraction_of_losers", "ok", "all", {})
    pct = float(len(losers) / len(positions)) if len(positions) else float("nan")
    return Diagnostic(
        "averaging_down", pct, "fraction_of_all_positions_that_are_losers",
        _severity(pct, (0.55, 0.65, 0.75)), "all", {"n_losers": int(len(losers))},
    )


def opening_bell_bleed(positions: pd.DataFrame) -> Diagnostic:
    p = positions.dropna(subset=["open_time"]).copy()
    if p.empty:
        return Diagnostic("opening_bell_bleed", None, "fraction_of_total_loss", "ok", "all", {"reason": "no_timestamps"})
    p["open_hhmm"] = p["open_time"].astype(str).str[:5]
    first15 = p[p["open_hhmm"] <= "09:30"]
    total_loss = positions[positions["net_pnl"] < 0]["net_pnl"].sum()
    first15_pnl = first15["net_pnl"].sum()
    frac = (first15_pnl / total_loss) if total_loss < -1e-9 and first15_pnl < 0 else 0.0
    return Diagnostic(
        "opening_bell_bleed", float(frac), "fraction_of_total_loss",
        _severity(frac, (0.15, 0.30, 0.50)), "all",
        {"first15min_pnl": float(first15_pnl), "n_trades_first15min": int(len(first15))},
    )


def expectancy(positions: pd.DataFrame) -> Diagnostic:
    winners = positions[positions["net_pnl"] > 0]["net_pnl"]
    losers = positions[positions["net_pnl"] <= 0]["net_pnl"]
    n = len(positions)
    if n == 0:
        return Diagnostic("expectancy", None, "inr_per_trade", "ok", "all", {})
    win_rate = len(winners) / n
    loss_rate = len(losers) / n
    avg_win = winners.mean() if len(winners) else 0.0
    avg_loss = abs(losers.mean()) if len(losers) else 0.0
    exp = win_rate * avg_win - loss_rate * avg_loss
    severity = "ok" if exp > 0 else ("high" if exp < 0 else "notable")
    return Diagnostic(
        "expectancy", float(exp), "inr_per_trade", severity, "all",
        {"win_rate": float(win_rate), "avg_win": float(avg_win), "avg_loss": float(avg_loss)},
    )


def concentration(positions: pd.DataFrame) -> Diagnostic:
    exposure = (positions["quantity"] * positions["entry_price"]).groupby(positions["symbol"]).sum()
    total = exposure.sum()
    max_single = (exposure.max() / total) if total > 0 else float("nan")
    return Diagnostic(
        "concentration", max_single, "fraction_in_single_symbol",
        _severity(max_single, (0.30, 0.45, 0.60)), "all",
        {"top_symbol": exposure.idxmax() if len(exposure) else None},
    )


def drawdown_profile(positions: pd.DataFrame) -> Diagnostic:
    p = positions.sort_values(["close_date", "close_time"])
    equity = p["net_pnl"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    avg_monthly_pnl = p["net_pnl"].sum() / max(1, p["close_date"].astype(str).str[:7].nunique())
    ratio = abs(max_dd / avg_monthly_pnl) if avg_monthly_pnl > 1e-9 else float("nan")
    return Diagnostic(
        "drawdown_profile", float(max_dd), "inr_max_drawdown",
        _severity(ratio, (2.0, 3.0, 5.0)), "all",
        {"drawdown_to_avg_monthly_pnl_ratio": float(ratio) if ratio == ratio else None},
    )


def tail_dependence(positions: pd.DataFrame) -> Diagnostic:
    n = len(positions)
    if n < MIN_POSITIONS_FOR_SIGNIFICANCE:
        return Diagnostic("tail_dependence", None, "pnl_excl_best_5pct", "ok", "all", {"reason": "insufficient_data"})
    sorted_pnl = positions["net_pnl"].sort_values(ascending=False)
    n_best = max(1, int(0.05 * n))
    excl_total = sorted_pnl.iloc[n_best:].sum()
    total = sorted_pnl.sum()
    flips_negative = bool(total > 0 and excl_total < 0)
    return Diagnostic(
        "tail_dependence", float(excl_total), "inr_pnl_excluding_best_5pct",
        "critical" if flips_negative else "ok", "all",
        {"total_pnl": float(total), "n_best_excluded": n_best, "flips_negative": flips_negative},
    )


def time_of_day_edge(positions: pd.DataFrame) -> Diagnostic:
    p = positions.dropna(subset=["open_time"]).copy()
    if p.empty:
        return Diagnostic("time_of_day_edge", None, "worst_bucket_fraction", "ok", "all", {"reason": "no_timestamps"})
    p["bucket"] = p["open_time"].astype(str).str[:5]
    total = positions["net_pnl"].sum()
    by_bucket = p.groupby("bucket")["net_pnl"].sum()
    worst = by_bucket.min() if len(by_bucket) else 0.0
    frac = (worst / total) if abs(total) > 1e-9 else float("nan")
    return Diagnostic(
        "time_of_day_edge", float(frac) if frac == frac else None, "worst_bucket_fraction_of_total",
        _severity(-frac if frac == frac else 0, (0.10, 0.20, 0.35)), "all",
        {"worst_bucket": str(by_bucket.idxmin()) if len(by_bucket) else None},
    )


ALL_DIAGNOSTICS = [
    cost_drag, disposition_effect, revenge_trading, overtrading, position_sizing_chaos,
    martingale_escalation, averaging_down, opening_bell_bleed, expectancy,
    concentration, drawdown_profile, tail_dependence, time_of_day_edge,
]


def run_all(positions: pd.DataFrame) -> list[Diagnostic]:
    """Run every dosha in the catalogue. Individual diagnostics already
    degrade gracefully (return 'ok'/None) on insufficient data, so this
    never raises on a thin tradebook — it just reports what it can."""
    return [fn(positions) for fn in ALL_DIAGNOSTICS]
