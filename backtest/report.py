"""
Renders backtest/walkforward.py output into a plain-markdown report, printed
to stdout and written to backtest/reports/walkforward_<timestamp>.md.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtest.walkforward import EXCLUDE_YEARS_FROM_HEADLINE, HOLD_DAYS, BENCHMARK_TICKER

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)
    return float(dd.min() * 100) if not dd.empty else float("nan")


def _sharpe_like(per_window_returns: pd.Series) -> float:
    """Simple mean/std ratio over per-window net expectancy — NOT annualised
    Sharpe (windows are quarterly and trade counts vary); a straightforward
    consistency measure, documented as such rather than mislabelled."""
    if per_window_returns.empty or per_window_returns.std(ddof=0) == 0:
        return float("nan")
    return float(per_window_returns.mean() / per_window_returns.std(ddof=0))


def build_report_text(windows: pd.DataFrame, trades: pd.DataFrame, bench: dict,
                       from_year: int, to_year: int | None) -> str:
    lines = []
    lines.append("# Walk-Forward Backtest Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"OOS range requested: {from_year} -> {to_year or 'latest'}")
    lines.append("")
    lines.append(
        "**What this tests:** the deployed LightGBM BUY-signal model retrained "
        "at each quarterly checkpoint using ONLY data strictly before that "
        f"quarter, scored out-of-sample on the quarter itself. Trades use a "
        f"{HOLD_DAYS}-trading-day hold with ATR-derived stop/target "
        "(signal_pipeline.horizon_stops, 2R target), net of transaction costs "
        "(intelligence.costs.net_return, product=MIS). This is a **daily-bar "
        "swing backtest**, not an intraday backtest — the model and its "
        "features are computed from ohlcv_daily. See backtest/walkforward.py "
        "module docstring for the full methodology and its deliberate "
        "deviations from models/ml/train.py (quarterly not monthly cadence, "
        "no 5-fold CV per window)."
    )
    lines.append("")

    if windows.empty:
        lines.append("**No windows produced any output — check DB coverage / --from year.**")
        return "\n".join(lines)

    headline = windows[~windows["in_sample_flagged"]].copy()
    excluded = windows[windows["in_sample_flagged"]].copy()

    lines.append("## Headline summary (excludes 2001-2003, in-sample bias per audit)")
    lines.append("")
    total_trades = int(headline["trades"].sum())
    if total_trades > 0:
        ht = trades[trades["window"].isin(headline["window"])]
        net_rets = ht["net_return"].to_numpy()
        expectancy_bps = float(net_rets.mean() * 10000)
        winrate = float((net_rets > 0).mean())
        turnover = headline["trades"].mean()

        # Portfolio-level equity curve, NOT per-trade sequential compounding.
        # BUG FIXED (was: (1 + Series(net_rets)).cumprod() over every individual
        # trade — with ~240k trades and a slightly negative mean return, that
        # compounds as if each trade re-invests the ENTIRE account into the
        # next one, i.e. (1+r)^240522, which collapses to -100% almost
        # immediately regardless of how small the edge actually is. That's an
        # arithmetic artifact of the compounding assumption, not a real
        # portfolio result — StockSense holds at most 8 concurrent positions
        # (intelligence/portfolio_guard.py MAX_OPEN_POSITIONS), never bets
        # 100% of capital on one sequential trade chain.
        # Fix: compound ONE return per WINDOW (quarter) — each window's mean
        # net expectancy stands in for that quarter's portfolio-level return,
        # which is the right granularity for ~4-100 quarterly windows rather
        # than hundreds of thousands of individual trades. This still assumes
        # full capital turnover each quarter (a simplification — real
        # position-level compounding under the 8-position cap would differ
        # further) but no longer produces a meaningless -100% figure from
        # pure compounding arithmetic.
        window_rets = (headline["expectancy_net_bps"].dropna() / 10000.0)
        equity = (1 + window_rets).cumprod()
        maxdd = _max_drawdown(equity)
        sharpe_like = _sharpe_like(headline["expectancy_net_bps"].dropna())

        total_compounded_return = float(equity.iloc[-1] - 1) * 100 if len(equity) else float("nan")

        lines.append(f"- Windows: {len(headline)} quarterly OOS folds")
        lines.append(f"- Total trades: {total_trades}")
        lines.append(f"- Net expectancy: {expectancy_bps:+.1f} bps/trade (raw per-trade average, not compounded)")
        lines.append(f"- Win rate: {winrate*100:.1f}%")
        lines.append(f"- Avg trades/window (turnover): {turnover:.1f}")
        lines.append(f"- Max drawdown (per-WINDOW portfolio equity curve, one return/quarter): {maxdd:.1f}%")
        lines.append(f"- Sharpe-like (mean/std of per-window net expectancy): {sharpe_like:.2f}")
        lines.append(f"- Cumulative compounded return across quarterly windows (assumes full capital turnover/quarter — a simplification, see code comment): {total_compounded_return:+.1f}%")
    else:
        lines.append("- No trades fired in any non-excluded window (model/threshold too conservative, or no data).")
    lines.append("")

    lines.append(f"## Benchmark: {BENCHMARK_TICKER} buy-and-hold, same period")
    lines.append("")
    if not np.isnan(bench.get("total_return_pct", float("nan"))):
        lines.append(f"- {bench['start_date']} -> {bench['end_date']}")
        lines.append(f"- Total return: {bench['total_return_pct']:+.1f}%")
        lines.append(f"- CAGR: {bench['cagr_pct']:+.1f}%")
    else:
        lines.append("- Benchmark data unavailable for this period.")
    lines.append("")

    lines.append("## Per-window detail")
    lines.append("")
    lines.append("| Window | Train rows | OOS rows | Threshold | Trades | Net exp (bps) | Win rate | Target | Stop | Expired | Flag |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in windows.iterrows():
        flag = "IN-SAMPLE (excluded from headline)" if r["in_sample_flagged"] else ""
        exp = "" if pd.isna(r["expectancy_net_bps"]) else f"{r['expectancy_net_bps']:+.1f}"
        wr = "" if pd.isna(r["winrate"]) else f"{r['winrate']*100:.1f}%"
        thr = "" if pd.isna(r["threshold"]) else f"{r['threshold']:.3f}"
        lines.append(
            f"| {r['window']} | {r['train_rows']:,} | {r['oos_rows']:,} | {thr} | "
            f"{r['trades']} | {exp} | {wr} | {r['hit_target']} | {r['hit_stop']} | {r['expired']} | {flag} |"
        )
    lines.append("")

    if not excluded.empty:
        lines.append(
            "Note: windows flagged IN-SAMPLE above are only in-sample in the "
            "sense that the calendar years 2001-2003 are excluded from "
            "headline math per the audit finding about edge_by_year.py; each "
            "window's own model here was still trained only on data strictly "
            "before that window (genuine walk-forward), so this flag is "
            "conservative labeling, not an admission of leakage in this harness."
        )
        lines.append("")

    return "\n".join(lines)


def render_report(windows: pd.DataFrame, trades: pd.DataFrame, bench: dict,
                   from_year: int, to_year: int | None) -> str:
    text = build_report_text(windows, trades, bench, from_year, to_year)
    print(text)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"walkforward_{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[report written to {path}]")
    return path
