"""Walk-forward runner (PLAN T6.7): one registered trial, one fit per fold and seed, out-of-fold scores.

For every fold the training rows are those whose label has ended before the embargo, the inner-validation
split picks the model's complexity (`Model.fit`), the model is refit on all training rows (`refit_full`),
and the test period is scored. Each seed's scores are ranked per date to [-0.5, 0.5] and averaged. Scores
go to preds/<model_id>/ (model_id = trial_id) with mode='walkforward'; a fold listed in
preds/<model_id>/folds_done.json is not fitted again. `backtest_walk_forward` turns the out-of-fold scores
into a top-k long book through the engine and records the trial's daily net returns for gate line 8.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stocksense.backtest.engine import BookState, PanelData, run_backtest
from stocksense.backtest.fills import Order
from stocksense.backtest.metrics import ic_stats
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.costs.slippage import SlippageTable
from stocksense.data import io
from stocksense.features.base import FeatureSpec
from stocksense.features.build import current_version, load_features
from stocksense.labels.forward import _END_GROUPS, load_labels
from stocksense.models.base import Model, ModelSpec, build_model
from stocksense.validation.splits import (
    Fold,
    inner_validation_split,
    sample_weights,
    train_mask,
    walk_forward_folds,
)
from stocksense.validation.trials import finish_trial, register_trial

log = get_logger(__name__)

END_COLUMN: dict[str, str] = {y: e for ys, e in _END_GROUPS for y in ys}
_DONE_FILE = "folds_done.json"

ModelFactory = Callable[[ModelSpec], Model]


@dataclass(frozen=True)
class WFResult:
    """preds: date, sid, score (average of per-date ranks), fold; fold_metrics: one dict per fold;
    ic: out-of-fold ic_stats over all folds; trial_id doubles as the model id of preds/<model_id>/."""

    preds: pl.DataFrame
    fold_metrics: list[dict[str, Any]]
    trial_id: str
    ic: dict[str, Any]


def _preds_dir(settings: Settings, model_id: str) -> Path:
    return Paths(settings).preds / model_id


def _done_folds(settings: Settings, model_id: str) -> set[str]:
    p = _preds_dir(settings, model_id) / _DONE_FILE
    return set(json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else set()


def _mark_done(settings: Settings, model_id: str, done: set[str]) -> None:
    p = _preds_dir(settings, model_id) / _DONE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(sorted(done)), encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def _manifest(settings: Settings) -> dict[str, Any]:
    v = current_version(settings)
    if v is None:
        raise FileNotFoundError("no feature store (features/CURRENT is missing)")
    return json.loads((Paths(settings).features / f"v{v}" / "manifest.json").read_text(encoding="utf-8"))


def _rank_average(scores: dict[str, np.ndarray], dates: pl.Series, sids: pl.Series) -> pl.DataFrame:
    """Per-seed per-date rank in [-0.5, 0.5] (non-finite scores count as the median), then their mean."""
    df = pl.DataFrame({"date": dates, "sid": sids, **{k: v.astype(np.float64) for k, v in scores.items()}})
    ranks = []
    for k in scores:
        c = pl.when(pl.col(k).is_finite()).then(pl.col(k))
        n = c.count().over("date")
        r = (c.rank("average").over("date") - 1.0) / pl.max_horizontal(n - 1, pl.lit(1)).cast(pl.Float64)
        ranks.append((r - 0.5).fill_null(0.0).alias(f"__r_{k}"))
    df = df.with_columns(ranks)
    mean = pl.mean_horizontal([f"__r_{k}" for k in scores]).cast(pl.Float32).alias("score")
    return df.select("date", "sid", mean)


def _fold_ic(preds: pl.DataFrame, lab: pl.DataFrame, label_col: str) -> dict[str, Any]:
    j = preds.join(lab.select("date", "sid", label_col), on=["date", "sid"], how="inner")
    s = ic_stats(j, label_col)
    return {
        "n_rows": j.height, "n_dates": s["n_dates"], "mean_ic": s["mean_ic"], "ic_std": s["ic_std"],
        "ic_tstat": s["ic_tstat"],
    }  # fmt: skip


def run_walk_forward(
    spec: ModelSpec,
    settings: Settings,
    state: State,
    purpose: str = "research",
    seeds: Sequence[int] = (1, 2, 3),
    registry: dict[str, FeatureSpec] | None = None,
    folds: Sequence[Fold] | None = None,
    *,
    model_factory: ModelFactory | None = None,
    calendar: TradingCalendar | None = None,
    clock: Clock | None = None,
    inner_months: int = 12,
    embargo_sessions: int = 5,
) -> WFResult:
    """Register the trial, then fit/predict every fold not yet done. `registry` is accepted for symmetry
    with build_features (the feature kinds are read from the store's manifest, which that registry built);
    `folds` defaults to the 12 calendar years of walk_forward_folds."""
    del registry
    clk = clock if clock is not None else SystemClock()
    trial_id = register_trial(state, spec.family, spec.target, spec.to_dict())  # before any evaluation
    model_id = trial_id
    factory = model_factory if model_factory is not None else build_model
    cal = calendar if calendar is not None else TradingCalendar.from_store(settings, clk)
    fold_list = list(folds) if folds is not None else walk_forward_folds(settings, cal)

    label_col = f"{factory(spec).target_prefix}{spec.target}"
    end_col = END_COLUMN[spec.target]
    blocked = f"entry_blocked_{spec.entry}"

    lab = (
        load_labels(settings, purpose, columns=[label_col, end_col, blocked])
        .collect()
        .filter(pl.col(label_col).is_not_null() & ~pl.col(blocked).fill_null(False))
    )
    if spec.universe != "u500":
        uni = io.read(settings, "panel_universe", columns=["date", "sid", spec.universe]).collect()
        lab = lab.join(uni.filter(pl.col(spec.universe)).select("date", "sid"), on=["date", "sid"])

    done = _done_folds(settings, model_id)
    pending = [f for f in fold_list if f.name not in done]
    parts: dict[str, pl.DataFrame] = {}
    extra: dict[str, dict[str, Any]] = {}

    if pending:
        man = _manifest(settings)
        kind_of = {f["name"]: f["kind"] for f in man["features"]}
        names = [f["name"] for f in man["features"]] if spec.features == "all" else list(spec.features)
        kinds = [kind_of[n] for n in names]
        feats = load_features(settings, purpose, columns=names).collect()
        if spec.params.get("auction"):  # T7.2 y_oc trial: the T4.4 auction features, keyed by decision date
            from stocksense.features.auction_matrix import auction_columns

            alf, anames, akinds = auction_columns(settings, purpose)
            feats = feats.join(alf.collect(), on=["date", "sid"], how="inner")
            names, kinds = names + anames, kinds + akinds
        df = lab.join(feats, on=["date", "sid"], how="inner").sort("date", "sid")
        dates, end, sid_all = df["date"], df[end_col], df["sid"]
        dates_np = dates.to_numpy()
        y_all = df[label_col].to_numpy().astype(np.float64)
        X_all = df.select([pl.col(n).cast(pl.Float32) for n in names]).to_numpy().astype(np.float32)
        for fold in pending:
            tr = train_mask(
                end, fold, embargo_sessions, calendar=cal, dates=dates, dev_start=settings.research.dev_start
            ).to_numpy()
            te = ((dates >= fold.test_start) & (dates <= fold.test_end)).to_numpy()
            if not tr.any() or not te.any():
                reason = "no training rows" if not tr.any() else "no test rows"
                log.info("walkforward_fold_skipped", fold=fold.name, reason=reason)
                extra[fold.name] = {"skipped": reason}
                continue
            t0 = time.perf_counter()
            itr, ite = np.flatnonzero(tr), np.flatnonzero(te)
            d_tr, end_tr = dates.filter(pl.Series(tr)), end.filter(pl.Series(tr))
            inner_tr, inner_va = inner_validation_split(
                d_tr, end_tr, inner_months, embargo_sessions, calendar=cal
            )
            a, b = inner_tr.to_numpy(), inner_va.to_numpy()
            X_tr, y_tr, dn_tr = X_all[itr], y_all[itr], dates_np[itr]
            w_tr = sample_weights(d_tr, fold, cal)
            X_te = X_all[ite]
            scores: dict[str, np.ndarray] = {}
            fit_params: dict[str, Any] = {}
            for s in seeds:
                model = factory(spec)
                model.configure(names, kinds)
                info = model.fit(X_tr[a], y_tr[a], w_tr[a], dn_tr[a], X_tr[b], y_tr[b], dn_tr[b], seed=int(s))
                model.refit_full(X_tr, y_tr, w_tr, dn_tr, info, seed=int(s))
                scores[f"s{s}"] = np.asarray(model.predict(X_te))
                fit_params = info.params
            sc = _rank_average(scores, dates.filter(pl.Series(te)), sid_all.filter(pl.Series(te)))
            out = sc.with_columns(
                pl.lit(clk.now()).alias("made_at"),
                pl.lit("walkforward").alias("mode"),
                pl.lit(fold.name).alias("fold"),
            )
            io.upsert(settings, "preds", out, sub=model_id, clock=clk)
            done.add(fold.name)
            _mark_done(settings, model_id, done)
            parts[fold.name] = out.select("date", "sid", "score", "fold")
            extra[fold.name] = {
                "n_train": int(tr.sum()),
                "n_inner_val": int(b.sum()),
                "seconds": time.perf_counter() - t0,
                "fit_params": fit_params,
            }
            log.info("walkforward_fold_done", fold=fold.name, trial=trial_id)

    for fold in fold_list:
        if fold.name in parts or fold.name not in done:
            continue
        stored = io.read(settings, "preds", start=fold.test_start, end=fold.test_end, sub=model_id).collect()
        parts[fold.name] = stored.select("date", "sid", "score").with_columns(pl.lit(fold.name).alias("fold"))

    if parts:
        preds = pl.concat([parts[f.name] for f in fold_list if f.name in parts]).sort("date", "sid")
    else:
        preds = pl.DataFrame(schema={"date": pl.Date, "sid": pl.Int32, "score": pl.Float32, "fold": pl.Utf8})
    metrics: list[dict[str, Any]] = []
    for fold in fold_list:
        m: dict[str, Any] = {"fold": fold.name, **extra.get(fold.name, {})}
        if fold.name in parts:
            m.update(_fold_ic(parts[fold.name], lab, label_col))
        metrics.append(m)
    overall = ic_stats(preds.join(lab.select("date", "sid", label_col), on=["date", "sid"]), label_col)
    return WFResult(preds, metrics, trial_id, overall)


# ---- backtest of out-of-fold scores (top-k long book) ------------------------------------------------


class TopKPolicy:
    """Each decision date t: buy the k best-scored names. Target `y_oc`: intraday at the open of the next
    session (the engine squares off at the close). Target `y_c2o_<h>`: delivery at the close of the next
    session, sold at the open h sessions later, 1/h of capital per daily sleeve."""

    def __init__(self, preds: pl.DataFrame, target: str, k: int = 10, invest_frac: float = 0.95) -> None:
        self.h: int | None
        if target == "y_oc":
            self.h = None
        elif m := re.fullmatch(r"y_c2o_(\d+)", target):
            self.h = int(m.group(1))
        else:
            raise ValueError(f"no top-k policy for target {target!r}")
        self.k, self.invest = k, invest_frac
        top = (
            preds.sort(["date", "score", "sid"], descending=[False, True, False])
            .group_by("date", maintain_order=True)
            .head(k)
        )
        self._top: dict[date, list[int]] = {}
        for d, s in zip(top["date"].to_list(), top["sid"].to_list(), strict=True):
            self._top.setdefault(d, []).append(int(s))

    def decide(self, t: date, book: BookState, capital_inr: float) -> list[Order]:
        names = self._top.get(t, [])
        if not names:
            return []
        d1 = book.next_session(t, 1)
        out: list[Order] = []
        if self.h is None:
            per = capital_inr * self.invest / len(names)
            for sid in names:
                px = book.close(sid)
                if px is None or px <= 0 or int(per / px) <= 0:
                    continue
                out.append(Order(sid, "BUY", "intraday", "open", int(per / px), trade_date=d1, reason="topk"))
            return out
        sell_day = book.next_session(t, 1 + self.h)
        per = capital_inr * self.invest / self.h / len(names)
        for sid in names:
            px = book.close(sid)
            if px is None or px <= 0 or int(per / px) <= 0:
                continue
            qty = int(per / px)
            out.append(Order(sid, "BUY", "delivery", "close", qty, trade_date=d1, reason="topk_entry"))
            out.append(Order(sid, "SELL", "delivery", "open", qty, trade_date=sell_day, reason="topk_exit"))
        return out


@dataclass(frozen=True)
class TrialBacktest:
    """daily: date, net, net_2x (slippage doubled), benchmark (ew500_ret); metrics as stored on the trial."""

    daily: pl.DataFrame
    metrics: dict[str, Any]
    returns_path: Path


def returns_path(settings: Settings, trial_id: str) -> Path:
    """Where a trial's daily net returns (date, ret) are kept."""
    return Paths(settings).models / "_trials" / trial_id / "returns.parquet"


def _trial_status(state: State, trial_id: str) -> str | None:
    con = state.connect()
    try:
        row = con.execute("SELECT status FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
    finally:
        con.close()
    return None if row is None else str(row[0])


def backtest_walk_forward(
    result: WFResult,
    spec: ModelSpec,
    settings: Settings,
    state: State,
    *,
    calendar: TradingCalendar | None = None,
    clock: Clock | None = None,
    k: int = 10,
    capital_inr: float = 500_000.0,
    finish: bool = True,
) -> TrialBacktest:
    """Backtest the out-of-fold scores (top-k long, reference capital, all costs and slippage), write the
    daily net returns as parquet (date, ret) and, with `finish`, record them on the trial (once)."""
    clk = clock if clock is not None else SystemClock()
    cal = calendar if calendar is not None else TradingCalendar.from_store(settings, clk)
    preds = result.preds
    start, end = preds["date"].min(), preds["date"].max()
    assert isinstance(start, date) and isinstance(end, date)
    daily_panel = io.read(settings, "panel_daily", start=start - timedelta(days=150), end=end).collect()
    panel = PanelData.from_frame(daily_panel.sort("sid", "date"))
    slip = SlippageTable.load_latest(settings)
    runs = {}
    for name, table in (("net", slip), ("net_2x", slip.scaled(2.0))):
        runs[name] = run_backtest(
            panel, cal, TopKPolicy(preds, spec.target, k), start, end, capital_inr,
            settings.costs, table, settings.risk, debug=False,
        )  # fmt: skip
    mkt = io.read(settings, "panel_market", start=start, end=end, columns=["date", "ew500_ret"]).collect()
    daily = (
        runs["net"].daily.select("date", pl.col("day_ret").alias("net"))
        .join(runs["net_2x"].daily.select("date", pl.col("day_ret").alias("net_2x")), on="date")
        .join(mkt.rename({"ew500_ret": "benchmark"}), on="date", how="left")
        .sort("date")
    )  # fmt: skip
    net = daily["net"].to_numpy()
    sd = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    sharpe_daily = float(net.mean() / sd) if sd > 0 else 0.0
    metrics = {
        "sharpe_daily": sharpe_daily, "n_days": int(len(net)), "mean_ic": result.ic["mean_ic"],
        "ic_tstat": result.ic["ic_tstat"], "k": k, "capital_inr": capital_inr,
    }  # fmt: skip
    path = returns_path(settings, result.trial_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    daily.select("date", pl.col("net").alias("ret")).write_parquet(tmp)
    os.replace(tmp, path)
    if finish and _trial_status(state, result.trial_id) != "done":
        finish_trial(state, result.trial_id, metrics, path)
    return TrialBacktest(daily, metrics, path)
