"""Gate G1 research run (PLAN T9.5): the portfolio-trial grid, references, candidates and the verdict.

The pipeline is resumable: every trial writes `reports/g1/trials/<trial_id>/` (result.json, daily.parquet,
trades/entries at the reference capital) and is finished on the trial registry only after those files exist,
so a second call re-runs nothing that is done. Every trial is registered BEFORE any backtest of the grid
runs. Nothing here changes a threshold or adds a grid point: the grids are the constants below.

Reading order of this module: configuration and survivor discovery -> the score source (stored walk-forward
predictions blended online, expected-return map) -> the backtest world (panel, events, market state) ->
trial planning/registration/running -> strict capture for S2 -> candidate selection, gate, lockbox runner,
status -> the orchestrator `run_g1` (added with the renderer).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from stocksense.backtest.engine import BookState, PanelData, run_backtest
from stocksense.backtest.fills import Order
from stocksense.backtest.metrics import ic_stats, perf_stats, trade_stats
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.costs.slippage import SlippageTable
from stocksense.data import io
from stocksense.features.build import load_features
from stocksense.features.events import CA_STRUCTURAL, RESULT_RE
from stocksense.labels.forward import load_labels
from stocksense.models.base import ModelSpec
from stocksense.models.ensemble import (
    blend_online,
    expected_return_map,
    expected_return_map_decile,
    rank_scores,
    smooth_scores,
)
from stocksense.portfolio.construct import (
    DayScores,
    EnginePanelAccess,
    NameRow,
    PanelDay,
    S1Policy,
    S2Policy,
    horizon_of,
)
from stocksense.portfolio.meta import MetaModel, build_meta_samples, size_multiplier, veto
from stocksense.validation.lockbox import research_end
from stocksense.validation.splits import Fold, walk_forward_folds
from stocksense.validation.trials import finish_trial, register_trial
from stocksense.validation.walkforward import END_COLUMN, ModelFactory, returns_path

log = get_logger(__name__)

# ---- the grids of the card (never extended) --------------------------------------------------------

S1_TARGETS = ("y_c2o_5", "y_c2o_10", "y_c2o_20")
S2_TARGET = "y_oc"
S1_K_MAX = (10, 20)
S1_OVERLAY = (False, True)
S2_K_MAX = (5, 10)
FAMILY_S1, FAMILY_S2, FAMILY_REF = "g1-s1", "g1-s2", "g1-ref"
G1_VERSION = 1
BARS_INTERVAL = "FIVE_MINUTE"
STRICT_ENTRY_BAR = time(9, 15)  # the bar that starts at 09:15; its close is the entry price
STRICT_EXIT_BAR = time(15, 10)  # the bar that starts at 15:10; its close is the exit price
SINGLE_FACTORS: tuple[tuple[str, float], ...] = (("mom_252_skip21", 1.0), ("vol_21", -1.0))


@dataclass(frozen=True)
class G1Config:
    """Run parameters. `None` fields take their value from the settings (start = wf_test_start, end = the
    research end, capitals = the cost grid, reference capital = gate.toml). The rest are test hooks."""

    start: date | None = None
    end: date | None = None
    capitals: tuple[float, ...] | None = None
    reference_capital: float | None = None
    slippage_stress: float = 2.0
    folds: tuple[Fold, ...] | None = None
    seeds: tuple[int, ...] = (1, 2, 3)
    inner_months: int = 12
    embargo_sessions: int = 5
    min_survivor_ic: float = 0.0
    model_factory: ModelFactory | None = None
    meta_factory: Callable[[], MetaModel] | None = None
    s1_targets: tuple[str, ...] = S1_TARGETS
    # backtest-only S1 variant (DECISIONS 2026-11-19): re-enter N sessions after the book-drawdown stop fired
    s1_auto_resume: int | None = None
    # G2-C2: causal EMA span applied to the blended score, and the shape of the expected-return map
    score_ema: int | None = None
    er_map: Literal["linear", "decile", "decile_js"] = "linear"
    # G2-C3: S1 universe membership follows the u500 hysteresis flag of panel_universe, not rank <= 500
    universe_flag: bool = False
    # G2-C5: S1Policy(policy_opts=) for the g2-s1cost grid, and the trial family S1 plans register under
    s1_policy: Mapping[str, Any] | None = None
    family_s1: str = FAMILY_S1

    def resolved(self, settings: Settings) -> G1Config:
        caps = self.capitals if self.capitals is not None else tuple(settings.costs.grid.capital_inr)
        return replace(
            self,
            start=self.start or settings.research.wf_test_start,
            end=self.end or research_end(settings),
            capitals=tuple(float(c) for c in caps),
            reference_capital=float(
                self.reference_capital
                if self.reference_capital is not None
                else settings.gate.research.reference_capital_inr
            ),
        )


# ---- paths and small helpers -----------------------------------------------------------------------


def g1_dir(settings: Settings) -> Path:
    return Paths(settings).reports / "g1"


def trial_dir(settings: Settings, trial_id: str) -> Path:
    return g1_dir(settings) / "trials" / trial_id


def _clean(x: Any) -> Any:
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python, dates -> ISO text."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, date):
        return x.isoformat()
    return x


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _write_parquet(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


# ---- survivors: the models whose scores are blended ------------------------------------------------


@dataclass(frozen=True)
class ModelTrial:
    """A finished walk-forward model trial (its out-of-fold scores are in preds/<trial_id>/)."""

    trial_id: str
    family: str
    target: str
    spec: dict[str, Any]
    mean_ic: float
    ic_tstat: float


def model_trials(state: State, strategy: str, target: str) -> list[dict[str, Any]]:
    """Every trial of strategy prefix `strategy` ('s1', 's2') for `target`: id, family, status, metrics."""
    con = state.connect()
    try:
        rows = con.execute(
            "SELECT trial_id, family, status, spec_json, metrics_json FROM trials "
            "WHERE target=? AND family LIKE ? ORDER BY trial_id",
            (target, f"{strategy}-%"),
        ).fetchall()
    finally:
        con.close()
    out = []
    for tid, fam, status, spec_json, metrics_json in rows:
        m = json.loads(metrics_json) if metrics_json else {}
        out.append(
            {
                "trial_id": tid, "family": fam, "status": status,
                "spec": json.loads(spec_json) if spec_json else None,
                "mean_ic": m.get("mean_ic"), "ic_tstat": m.get("ic_tstat"),
            }
        )  # fmt: skip
    return out


def surviving_models(state: State, strategy: str, target: str, min_ic: float = 0.0) -> list[ModelTrial]:
    """Finished walk-forward trials of the strategy and target whose out-of-fold mean IC is above `min_ic`.
    Fixed rule, set before any result is read: a model with no positive out-of-fold IC does not enter the
    blend (the online weights would give it the floor weight anyway)."""
    out = []
    for r in model_trials(state, strategy, target):
        ic = r["mean_ic"]
        if r["status"] != "done" or r["spec"] is None or ic is None or not math.isfinite(ic) or ic <= min_ic:
            continue
        t = r["ic_tstat"]
        out.append(
            ModelTrial(
                r["trial_id"],
                r["family"],
                target,
                r["spec"],
                float(ic),
                float(t) if t is not None else float("nan"),
            )  # fmt: skip
        )
    return out


# ---- the score source: stored out-of-fold predictions, blended online ------------------------------


def load_oof(settings: Settings, trial_id: str, end: date | None = None) -> pl.DataFrame:
    """date, sid, score of a model trial's stored walk-forward predictions (up to `end`)."""
    return (
        io.read(settings, "preds", end=end, columns=["date", "sid", "score"], sub=trial_id)
        .collect()
        .sort("date", "sid")
    )


def load_label_frame(settings: Settings, target: str, purpose: str = "research") -> pl.DataFrame:
    """date, sid, raw `target` and its end-date column, through the purpose guard."""
    return load_labels(settings, purpose, columns=[target, END_COLUMN[target]]).collect()


@dataclass(frozen=True)
class Blended:
    """frame: date, sid, score, exp_ret, disagreement; weights: online weights by date; ic: out-of-fold IC."""

    frame: pl.DataFrame
    weights: pl.DataFrame
    ic: dict[str, Any]


def _disagreement(frames: Mapping[str, pl.DataFrame]) -> pl.DataFrame | None:
    if len(frames) < 2:
        return None
    wide: pl.DataFrame | None = None
    for name, f in frames.items():
        r = rank_scores(f.select("date", "sid", "score")).rename({"score": name})
        wide = r if wide is None else wide.join(r, on=["date", "sid"], how="inner")
    assert wide is not None
    return wide.select("date", "sid", pl.concat_list(list(frames)).list.std(ddof=1).alias("disagreement"))


def _with_er(blend: pl.DataFrame, labels: pl.DataFrame, target: str, er_map: str = "linear") -> pl.DataFrame:
    if er_map == "decile":
        er = expected_return_map_decile(blend, labels, target)
    elif er_map == "decile_js":
        er = expected_return_map_decile(blend, labels, target, shrink="james_stein")
    elif er_map == "linear":
        er = expected_return_map(blend, labels, target)
    else:
        raise ValueError(f"unknown er_map {er_map!r}")
    return er.with_columns(pl.col("exp_ret").fill_nan(None))


def blend_models(
    frames: Mapping[str, pl.DataFrame],
    labels: pl.DataFrame,
    target: str,
    *,
    score_ema: int | None = None,
    er_map: str = "linear",
) -> Blended:
    """The online blend (causal), the expected-return map and the models' disagreement for one target.
    `score_ema` smooths the blended score (causal EMA, re-ranked) before the map; `er_map` picks the map."""
    blend, weights = blend_online(dict(frames), labels, target)
    if score_ema is not None:
        blend = smooth_scores(blend, score_ema)
    er = _with_er(blend, labels, target, er_map)
    dis = _disagreement(frames)
    out = er if dis is None else er.join(dis, on=["date", "sid"], how="left")
    if "disagreement" not in out.columns:
        out = out.with_columns(pl.lit(0.0).alias("disagreement"))
    out = out.with_columns(pl.col("disagreement").fill_null(0.0).fill_nan(0.0))
    ic = ic_stats(blend.join(labels.select("date", "sid", target), on=["date", "sid"], how="inner"), target)
    return Blended(out, weights, ic)


def factor_scores(
    settings: Settings, feature: str, sign: float, labels: pl.DataFrame, target: str
) -> Blended:
    """A single factor as a 'blend': its cross-sectional rank (times `sign`) with the same
    expected-return map."""
    f = (
        load_features(settings, "research", columns=[feature])
        .collect()
        .drop_nulls(feature)
        .select("date", "sid", (pl.col(feature).cast(pl.Float64) * sign).alias("score"))
    )
    er = _with_er(f, labels, target).with_columns(pl.lit(0.0).alias("disagreement"))
    ic = ic_stats(f.join(labels.select("date", "sid", target), on=["date", "sid"], how="inner"), target)
    return Blended(er, pl.DataFrame(), ic)


class StoredScoreSource:
    """`ScoreSource` over a blended frame. With `calendar` and `trade_date=True` (S2) a request for trade
    date d is answered with the scores of the previous session (the decision date of the fit)."""

    def __init__(
        self, frame: pl.DataFrame, *, calendar: TradingCalendar | None = None, trade_date: bool = False
    ) -> None:
        self._by = {k[0]: g for k, g in frame.partition_by("date", as_dict=True).items()}
        self._cal, self._shift = calendar, trade_date
        if trade_date and calendar is None:
            raise ValueError("trade_date=True needs a calendar")

    def scores(self, d: date) -> DayScores | None:
        key = d
        if self._shift:
            assert self._cal is not None
            try:
                key = self._cal.prev_session(d)
            except ValueError:
                return None
        g = self._by.get(key)
        if g is None:
            return None
        sids = g["sid"].to_list()
        nan = float("nan")
        return DayScores(
            dict(zip(sids, g["score"].to_list(), strict=True)),
            dict(zip(sids, g["exp_ret"].fill_null(nan).to_list(), strict=True)),
            dict(zip(sids, g["disagreement"].to_list(), strict=True)),
        )

    @property
    def dates(self) -> list[date]:
        return sorted(self._by)


# ---- the backtest world ----------------------------------------------------------------------------


class _FnoAccess(EnginePanelAccess):
    """EnginePanelAccess whose `in_fno` flag follows the per-date F&O membership of panel_universe."""

    def __init__(self, *a: Any, fno_by_date: Mapping[date, set[int]], **kw: Any) -> None:
        super().__init__(*a, **kw)
        self._fno_by_date = fno_by_date

    def day(self, t: date, next_date: date) -> PanelDay:
        pd_ = super().day(t, next_date)
        members = self._fno_by_date.get(t)
        if not members:
            return pd_
        rows: dict[int, NameRow] = {
            s: (replace(r, in_fno=True) if s in members else r) for s, r in pd_.rows.items()
        }
        return PanelDay(pd_.date, pd_.next_date, rows, pd_.breadth_200, pd_.mkt_ret_21, pd_.p_bear)


@dataclass
class G1World:
    """Everything a backtest of the grid reads, built once: panel, calendar, slippage, events, state."""

    settings: Settings
    calendar: TradingCalendar
    slippage: SlippageTable
    panel: PanelData
    start: date
    end: date
    industry: dict[int, str]
    split_dates: dict[int, list[date]]
    result_dates: dict[int, list[date]]
    market: dict[date, tuple[float | None, float | None]]
    stale: set[tuple[date, int]]
    fno_by_date: dict[date, set[int]]
    benchmark: pl.DataFrame  # date, ew500_ret, nifty500_ret
    u500_by_date: dict[date, set[int]] = field(default_factory=dict)  # panel_universe.u500 members
    _volfc: tuple[np.ndarray, np.ndarray] | None = None
    _p_bear: dict[date, float] | None = None
    _es: np.ndarray | None = None

    def es_array(self) -> np.ndarray:
        """Per panel row GPD expected-shortfall multiplier (G2-E6): the monthly pooled-by-tercile table
        built from rows before each month (causal), cached on disk; NaN where a name has none."""
        if self._es is None:
            from stocksense.core.paths import Paths
            from stocksense.portfolio import evt

            p = self.panel
            path = evt.cache_path(Paths(self.settings).root, self.end, self.start, len(set(p.sid.tolist())))
            if path.exists():
                table = pl.read_parquet(path)
            else:
                table = evt.es_mult_table(
                    p.date, p.sid, p.ret, p.vol_21, p.rank, from_date=self.start, to_date=self.end
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                table.write_parquet(path)
            self._es = evt.align_es_mult(table, p.sid, p.date)
        return self._es

    def regime_p_bear(self) -> dict[date, float]:
        """date -> filtered HMM bear probability (G2-E4), computed on demand and causally (each year's
        parameters come from data before that year; the filter uses data up to the day)."""
        if self._p_bear is None:
            from stocksense.features import regime

            self._p_bear = regime.p_bear_by_date(self.settings, self.end)
        return self._p_bear

    def vol_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(HAR, GARCH) daily-vol forecasts aligned to the panel rows (G2-E1), computed on demand from the
        causal vol_panel of the world's names (cached on disk); NaN where a name has no forecast."""
        if self._volfc is None:
            from stocksense.features import volforecast as vf

            sids = sorted({int(s) for s in self.panel.sid})
            frame = vf.vol_panel(self.settings, self.end, sids, fit_from=self.start - timedelta(days=40))
            self._volfc = vf.align_daily_vol(self.panel.sid, self.panel.date, frame)
        return self._volfc

    def s1_access(
        self,
        universe_flag: bool = False,
        vol_model: str = "vol21",
        regime: bool = False,
        stop_model: str = "fixed",
    ) -> EnginePanelAccess:
        return EnginePanelAccess(
            self.panel, self.calendar, industry=self.industry, split_dates=self.split_dates,
            result_dates=self.result_dates, market=self.market, stale=self.stale,
            u500=self.u500_by_date if universe_flag else None, universe_flag=universe_flag,
            volfc=self.vol_arrays() if vol_model != "vol21" else None,
            regime=self.regime_p_bear() if regime else None,
            es=self.es_array() if stop_model == "evt" else None,
        )  # fmt: skip

    def s2_access(self) -> EnginePanelAccess:
        return _FnoAccess(
            self.panel, self.calendar, industry=self.industry, split_dates=self.split_dates,
            result_dates=self.result_dates, market=self.market, stale=self.stale,
            fno_by_date=self.fno_by_date,
        )  # fmt: skip


def _event_maps(
    settings: Settings, daily: pl.DataFrame, end: date | None
) -> tuple[dict[int, list[date]], dict[int, list[date]]]:
    """sid -> sorted ex-dates of structural corporate actions, sid -> sorted results board-meeting dates.
    Dates after `end` are not read (a research world never sees the lockbox period). Static lists: the
    14-day look-ahead of EnginePanelAccess uses them regardless of when they were announced, which can
    only make the book avoid a name slightly more often than it could have known to."""
    sym = daily.sort("date").select("symbol", "sid").unique(subset="symbol", keep="last")
    splits: dict[int, list[date]] = {}
    results: dict[int, list[date]] = {}
    try:
        ca = io.read(settings, "corporate_actions", columns=["symbol", "ex_date", "action_type"]).collect()
        if end is not None:
            ca = ca.filter(pl.col("ex_date") <= end)
        ca = ca.filter(pl.col("action_type").is_in(list(CA_STRUCTURAL))).join(sym, on="symbol")
        for sid, g in ca.group_by("sid"):
            splits[int(sid[0])] = sorted(set(g["ex_date"].to_list()))
    except (DataNotAvailable, KeyError, pl.exceptions.PolarsError) as e:  # empty or missing dataset
        log.info("g1_no_corporate_actions", error=str(e))
    try:
        bm = io.read(settings, "board_meetings", columns=["symbol", "meeting_date", "purpose", "description"])
        bm = bm.collect().filter(
            pl.col("purpose").fill_null("").str.contains(RESULT_RE)
            | pl.col("description").fill_null("").str.contains(RESULT_RE)
        )
        if end is not None:
            bm = bm.filter(pl.col("meeting_date") <= end)
        for sid, g in bm.join(sym, on="symbol").group_by("sid"):
            results[int(sid[0])] = sorted(set(g["meeting_date"].to_list()))
    except (DataNotAvailable, KeyError, pl.exceptions.PolarsError) as e:
        log.info("g1_no_board_meetings", error=str(e))
    return splits, results


def build_world(
    settings: Settings,
    calendar: TradingCalendar,
    *,
    start: date,
    end: date,
    slippage: SlippageTable | None = None,
    cut_events: bool = True,
) -> G1World:
    """Load the panel for [start - 150 days, end] and everything the policies read. Only names that are ever
    a universe member are kept; a row outside panel_universe gets turnover rank 9999 (never a universe
    name, priced with the beyond-table slippage)."""
    lo = start - timedelta(days=150)
    daily = io.read(settings, "panel_daily", start=lo, end=end).collect()
    uni = io.read(
        settings, "panel_universe", start=lo, end=end,
        columns=["date", "sid", "turnover_rank", "u500", "u200", "in_fno"],
    ).collect()  # fmt: skip
    ever = uni.filter(pl.col("u500") | pl.col("u200"))["sid"].unique()
    daily = daily.filter(pl.col("sid").is_in(ever.implode())).sort("sid", "date")
    panel = PanelData.from_frame(daily, uni.select("date", "sid", "turnover_rank"))
    member = (
        pl.DataFrame({"date": panel.date, "sid": panel.sid.astype(np.int32)})
        .join(
            uni.select("date", "sid").with_columns(pl.lit(True).alias("_u")), on=["date", "sid"], how="left"
        )
        .get_column("_u")
        .fill_null(False)
        .to_numpy()
    )
    panel.rank = np.where(member, panel.rank, 9999).astype(np.int64)
    industry = {
        int(r["sid"]): r["industry"] or "UNKNOWN"
        for r in daily.group_by("sid").agg(pl.col("industry").last()).to_dicts()
    }
    stale_rows = daily.filter(pl.col("flag_stale").fill_null(False)).select("date", "sid")
    stale = set(zip(stale_rows["date"].to_list(), stale_rows["sid"].to_list(), strict=True))
    u500_by_date: dict[date, set[int]] = {}
    u_rows = uni.filter(pl.col("u500").fill_null(False)).select("date", "sid")
    for d, s in zip(u_rows["date"].to_list(), u_rows["sid"].to_list(), strict=True):
        u500_by_date.setdefault(d, set()).add(int(s))
    fno: dict[date, set[int]] = {}
    f_rows = uni.filter(pl.col("in_fno").fill_null(False)).select("date", "sid")
    for d, s in zip(f_rows["date"].to_list(), f_rows["sid"].to_list(), strict=True):
        fno.setdefault(d, set()).add(int(s))
    mk = io.read(
        settings, "panel_market", start=lo, end=end,
        columns=["date", "ew500_ret", "nifty500_ret", "breadth_200"],
    ).collect().sort("date")  # fmt: skip
    mk21 = (pl.col("ew500_ret").fill_null(0.0) + 1.0).log().rolling_sum(21)
    mk = mk.with_columns(mk21.alias("_m21"))
    market: dict[date, tuple[float | None, float | None]] = {}
    for d, b, m in zip(mk["date"].to_list(), mk["breadth_200"].to_list(), mk["_m21"].to_list(), strict=True):
        market[d] = (
            None if b is None or not math.isfinite(b) else float(b),
            None if m is None or not math.isfinite(m) else float(m),
        )
    splits, results = _event_maps(settings, daily, end if cut_events else None)
    return G1World(
        settings, calendar, slippage or SlippageTable.load_latest(settings), panel, start, end, industry,
        splits, results, market, stale, fno, mk.select("date", "ew500_ret", "nifty500_ret"),
        u500_by_date,
    )  # fmt: skip


# ---- policies and wrappers -------------------------------------------------------------------------


class WalkForwardMeta:
    """MetaLike backed by one `MetaModel` per fold, each fitted on the trades that CLOSED strictly before
    that fold's test start (so nothing from the test window leaks). `set_date(t)` picks the fold of the
    decision date; outside every fold the layer is neutral (NaN: multiplier 1, no veto)."""

    def __init__(
        self,
        samples: pl.DataFrame,
        folds: Sequence[Fold],
        factory: Callable[[], MetaModel] = MetaModel,
        seed: int = 0,
    ) -> None:
        self.folds = sorted(folds, key=lambda f: f.test_start)
        self.models: list[MetaModel] = []
        for f in self.folds:
            m = factory()
            m.fit(samples, f.test_start, seed=seed)
            self.models.append(m)
        self._cur: MetaModel | None = None

    def set_date(self, t: date) -> None:
        self._cur = None
        for f, m in zip(self.folds, self.models, strict=True):
            if f.test_start <= t <= f.test_end:
                self._cur = m
                return

    def predict_rows(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        if self._cur is None:
            return np.full(len(rows), np.nan)
        return self._cur.predict_rows(rows)

    def size_multiplier(self, p: float) -> float:
        return size_multiplier(p)

    def veto(self, p: float) -> bool:
        return veto(p)


class _PolicyWrapper:
    """Delegates to an S1Policy (so `entry_log` and `plans` stay reachable) and transforms around decide()."""

    def __init__(self, base: Any) -> None:
        self.base = base

    @property
    def entry_log(self) -> list[dict[str, Any]]:
        return self.base.entry_log

    def decide(self, t: date, book: BookState, capital_inr: float) -> list[Order]:
        return self.base.decide(t, book, capital_inr)


class _MetaDated(_PolicyWrapper):
    def __init__(self, base: S1Policy, meta: WalkForwardMeta) -> None:
        super().__init__(base)
        self.meta = meta

    def decide(self, t: date, book: BookState, capital_inr: float) -> list[Order]:
        self.meta.set_date(t)
        return self.base.decide(t, book, capital_inr)


class _OpenEntry(_PolicyWrapper):
    """The same S1 decisions with the entry at the OPEN of the next session instead of its close: the
    reference that quantifies the entry-time effect (B.3 Finding 3)."""

    def decide(self, t: date, book: BookState, capital_inr: float) -> list[Order]:
        return [
            replace(o, when="open") if (o.side == "BUY" and o.product == "delivery") else o
            for o in self.base.decide(t, book, capital_inr)
        ]


# ---- trial planning --------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialPlan:
    """One portfolio trial to register and run. kind: s1, s1-meta, s2, ref-factor, ref-open."""

    kind: str
    family: str
    target: str
    spec: dict[str, Any]
    label: str
    trial_id: str | None = None

    def with_id(self, trial_id: str) -> TrialPlan:
        return replace(self, trial_id=trial_id)


def _spec_s1(
    cfg: G1Config, target: str, k_max: int, overlay: bool, models: Sequence[ModelTrial], **extra: Any
) -> dict[str, Any]:
    return {
        "g1": G1_VERSION, "strategy": "s1", "target": target, "k_max": k_max, "overlay": overlay,
        "blend": "online", "models": sorted(m.trial_id for m in models), "entry": "close",
        "start": cfg.start, "end": cfg.end, "capitals": list(cfg.capitals or ()),
        "reference_capital": cfg.reference_capital, "stress": cfg.slippage_stress,
        **({"auto_resume_sessions": cfg.s1_auto_resume} if cfg.s1_auto_resume is not None else {}),
        **({"score_ema": cfg.score_ema} if cfg.score_ema is not None else {}),
        **({"er_map": cfg.er_map} if cfg.er_map != "linear" else {}),
        **({"universe_flag": True} if cfg.universe_flag else {}),
        **({"policy": dict(cfg.s1_policy)} if cfg.s1_policy else {}),
        **extra,
    }  # fmt: skip


def _spec_s2(cfg: G1Config, k_max: int, models: Sequence[ModelTrial]) -> dict[str, Any]:
    return {
        "g1": G1_VERSION, "strategy": "s2", "target": S2_TARGET, "k_max": k_max, "blend": "online",
        "models": sorted(m.trial_id for m in models), "entry": "open", "auction_assumption": True,
        "start": cfg.start, "end": cfg.end, "capitals": list(cfg.capitals or ()),
        "reference_capital": cfg.reference_capital, "stress": cfg.slippage_stress,
    }  # fmt: skip


def plan_grid(
    cfg: G1Config, s1_models: Mapping[str, Sequence[ModelTrial]], s2_models: Sequence[ModelTrial]
) -> list[TrialPlan]:
    """The card's grid: for each S1 target with surviving models k_max {10, 20} x overlay {off, on}, and the
    two S2 trials. A target without surviving models contributes nothing (the report says so)."""
    plans: list[TrialPlan] = []
    for target in cfg.s1_targets:
        models = s1_models.get(target, ())
        if not models:
            continue
        for k in S1_K_MAX:
            for ov in S1_OVERLAY:
                plans.append(
                    TrialPlan(
                        "s1",
                        cfg.family_s1,
                        target,
                        _spec_s1(cfg, target, k, ov, models),
                        f"S1 {target} k={k} overlay={'on' if ov else 'off'}",
                    )  # fmt: skip
                )
    if s2_models and cfg.s1_auto_resume is None:  # the auto-resume variant concerns S1 only
        for k in S2_K_MAX:
            plans.append(TrialPlan("s2", FAMILY_S2, S2_TARGET, _spec_s2(cfg, k, s2_models), f"S2 k={k}"))
    return plans


@dataclass(frozen=True)
class S1CostPlan:
    """One trial of the g2-s1cost grid: the variant name, the G1Config that produced its spec and the plan."""

    variant: str
    cfg: G1Config
    plan: TrialPlan


FAMILY_S1COST = "g2-s1cost"
S1COST_TARGETS = ("y_c2o_10", "y_c2o_20")


def s1cost_variants(h: int) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """The eight variants of card C5: name -> (policy options, G1Config overrides). h = target horizon."""
    v1 = {"exit_rank": 50, "min_hold": h // 2}
    v2 = {**v1, "free_extension": True, "rebuy_block": 5}
    v5 = {**v2, "switch_hurdle": True}
    return {
        "V1": (v1, {}),
        "V2": (v2, {}),
        "V3": (v2, {"score_ema": 3}),
        "V4": (v2, {"score_ema": 5}),
        "V5": (v5, {"score_ema": 5}),
        "V6": (v5, {"score_ema": 5, "er_map": "decile"}),
        "V7": ({**v5, "exit_rank": 75}, {"score_ema": 5, "er_map": "decile"}),
        "V8": ({**v5, "max_fixed_cost_bps": 10}, {"score_ema": 5, "er_map": "decile"}),
    }


S1COST_E_TARGETS = ("y_c2o_10",)


def s1cost_e_variants(h: int) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """G2-E grid: the C5 winner V8 and five single-change variants (name -> (policy options, G1Config
    overrides)). E0 is V8 itself (same spec, so the same trial id as the C5 trial)."""
    opts, over = s1cost_variants(h)["V8"]
    return {
        "E0": (opts, over),
        "E1": ({**opts, "vol_model": "har"}, over),
        "E2": ({**opts, "vol_model": "garch"}, over),
        "E3": (opts, {**over, "er_map": "decile_js"}),
        "E4": ({**opts, "sizing": "kelly", "kelly_c": 0.25}, over),
        "E5": ({**opts, "sizing": "kelly", "kelly_c": 0.5}, over),
    }


def s1cost_e2_variants(h: int) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """G2-E4/E6 portfolio-option grid on the E3 winner (V8 + er_map decile_js): R0 is that winner itself (same
    spec, same trial id `g2-s1cost-y_c2o_10-39cae29708`), R1 adds the regime overlay (gross x 0.5 when the
    filtered bear probability is above 0.6), R2 the extreme-value stop, R3 both."""
    opts, over = s1cost_variants(h)["V8"]
    over = {**over, "er_map": "decile_js"}
    return {
        "R0": (opts, over),
        "R1": ({**opts, "regime_overlay": True}, over),
        "R2": ({**opts, "stop_model": "evt"}, over),
        "R3": ({**opts, "regime_overlay": True, "stop_model": "evt"}, over),
    }


def plan_s1cost(
    cfg: G1Config,
    s1_models: Mapping[str, Sequence[ModelTrial]],
    *,
    k_max: int,
    overlay: bool,
    auto_resume: int = 20,
    targets: Sequence[str] = S1COST_TARGETS,
    variants: Callable[[int], dict[str, tuple[dict[str, Any], dict[str, Any]]]] | None = None,
) -> list[S1CostPlan]:
    """Card C5: 8 variants x targets on the fixed GBM scores (base: k_max/overlay of the best G1 S1 trial,
    auto-resume 20, hysteresis universe flag). A target without models contributes nothing."""
    out: list[S1CostPlan] = []
    for target in targets:
        models = s1_models.get(target, ())
        if not models:
            continue
        for name, (opts, over) in (variants or s1cost_variants)(horizon_of(target)).items():
            vcfg = replace(
                cfg, family_s1=FAMILY_S1COST, s1_auto_resume=auto_resume, universe_flag=True,
                s1_policy=opts, **over,
            )  # fmt: skip
            plan = TrialPlan(
                "s1", FAMILY_S1COST, target, _spec_s1(vcfg, target, k_max, overlay, models),
                f"S1 cost {name} {target} k={k_max}",
            )  # fmt: skip
            out.append(S1CostPlan(name, vcfg, plan))
    return out


def register_plans(state: State, plans: Sequence[TrialPlan]) -> list[TrialPlan]:
    """Register every plan on the trial registry (idempotent: a repeated spec returns its trial id)."""
    return [p.with_id(register_trial(state, p.family, p.target, p.spec)) for p in plans]


# ---- running one trial -----------------------------------------------------------------------------


def _stats(res: Any, capital: float, years: float) -> dict[str, Any]:
    d = res.daily
    s = perf_stats(d["day_ret"].to_numpy(), None, dates=d["date"].to_list()) if d.height else {}
    g = perf_stats(d["ret_gross"].to_numpy(), None) if d.height else {}
    paid = float(d["costs_inr"].sum() + d["slip_inr"].sum()) if d.height else 0.0
    t = res.trades
    return {
        "capital": capital, "n_days": d.height,
        "sharpe": s.get("sharpe"), "cagr": s.get("cagr"), "vol": s.get("vol"),
        "max_drawdown": s.get("max_drawdown"), "frac_positive_years": s.get("frac_positive_years"),
        "net_return": (float(d["nav"][-1]) / 100.0 - 1.0) if d.height else None,
        "gross_sharpe": g.get("sharpe"),
        "mean_turnover": float(d["turnover"].mean()) if d.height else None,
        "mean_gross": float(d["gross"].mean()) if d.height else None,
        "cost_drag_total": paid / capital,
        "cost_drag_annual": paid / capital / years if years > 0 else None,
        "n_trades": t.height,
        "avg_trade_cost_frac": float(t["cost_frac"].mean()) if t.height else None,
        "trade_stats": trade_stats(t, res.rejects) if t.height else None,
    }  # fmt: skip


@dataclass
class TrialResult:
    """What a finished trial left on disk (result.json) plus its path."""

    trial_id: str
    data: dict[str, Any]
    dir: Path

    @property
    def ref_sharpe(self) -> float:
        v = self.data.get("ref_sharpe")
        return float("nan") if v is None else float(v)

    def daily(self) -> pl.DataFrame:
        return pl.read_parquet(self.dir / "daily.parquet")


def load_result(settings: Settings, trial_id: str) -> TrialResult | None:
    d = trial_dir(settings, trial_id)
    p = d / "result.json"
    if not p.is_file() or not (d / "daily.parquet").is_file():
        return None
    return TrialResult(trial_id, json.loads(p.read_text(encoding="utf-8")), d)


class G1Runner:
    """Plans, registers and runs the portfolio trials. `current_trial` names the trial whose backtest is
    running (a test hook for the register-before-run assertion)."""

    def __init__(
        self,
        settings: Settings,
        state: State,
        calendar: TradingCalendar,
        cfg: G1Config | None = None,
        *,
        clock: Clock | None = None,
        world: G1World | None = None,
    ) -> None:
        self.settings, self.state, self.calendar = settings, state, calendar
        self.cfg = (cfg or G1Config()).resolved(settings)
        self.clock = clock or SystemClock()
        self._world = world
        self._labels: dict[str, pl.DataFrame] = {}
        self._blends: dict[tuple[Any, ...], Blended] = {}
        self.current_trial: str | None = None

    # -- shared inputs
    @property
    def world(self) -> G1World:
        if self._world is None:
            assert self.cfg.start is not None and self.cfg.end is not None
            self._world = build_world(self.settings, self.calendar, start=self.cfg.start, end=self.cfg.end)
        return self._world

    def labels(self, target: str) -> pl.DataFrame:
        if target not in self._labels:
            self._labels[target] = load_label_frame(self.settings, target, "research")
        return self._labels[target]

    def blended(self, target: str, models: Sequence[str]) -> Blended:
        key = (target, tuple(sorted(models)), self.cfg.score_ema, self.cfg.er_map)
        if key not in self._blends:
            end = research_end(self.settings)
            frames = {m: load_oof(self.settings, m, end) for m in sorted(models)}
            self._blends[key] = blend_models(
                frames, self.labels(target), target, score_ema=self.cfg.score_ema, er_map=self.cfg.er_map
            )
        return self._blends[key]

    # -- policy factories (a policy is stateful: one per run)
    def _s1_factory(
        self,
        target: str,
        source: StoredScoreSource,
        k_max: int,
        overlay: bool,
        *,
        meta: WalkForwardMeta | None = None,
        entry: str = "close",
        auto_resume: int | None = None,
        policy_opts: Mapping[str, Any] | None = None,
    ) -> Callable[[SlippageTable], Any]:
        world, settings = self.world, self.settings
        spec = ModelSpec(family="s1-g1", target=target)
        vol_model = str((policy_opts or {}).get("vol_model", "vol21"))
        regime = bool((policy_opts or {}).get("regime_overlay", False))
        stop_model = str((policy_opts or {}).get("stop_model", "fixed"))

        def make(slip: SlippageTable) -> Any:
            base = S1Policy(
                spec, settings, source,
                world.s1_access(self.cfg.universe_flag, vol_model, regime, stop_model),
                settings.costs, slip,
                k_max=k_max, overlay=overlay, meta=meta, state=None, auto_resume_sessions=auto_resume,
                policy_opts=policy_opts,
            )  # fmt: skip
            pol: Any = _MetaDated(base, meta) if meta is not None else _PolicyWrapper(base)
            return _OpenEntry(pol) if entry == "open" else pol

        return make

    def _s2_factory(self, source: StoredScoreSource, k_max: int) -> Callable[[SlippageTable], Any]:
        world, settings = self.world, self.settings
        return lambda slip: S2Policy(settings, source, world.s2_access(), slip, k_max=k_max)

    def factory_for(self, plan: TrialPlan) -> Callable[[SlippageTable], Any]:
        sp = plan.spec
        if plan.kind == "s2":
            b = self.blended(plan.target, sp["models"])
            return self._s2_factory(
                StoredScoreSource(b.frame, calendar=self.calendar, trade_date=True), sp["k_max"]
            )
        if plan.kind == "ref-factor":
            b = factor_scores(self.settings, sp["factor"], sp["sign"], self.labels(plan.target), plan.target)
            return self._s1_factory(
                plan.target, StoredScoreSource(b.frame), sp["k_max"], sp["overlay"],
                auto_resume=sp.get("auto_resume_sessions"),
            )  # fmt: skip
        b = self.blended(plan.target, sp["models"])
        src = StoredScoreSource(b.frame)
        meta: WalkForwardMeta | None = None
        if plan.kind == "s1-meta":
            meta = self._meta_for(sp["base_trial"])
        return self._s1_factory(
            plan.target,
            src,
            sp["k_max"],
            sp["overlay"],
            meta=meta,
            entry=sp.get("entry", "close"),
            auto_resume=sp.get("auto_resume_sessions"),
            policy_opts=sp.get("policy"),
        )

    def _meta_for(self, base_trial: str) -> WalkForwardMeta:
        d = trial_dir(self.settings, base_trial)
        samples = build_meta_samples(
            pl.read_parquet(d / "entries.parquet"), pl.read_parquet(d / "trades.parquet")
        )
        folds = self.cfg.folds or tuple(walk_forward_folds(self.settings, self.calendar))
        return WalkForwardMeta(samples, folds, self.cfg.meta_factory or MetaModel)

    # -- registration and running
    def register(self, plans: Sequence[TrialPlan]) -> list[TrialPlan]:
        return register_plans(self.state, plans)

    def run_trial(self, plan: TrialPlan) -> TrialResult:
        """Run (or load) one registered trial: every capital of the grid plus the stressed-slippage run at
        the reference capital; writes the files, then finishes the trial on the registry."""
        assert plan.trial_id is not None, "register the plan before running it"
        tid = plan.trial_id
        done = load_result(self.settings, tid)
        if done is not None:
            self._ensure_finished(tid, done)
            return done
        cfg, world, slip = self.cfg, self.world, self.world.slippage
        assert cfg.start is not None and cfg.end is not None and cfg.reference_capital is not None
        make = self.factory_for(plan)
        years = max(len(self.calendar.sessions(cfg.start, cfg.end)) / 252.0, 1e-9)
        self.current_trial = tid
        by_capital: dict[str, Any] = {}
        daily: pl.DataFrame | None = None
        ref_pol: Any = None
        ref_trades = pl.DataFrame()
        caps = list(cfg.capitals or ())
        if cfg.reference_capital not in caps:
            caps.append(cfg.reference_capital)
        extra_cols: list[pl.DataFrame] = []
        for cap in caps:
            pol = make(slip)
            res = run_backtest(
                world.panel, self.calendar, pol, cfg.start, cfg.end, cap, self.settings.costs, slip,
                self.settings.risk, debug=False,
            )  # fmt: skip
            by_capital[str(int(cap))] = _stats(res, cap, years)
            col = f"net_{int(cap)}"
            extra_cols.append(res.daily.select("date", pl.col("day_ret").alias(col)))
            if cap == cfg.reference_capital:
                ref_pol, ref_trades = pol, res.trades
                daily = res.daily.select(
                    "date", pl.col("day_ret").alias("net"), "ret_gross", "n_pos", "turnover", "costs_frac",
                    "gross", "cash_frac",
                )  # fmt: skip
        stress = slip.scaled(cfg.slippage_stress)
        pol2 = make(stress)
        res2 = run_backtest(
            world.panel, self.calendar, pol2, cfg.start, cfg.end, cfg.reference_capital,
            self.settings.costs, stress, self.settings.risk, debug=False,
        )  # fmt: skip
        stress_stats = _stats(res2, cfg.reference_capital, years)
        assert daily is not None
        daily = daily.join(
            res2.daily.select("date", pl.col("day_ret").alias("net_2x")), on="date", how="left"
        )
        for e in extra_cols:
            daily = daily.join(e, on="date", how="left")
        bench = (
            world.benchmark.select("date", pl.col("ew500_ret").alias("benchmark"))
            if plan.kind != "s2"
            else pl.DataFrame(
                {"date": daily["date"], "benchmark": [None] * daily.height},
                schema={"date": pl.Date, "benchmark": pl.Float64},
            )
        )
        daily = daily.join(bench, on="date", how="left").sort("date")
        net = daily["net"].to_numpy()
        bm = daily["benchmark"].fill_null(0.0).to_numpy()
        ref = by_capital[str(int(cfg.reference_capital))]
        excess = perf_stats(net, bm)["excess_tstat"]
        sd = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
        sharpe_daily = float(net.mean() / sd) if sd > 0 else 0.0
        extra: dict[str, Any] = {}
        if plan.kind == "s2":
            extra["strict_capture"] = strict_capture(self.settings, ref_trades, cfg.end)
        data = {
            "trial_id": tid, "kind": plan.kind, "family": plan.family, "target": plan.target,
            "label": plan.label, "spec": plan.spec, "reference_capital": cfg.reference_capital,
            "by_capital": by_capital, "stress_2x": stress_stats, "ref_sharpe": ref["sharpe"],
            "excess_tstat": excess, "sharpe_daily": sharpe_daily, "n_days": daily.height, **extra,
        }  # fmt: skip
        d = trial_dir(self.settings, tid)
        _write_parquet(d / "daily.parquet", daily)
        _write_parquet(d / "trades.parquet", ref_trades)
        if ref_pol is not None and getattr(ref_pol, "entry_log", None):
            _write_parquet(d / "entries.parquet", pl.DataFrame(ref_pol.entry_log))
        _write_json(d / "result.json", data)
        self.current_trial = None
        res_obj = TrialResult(tid, json.loads((d / "result.json").read_text(encoding="utf-8")), d)
        self._ensure_finished(tid, res_obj)
        return res_obj

    def _ensure_finished(self, tid: str, res: TrialResult) -> None:
        con = self.state.connect()
        try:
            row = con.execute("SELECT status FROM trials WHERE trial_id=?", (tid,)).fetchone()
        finally:
            con.close()
        if row is not None and row[0] == "done":
            return
        path = returns_path(self.settings, tid)
        daily = res.daily()
        _write_parquet(path, daily.select("date", pl.col("net").alias("ret")))
        metrics = {
            "sharpe_daily": res.data.get("sharpe_daily"), "n_days": res.data.get("n_days"),
            "ref_sharpe": res.data.get("ref_sharpe"), "kind": res.data.get("kind"),
        }  # fmt: skip
        finish_trial(self.state, tid, metrics, path)


# ---- strict capture (S2) ---------------------------------------------------------------------------


def strict_capture(settings: Settings, trades: pl.DataFrame, end: date | None = None) -> dict[str, Any]:
    """Recompute each S2 trade inside the 5-minute-bar history from the close of the 09:15 bar to the close
    of the 15:10 bar (same cost and slippage load as the auction trade) and return
    strict_capture = mean strict net return / mean auction net return over the same trades.

    Only dates inside the stored bars and not after `end` (the research end) are used. Returns
    {"available": False, "reason": ...} when there are no bars or no matching trades."""
    bars = (
        io.read(settings, "intraday_bars", end=end, columns=["symbol", "ts", "interval", "close"])
        .filter(pl.col("interval") == BARS_INTERVAL)
        .collect()
    )
    if bars.is_empty():
        return {"available": False, "reason": "no intraday_bars history in the research range"}
    bars = bars.with_columns(pl.col("ts").dt.date().alias("date"), pl.col("ts").dt.time().alias("_t"))
    ends = bars.filter(pl.col("_t") == STRICT_EXIT_BAR).select(
        "symbol", "date", pl.col("close").alias("c_exit")
    )
    starts = bars.filter(pl.col("_t") == STRICT_ENTRY_BAR).select(
        "symbol", "date", pl.col("close").alias("c_entry")
    )
    lo, hi = bars["date"].min(), bars["date"].max()
    t = trades.filter(
        (pl.col("product") == "intraday") & (pl.col("entry_date") >= lo) & (pl.col("entry_date") <= hi)
    )
    out: dict[str, Any] = {"available": False, "window": [lo, hi], "n_trades_in_window": t.height}
    if t.is_empty():
        out["reason"] = "no S2 trade inside the 5-minute-bar window"
        return out
    px = io.read(
        settings, "panel_daily", start=lo, end=hi, columns=["date", "sid", "symbol", "open", "close"]
    ).collect()
    j = (
        t.join(px.rename({"date": "entry_date"}), on=["sid", "entry_date"], how="inner")
        .join(starts.rename({"date": "entry_date"}), on=["symbol", "entry_date"], how="inner")
        .join(ends.rename({"date": "entry_date"}), on=["symbol", "entry_date"], how="inner")
    )
    out["n_matched"] = j.height
    if j.is_empty():
        out["reason"] = "no S2 trade has both bars"
        return out
    sign = pl.when(pl.col("direction") == "long").then(1.0).otherwise(-1.0)
    j = j.with_columns(
        (sign * (pl.col("close") / pl.col("open") - 1.0)).alias("g_auction"),
        (sign * (pl.col("c_exit") / pl.col("c_entry") - 1.0)).alias("g_strict"),
    ).with_columns((pl.col("ret") - pl.col("g_auction") + pl.col("g_strict")).alias("strict_net"))
    auction = float(j["ret"].mean())
    strict = float(j["strict_net"].mean())
    out.update(
        {
            "available": True, "auction_net_mean": auction, "strict_net_mean": strict,
            "strict_capture": strict / auction if auction > 0 else None,
            "note": None if auction > 0 else "auction net return is not positive: the ratio is undefined",
        }
    )  # fmt: skip
    return out
