"""
Phase J0.2: the honest progress bar for this project's forward record.

`docs/STATUS.md`'s promotion staircase (Backtest -> Paper -> Live shadow
-> Small capital -> Scaled) has only ever reached step 1 -- as of this
module's introduction, `predictions` holds rows across three data-dates
and `graded_at` is NULL on every single one. Every piece of grading,
calibration, and self-demotion machinery in this repo
(`grade_matured_predictions`, `evaluation.gate.evaluate_forward_record`,
`evaluation.calibration`) has never processed a real outcome. This
module answers, without editorializing, exactly one question: how far
is the live model's own forward record from the ≥30-graded-predictions
threshold (`evaluation.gate.ForwardRecordCriteria.min_graded_predictions`)
that the demotion check itself requires before it can mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stocksense.core.calendar import bar_shift, trading_days_index
from stocksense.evaluation.gate import ForwardRecordCriteria


@dataclass(frozen=True)
class LedgerStatus:
    model_type: str
    horizon_bars: int
    has_live_model: bool
    model_id: str | None
    n_recorded: int
    n_graded: int
    n_ungraded: int
    min_graded_required: int
    predictions_until_threshold: int
    earliest_as_of_date: str | None
    estimated_first_maturity_date: str | None
    latest_calendar_date: str | None


def ledger_status(
    store, model_type: str = "cross_sectional_ranker", horizon_bars: int = 10,
    criteria: ForwardRecordCriteria | None = None,
) -> LedgerStatus:
    """Reads real state only -- no side effects, safe to call as often as
    wanted (e.g. every dashboard refresh). `estimated_first_maturity_date`
    is computed from the OBSERVED trading calendar (bhavcopy_eq's own
    dates, via core.calendar.trading_days_index -- never a hardcoded
    holiday list, same discipline as everywhere else that touches
    calendar math in this codebase), so it correctly returns None rather
    than a wrong date once the calendar's known future runs out."""
    criteria = criteria or ForwardRecordCriteria()

    live = store.get_live_model(model_type, horizon_bars)
    has_live_model = not live.empty
    model_id = live.iloc[0]["model_id"] if has_live_model else None

    preds = store.read_predictions()
    if model_id is not None:
        preds = preds[preds["model_version"] == model_id]
    preds = preds[preds["horizon_bars"] == horizon_bars]

    n_recorded = int(len(preds))
    n_graded = int(preds["graded_at"].notna().sum()) if n_recorded else 0
    n_ungraded = n_recorded - n_graded

    earliest_as_of = None
    estimated_maturity = None
    if n_recorded:
        earliest_as_of = pd.Timestamp(preds["as_of_date"].min())

    calendar_dates = store.con.execute("SELECT DISTINCT date FROM bhavcopy_eq ORDER BY date").fetchdf()["date"]
    latest_calendar_date = None
    if not calendar_dates.empty:
        index = trading_days_index(calendar_dates)
        latest_calendar_date = index.max()
        if earliest_as_of is not None:
            estimated_maturity = bar_shift(index, earliest_as_of, horizon_bars)

    return LedgerStatus(
        model_type=model_type,
        horizon_bars=horizon_bars,
        has_live_model=has_live_model,
        model_id=model_id,
        n_recorded=n_recorded,
        n_graded=n_graded,
        n_ungraded=n_ungraded,
        min_graded_required=criteria.min_graded_predictions,
        predictions_until_threshold=max(0, criteria.min_graded_predictions - n_graded),
        earliest_as_of_date=earliest_as_of.date().isoformat() if earliest_as_of is not None else None,
        estimated_first_maturity_date=estimated_maturity.date().isoformat() if estimated_maturity is not None else None,
        latest_calendar_date=latest_calendar_date.date().isoformat() if latest_calendar_date is not None else None,
    )
