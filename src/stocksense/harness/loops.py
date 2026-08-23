"""
The reconcile loop (docs/00-overview.md's founding claim: "every night
the system learns from its own mistakes"). Closes CRITICAL-1 from the
original audit — `predictions` was created in the schema and never
written to. Two halves:

`record_predictions` — scores the current cross-section with a
registered model and writes one row per (symbol, date) to `predictions`,
frozen at prediction time (feature_snapshot_hash lets a later reviewer
confirm nothing was recomputed with hindsight).

`grade_matured_predictions` — once a prediction's horizon has actually
elapsed, computes what really happened and writes it back. Grading is
pure arithmetic (this module computes it); only the narrative
explanation of *why* a prediction was wrong is ever handed to Claude,
and that happens in report.py, not here — the same compute/narrate
split as everywhere else in this system.

Neither half promotes or demotes a model. Grading produces a record for
the next `train_candidate` gate evaluation to eventually take into
account (a future extension, not built here); this loop's job is
strictly to make StockSense's own predictions checkable against reality.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import structlog

from stocksense.core.config import get_settings
from stocksense.data.adjust import quarantine_unexplained_jumps
from stocksense.data.loader import load_candles
from stocksense.data.validate import quarantine_symbols
from stocksense.evaluation.gate import apply_forward_record_decision, evaluate_forward_record
from stocksense.features.engine import build_features
from stocksense.harness.graph import Graph, Node
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return
from stocksense.models.registry import load_model
from stocksense.models.train_candidate import QUANTILE_ARTIFACT_NAME

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RecordResult:
    run_id: str
    model_id: str
    as_of_date: str
    n_predictions: int


@dataclass(frozen=True)
class GradeResult:
    n_graded: int
    n_correct_direction: int
    mean_abs_error: float | None


def _feature_snapshot_hash(row: pd.Series, feature_names: list[str]) -> str:
    """Content hash of the exact feature values a prediction was made
    from. Lets a later reviewer confirm a graded prediction really was
    frozen at prediction time and not silently recomputed — the same
    "point-in-time, not recomputed with hindsight" discipline
    test_leakage.py enforces for training."""
    payload = json.dumps({k: (None if pd.isna(row[k]) else float(row[k])) for k in feature_names}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def record_predictions(store, horizon_bars: int, lifecycle: str = "live",
                        model_type: str = "cross_sectional_ranker",
                        turnover_rank_band: tuple[float, float] | None = None,
                        settings=None) -> RecordResult | None:
    """Scores the most recent available cross-section with the
    registered `lifecycle` model for `horizon_bars` and writes one
    predictions row per symbol. Returns None (rather than raising) if no
    such model exists or there is no clean data to score — the caller
    (a nightly loop) should treat that as "nothing to do today," not a
    hard failure, since a missing model on a given night is an expected
    state before the first model is ever promoted.

    Loads candles via data.loader.load_candles (Phase G) rather than
    store.read_candles() directly, so the ledger scores the SAME
    universe/price source `settings.price_source`/
    `settings.use_point_in_time_universe` selects for training — a
    ledger scoring the hardcoded 98-symbol path while train_candidate
    trains on bhavcopy would silently grade a model against a universe
    it was never shown. `turnover_rank_band`: see universe_pit.py's
    docstring -- only meaningful when settings.use_point_in_time_universe
    is set; forwarded so the ledger can be restricted to a specific cap
    band once Phase G1's re-run identifies which one to run live.

    `settings`: optional override, threaded the same way train_candidate_
    core's `settings` parameter is (Phase H1's --cap-band). Found live
    running Phase H1's reconcile for real: this function used to call
    get_settings() unconditionally, ignoring any caller-mutated settings
    object entirely -- get_settings() constructs a brand-new Settings()
    from the process environment on EVERY call (no caching), so a CLI
    command's local settings.price_source="bhavcopy" override had zero
    effect here. The bug was caught because the first real reconcile run
    against a bhavcopy-trained live model recorded 94 predictions -- the
    98-symbol 'candles' count, not the thousands expected on the
    full_pit universe that model was actually trained and gated on."""
    settings = settings or get_settings()
    row = store.con.execute(
        "SELECT * FROM model_registry WHERE model_type = ? AND horizon_bars = ? AND lifecycle_state = ? "
        "ORDER BY created_at DESC LIMIT 1",
        [model_type, horizon_bars, lifecycle],
    ).fetchdf()
    if row.empty:
        return None

    model_id = row.iloc[0]["model_id"]
    artifact_path = row.iloc[0]["artifact_path"]
    ranker = load_model(artifact_path)

    candles = load_candles(settings, store, turnover_rank_band=turnover_rank_band)
    if candles.empty:
        return None
    if settings.price_source == "bhavcopy":
        candles, _ = quarantine_unexplained_jumps(store, candles)
    else:
        candles, _ = quarantine_symbols(candles)
    feats = build_features(candles)

    latest_date = feats["date"].max()
    today_rows = feats[feats["date"] == latest_date].dropna(subset=ranker.feature_names_, how="any")
    if today_rows.empty:
        return None

    scores = ranker.predict(today_rows[ranker.feature_names_])
    ranks = pd.Series(scores).rank(ascending=False, method="first").astype(int)

    # Phase H2: a sibling QuantileRanker artifact (models/train_
    # candidate.py's _fit_and_save_quantile_ranker, Phase G3's class)
    # gives a real expected-move band instead of the point ranker's
    # raw score doubling as "predicted_return" with confidence always
    # NULL. Only models trained after Phase H2 have this sibling file --
    # falls back to exactly the pre-H2 behavior when it's absent, so an
    # older registered model keeps scoring correctly.
    predicted_return = scores
    confidence = pd.Series(None, index=today_rows.index, dtype=object)
    quantile_artifact_path = Path(artifact_path).parent / QUANTILE_ARTIFACT_NAME
    if quantile_artifact_path.exists():
        try:
            quantile_ranker = load_model(str(quantile_artifact_path))
            bands = quantile_ranker.predict_bands(today_rows[quantile_ranker.feature_names_])
            predicted_return = bands["predicted_return"].to_numpy()
            confidence = bands["confidence"]
        except Exception:
            log.warning("quantile_ranker_load_failed", model_id=model_id, exc_info=True)

    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    pred_rows = pd.DataFrame(
        {
            "run_id": run_id,
            "symbol": today_rows["symbol"].values,
            "as_of_date": latest_date,
            "horizon_bars": horizon_bars,
            "score": scores,
            "rank": ranks.values,
            "model_version": model_id,
            "horizon_type": "monthly" if horizon_bars >= 10 else "short",
            "predicted_return": predicted_return,
            "confidence": confidence.to_numpy(),
            "feature_snapshot_hash": [
                _feature_snapshot_hash(today_rows.iloc[i], ranker.feature_names_) for i in range(len(today_rows))
            ],
        }
    )
    store.write_predictions(pred_rows)
    return RecordResult(run_id=run_id, model_id=model_id, as_of_date=str(latest_date), n_predictions=len(pred_rows))


def grade_matured_predictions(store, horizon_bars: int, as_of_before: date | None = None,
                               turnover_rank_band: tuple[float, float] | None = None,
                               settings=None) -> GradeResult:
    """Finds predictions whose horizon has elapsed (as_of_date +
    horizon_bars trading bars <= as_of_before, defaulting to the latest
    available candle date) and grades them against realized relative
    forward return, computed by the SAME function training uses
    (labels.forward_return) so grading and training can never disagree
    about what "actual return" means.

    Loads candles via the same load_candles path as record_predictions
    (Phase G) -- a prediction recorded against the bhavcopy universe
    must be graded against the same universe's realized returns, not
    silently re-scored against the 98-symbol candles table.

    `settings`: see record_predictions' docstring -- same fix, same
    reason (get_settings() ignores a caller's mutated settings object;
    an explicit override must be threaded through)."""
    as_of_before = as_of_before or date.today()

    settings = settings or get_settings()
    candles = load_candles(settings, store, turnover_rank_band=turnover_rank_band)
    if candles.empty:
        return GradeResult(n_graded=0, n_correct_direction=0, mean_abs_error=None)
    if settings.price_source == "bhavcopy":
        candles, _ = quarantine_unexplained_jumps(store, candles)
    else:
        candles, _ = quarantine_symbols(candles)
    labeled = add_forward_return_labels(candles, horizon_bars=horizon_bars)
    labeled = add_relative_forward_return(labeled, horizon_bars=horizon_bars)
    rel_col = f"fwd_ret_{horizon_bars}b_rel"

    ungraded = store.read_ungraded_predictions(as_of_before)
    ungraded = ungraded[ungraded["horizon_bars"] == horizon_bars]
    if ungraded.empty:
        return GradeResult(n_graded=0, n_correct_direction=0, mean_abs_error=None)

    labeled_lookup = labeled.set_index(["symbol", "date"])[rel_col]

    n_graded = 0
    n_correct = 0
    abs_errors = []
    now = datetime.now(timezone.utc)

    for _, pred in ungraded.iterrows():
        key = (pred["symbol"], pd.Timestamp(pred["as_of_date"]))
        if key not in labeled_lookup.index:
            continue
        actual = labeled_lookup.loc[key]
        if pd.isna(actual):
            continue  # horizon hasn't actually matured in the data yet, despite the date filter

        predicted = pred["predicted_return"]
        direction_correct = bool((predicted > 0) == (actual > 0))
        abs_error = abs(predicted - actual)

        grade = {
            "direction_correct": direction_correct,
            "predicted_return": float(predicted),
            "actual_return": float(actual),
            "abs_error": float(abs_error),
        }
        store.grade_prediction(
            pred["run_id"], pred["symbol"], pred["as_of_date"], int(pred["horizon_bars"]),
            actual_return=float(actual), grade_json=json.dumps(grade), graded_at=now,
        )
        n_graded += 1
        n_correct += int(direction_correct)
        abs_errors.append(abs_error)

    mean_abs_error = float(sum(abs_errors) / len(abs_errors)) if abs_errors else None
    return GradeResult(n_graded=n_graded, n_correct_direction=n_correct, mean_abs_error=mean_abs_error)


def build_reconcile_graph(store, horizon_bars: int, lifecycle: str = "live",
                           turnover_rank_band: tuple[float, float] | None = None,
                           model_type: str = "cross_sectional_ranker", settings=None) -> Graph:
    """The reconcile loop as a harness.Graph, so it gets resumability,
    idempotency, and job_runs logging for free (harness/runner.py) --
    the loop engineering this whole module exists to demonstrate isn't a
    separate mechanism from the Foreman's; it's the same one.

    Three nodes: grade runs first (grading yesterday's matured
    predictions doesn't depend on today's new ones), then
    check_forward_record (Phase G2 -- closes docs/STATUS.md's named gap:
    a live model's own graded ledger can now roll it back to 'shadow' on
    a significantly-below-chance forward hit rate, per
    evaluation.gate.evaluate_forward_record/apply_forward_record_decision;
    only meaningful for lifecycle='live', so it no-ops for 'shadow'),
    then record runs last -- deliberately AFTER the demotion check, so a
    just-demoted model does not also get a fresh prediction recorded
    against it in the same run (record_predictions naturally returns
    None once no 'live' model remains, which is the correct fail-closed
    behavior, not a bug to work around). Each node is idempotency-keyed
    by calendar date, so re-running the graph twice on the same day is a
    no-op on the second run rather than double-recording, double-
    grading, or double-demoting -- the exact property
    docs/05-nightly-pipeline.md requires of every step."""
    today_key = date.today().isoformat()

    def _grade(ctx: dict) -> dict:
        result = grade_matured_predictions(store, horizon_bars=horizon_bars, turnover_rank_band=turnover_rank_band, settings=settings)
        return {"n_graded": result.n_graded, "n_correct_direction": result.n_correct_direction, "mean_abs_error": result.mean_abs_error}

    def _check_forward_record(ctx: dict) -> dict:
        if lifecycle != "live":
            return {"checked": False, "reason": "only the live model's forward record is checked"}
        live = store.get_live_model(model_type, horizon_bars)
        if live.empty:
            return {"checked": False, "reason": "no live model for this horizon"}
        model_id = live.iloc[0]["model_id"]
        verdict = evaluate_forward_record(store, model_id)
        if verdict.demote:
            apply_forward_record_decision(model_id, verdict, store)
        return {"checked": True, "model_id": model_id, "demoted": verdict.demote, "reason": verdict.reason}

    def _record(ctx: dict) -> dict:
        result = record_predictions(store, horizon_bars=horizon_bars, lifecycle=lifecycle,
                                     model_type=model_type, turnover_rank_band=turnover_rank_band, settings=settings)
        if result is None:
            return {"recorded": False}
        return {"recorded": True, "run_id": result.run_id, "n_predictions": result.n_predictions}

    return Graph(
        [
            Node("grade_matured_predictions", fn=_grade, idempotency_key=f"reconcile_grade:{horizon_bars}:{today_key}"),
            Node("check_forward_record", fn=_check_forward_record, depends_on=("grade_matured_predictions",),
                 idempotency_key=f"reconcile_forward_check:{horizon_bars}:{today_key}"),
            Node("record_predictions", fn=_record, depends_on=("check_forward_record",),
                 idempotency_key=f"reconcile_record:{horizon_bars}:{today_key}"),
        ]
    )
