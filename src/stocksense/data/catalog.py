from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from stocksense.core.errors import DataValidationError

Layer = Literal["raw_adopted", "raw_new", "derived"]
Partition = Literal["month", "year", "all", "day"]

_D = pl.Date
_F = pl.Float64
_S = pl.Utf8
_I64 = pl.Int64
_I32 = pl.Int32
_B = pl.Boolean
# every timestamp column is tz-aware IST with microsecond precision (PLAN D.3)
_TS = pl.Datetime("us", "Asia/Kolkata")


@dataclass(frozen=True)
class DatasetSpec:
    """Contract of one dataset (PLAN D.5).

    path       folder under data_root holding the partition files (`sub` is appended when nested)
    file       fixed file name for single-file datasets (partition "all"); default `part-all.parquet`
    nested     True when files live in a sub-folder chosen by the caller (nse_snapshots/<kind>,
               features/v<N>, labels/v<N>, preds/<model_id>)
    keep_extra True when columns beyond `schema` are legitimate and pass through unchanged
    nullable_key  key columns that may legitimately be null (excluded from the null-key check)
    freshness  max tolerated age of the newest `date_col` value, in calendar days (None = static)
    """

    name: str
    layer: Layer
    key: tuple[str, ...]
    date_col: str | None
    partition: Partition
    schema: dict[str, pl.DataType]
    freshness: int | None = None
    path: str = ""
    file: str | None = None
    nested: bool = False
    keep_extra: bool = False
    nullable_key: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.path:
            object.__setattr__(self, "path", f"parquet/{self.name}")
        missing = [
            c for c in (*self.key, *([self.date_col] if self.date_col else [])) if c not in self.schema
        ]
        if missing:
            raise DataValidationError(f"{self.name}: key/date columns not in schema: {missing}")


def _spec(
    name: str,
    layer: Layer,
    key: str,
    date_col: str | None,
    partition: Partition,
    schema: dict[str, pl.DataType],
    **kw,
) -> DatasetSpec:
    return DatasetSpec(
        name, layer, tuple(k.strip() for k in key.split(",") if k.strip()), date_col, partition, schema, **kw
    )


_OHLC = {"open": _F, "high": _F, "low": _F, "close": _F}

_RAW_ADOPTED = [
    _spec(
        "bhavcopy_eq",
        "raw_adopted",
        "symbol, series, date",
        "date",
        "month",
        {
            "symbol": _S,
            "series": _S,
            "date": _D,
            **_OHLC,
            "prev_close": _F,
            "last_price": _F,
            "volume": _I64,
            "turnover_inr": _F,
            "n_trades": _F,
            "era": _S,
            "isin": _S,
            "name": _S,
            "avg_price": _F,
        },
        freshness=5,
    ),
    _spec(
        "bhavcopy_delivery",
        "raw_adopted",
        "symbol, series, date",
        "date",
        "month",
        {"symbol": _S, "series": _S, "date": _D, "deliv_qty": _F, "deliv_pct": _F},
        freshness=5,
    ),
    _spec(
        "corporate_actions",
        "raw_adopted",
        "symbol, ex_date, subject_raw",
        "ex_date",
        "all",
        {
            "symbol": _S,
            "ex_date": _D,
            "action_type": _S,
            "ratio_num": _F,
            "ratio_den": _F,
            "factor_price": _F,
            "dividend_amount": _F,
            "face_before": _F,
            "face_after": _F,
            "subject_raw": _S,
            "parse_status": _S,
            "record_date": _D,
            "series": _S,
            "isin": _S,
            "source": _S,
            "rights_price": _F,
        },
        freshness=30,
    ),
    _spec(
        "fo_bhavcopy",
        "raw_adopted",
        "symbol, instrument_type, expiry_date, strike_price, option_type, date",
        "date",
        "month",
        {
            "symbol": _S,
            "instrument_type": _S,
            "expiry_date": _D,
            "strike_price": _F,
            "option_type": _S,
            "date": _D,
            **_OHLC,
            "prev_close": _F,
            "settle_price": _F,
            "oi": _I64,
            "change_in_oi": _I64,
            "underlying_price": _F,
        },
        freshness=5,
        nullable_key=("strike_price", "option_type"),
    ),
    _spec("india_vix", "raw_adopted", "date", "date", "all", {"date": _D, "close": _F}, freshness=5),
    _spec(
        "intraday_bars",
        "raw_adopted",
        "symbol, ts, interval",
        "ts",
        "month",
        {
            "symbol": _S,
            "ts": _TS,
            "interval": _S,
            **_OHLC,
            "volume": _I64,
            "source": _S,
            "adj_asof": _D,
        },
        freshness=5,
    ),
    # Per-kind schemas differ and the folder is nested by kind; read(..., sub="gainers").
    DatasetSpec(
        "nse_snapshots",
        "raw_adopted",
        (),
        None,
        "day",
        {},
        path="parquet/nse_snapshots",
        nested=True,
        keep_extra=True,
    ),
]

_RAW_NEW = [
    _spec(
        "index_daily",
        "raw_new",
        "index_name, date",
        "date",
        "year",
        {
            "index_name": _S,
            "date": _D,
            **_OHLC,
            "pe": _F,
            "pb": _F,
            "div_yield": _F,
            "volume": _F,
            "turnover_cr": _F,
        },
        freshness=5,
    ),
    _spec(
        "symbol_changes",
        "raw_new",
        "old_symbol, new_symbol, change_date",
        "change_date",
        "all",
        {"company": _S, "old_symbol": _S, "new_symbol": _S, "change_date": _D},
        freshness=30,
    ),
    _spec(
        "listed_equities",
        "raw_new",
        "symbol, as_of",
        "as_of",
        "all",
        {
            "symbol": _S,
            "name": _S,
            "series": _S,
            "listing_date": _D,
            "isin": _S,
            "face_value": _F,
            "as_of": _D,
        },
        freshness=10,
    ),
    _spec(
        "etf_list",
        "raw_new",
        "symbol, as_of",
        "as_of",
        "all",
        {"symbol": _S, "underlying": _S, "name": _S, "as_of": _D},
        freshness=10,
    ),
    _spec(
        "sectors",
        "raw_new",
        "symbol, as_of",
        "as_of",
        "all",
        {"symbol": _S, "industry": _S, "company": _S, "isin": _S, "source": _S, "as_of": _D},
        freshness=30,
    ),
    _spec(
        "price_bands",
        "raw_new",
        "symbol, series, date",
        "date",
        "month",
        {"symbol": _S, "series": _S, "band": _S, "date": _D},
        freshness=5,
    ),
    _spec(
        "holidays",
        "raw_new",
        "date",
        "date",
        "all",
        {"date": _D, "description": _S, "fetched_at": _TS},
    ),
    _spec(
        "announcements",
        "raw_new",
        "seq_id",
        "ts",
        "month",
        {
            "seq_id": _S,
            "symbol": _S,
            "ts": _TS,
            "exch_ts": _TS,
            "category": _S,
            "summary": _S,
            "attachment_url": _S,
            "company": _S,
            "isin": _S,
            "has_xbrl": _B,
            "first_seen_ts": _TS,
        },
        freshness=3,
    ),
    _spec(
        "board_meetings",
        "raw_new",
        "symbol, meeting_date, announced_ts",
        "meeting_date",
        "all",
        {"symbol": _S, "meeting_date": _D, "purpose": _S, "description": _S, "announced_ts": _TS},
        freshness=10,
    ),
    _spec(
        "results_index",
        "raw_new",
        "seq_number",
        "filing_ts",
        "all",
        {
            "seq_number": _S,
            "symbol": _S,
            "period_from": _D,
            "period_to": _D,
            "filing_ts": _TS,
            "exch_ts": _TS,
            "audited": _S,
            "consolidated": _S,
            "relating_to": _S,
            "xbrl_url": _S,
        },
        freshness=10,
    ),
    _spec(
        "macro_daily",
        "raw_new",
        "series, date",
        "date",
        "all",
        {"series": _S, "date": _D, "close": _F, "available_ts": _TS},
        freshness=5,
    ),
    _spec(
        "participant_oi",
        "raw_new",
        "date, client_type",
        "date",
        "year",
        {
            "date": _D,
            "client_type": _S,
            "fut_idx_long": _F,
            "fut_idx_short": _F,
            "fut_stk_long": _F,
            "fut_stk_short": _F,
            "opt_idx_call_long": _F,
            "opt_idx_put_long": _F,
            "opt_idx_call_short": _F,
            "opt_idx_put_short": _F,
        },
        freshness=5,
        keep_extra=True,  # "plus stock-option columns as present"
    ),
    _spec(
        "fii_dii",
        "raw_new",
        "date, category",
        "date",
        "all",
        {"date": _D, "category": _S, "buy_value": _F, "sell_value": _F, "net_value": _F},
        freshness=5,
    ),
    _spec(
        "deals",
        "raw_new",
        "date, symbol, client, side, kind, qty, price",
        "date",
        "year",
        {"date": _D, "kind": _S, "symbol": _S, "client": _S, "side": _S, "qty": _F, "price": _F},
        freshness=5,
    ),
    _spec(
        "spreads",
        "raw_new",
        "symbol, ts",
        "ts",
        "month",
        {
            "symbol": _S,
            "ts": _TS,
            "bid": _F,
            "ask": _F,
            "ltp": _F,
            "bid_qty": _F,
            "ask_qty": _F,
            "source": _S,
        },
    ),
    _spec(
        "preopen",
        "raw_new",
        "symbol, ts",
        "ts",
        "month",
        {
            "symbol": _S,
            "ts": _TS,
            "prev_close": _F,
            "indicative_price": _F,
            "final_price": _F,
            "buy_qty": _F,
            "sell_qty": _F,
            "source": _S,
        },
    ),
    _spec(
        "quotes_1m",
        "raw_new",
        "symbol, ts",
        "ts",
        "day",
        {
            "symbol": _S,
            "ts": _TS,
            "ltp": _F,
            "open": _F,
            "high": _F,
            "low": _F,
            "close_prev": _F,
            "volume": _F,
            "bid": _F,
            "ask": _F,
            "source": _S,
        },
    ),
    _spec(
        "bars_1m",
        "raw_new",
        "symbol, ts",
        "ts",
        "month",
        {"symbol": _S, "ts": _TS, **_OHLC, "volume": _I64, "source": _S, "adj_asof": _D},
    ),
    _spec(
        "news_items",
        "raw_new",
        "source, raw_id",
        "ts_seen",
        "month",
        {
            "source": _S,
            "raw_id": _S,
            "ts_published": _TS,
            "ts_seen": _TS,
            "title": _S,
            "summary": _S,
            "url": _S,
        },
        freshness=2,
    ),
]

_ADJ = {"adj_open": _F, "adj_high": _F, "adj_low": _F, "adj_close": _F}

_DERIVED = [
    _spec(
        "security_master",
        "derived",
        "sid, valid_from",
        None,
        "all",
        {
            "sid": _I32,
            "symbol": _S,
            "valid_from": _D,
            "valid_to": _D,
            "isin_last": _S,
            "name_last": _S,
            "first_date": _D,
            "last_date": _D,
        },
        path="panel",
        file="security_master.parquet",
    ),
    _spec(
        "panel_daily",
        "derived",
        "sid, date",
        "date",
        "year",
        {
            "sid": _I32,
            "date": _D,
            "symbol": _S,
            "series": _S,
            **_OHLC,
            "prev_close": _F,
            "volume": _I64,
            "turnover_inr": _F,
            "n_trades": _F,
            "deliv_qty": _F,
            "deliv_pct": _F,
            "gap_factor": _F,
            "gap_src": _S,
            "adj": _F,
            **_ADJ,
            "adj_volume": _F,
            "ret": _F,
            "ret_co": _F,
            "ret_oc": _F,
            "div_ret": _F,
            "tret": _F,
            "sessions_since_prev": _I32,
            "flag_bigmove": _B,
            "flag_unexplained_gap": _B,
            "flag_stale": _B,
            "flag_one_price": _B,
            "flag_locked_up": _B,
            "flag_locked_dn": _B,
            "industry": _S,
        },
        path="panel/daily",
        freshness=5,
    ),
    _spec(
        "panel_universe",
        "derived",
        "date, sid",
        "date",
        "year",
        {
            "date": _D,
            "sid": _I32,
            "turnover_rank": _I32,
            "med_turn_60": _F,
            "u500": _B,
            "u200": _B,
            "in_fno": _B,
        },
        path="panel/universe",
        freshness=5,
    ),
    _spec(
        "panel_market",
        "derived",
        "date",
        "date",
        "all",
        {
            "date": _D,
            "ew500_ret": _F,
            "ew500_co": _F,
            "ew500_oc": _F,
            "ew200_ret": _F,
            "nifty50_ret": _F,
            "nifty500_ret": _F,
            "n_u500": _I32,
            "breadth_200": _F,
            "xs_dispersion": _F,
        },
        path="panel",
        file="market.parquet",
        freshness=5,
    ),
    _spec(
        "adj_events",
        "derived",
        "sid, date, symbol",
        "date",
        "all",
        {
            "sid": _I32,
            "date": _D,
            "symbol": _S,
            "gap_factor": _F,
            "gap_src": _S,
            "r_obs": _F,
            "f_ca": _F,
            "ca_types": _S,
            "note": _S,
        },
        path="panel",
        file="adj_events.parquet",
    ),
    # Feature / label columns are data-dependent: only the identifying columns are fixed.
    _spec(
        "features",
        "derived",
        "date, sid",
        "date",
        "year",
        {"date": _D, "sid": _I32},
        path="features",
        nested=True,
        keep_extra=True,
    ),
    _spec(
        "labels",
        "derived",
        "date, sid",
        "date",
        "year",
        {
            "date": _D,
            "sid": _I32,
            "y_oc": _F,
            "y_c2o_5": _F,
            "y_c2o_10": _F,
            "y_c2o_20": _F,
            "y_c2c_1": _F,
            "y_o2o_5": _F,
            "end_oc": _D,
            "end_5": _D,
            "end_10": _D,
            "end_20": _D,
            "entry_blocked_open": _B,
            "entry_blocked_close": _B,
            "label_delisted": _B,
        },
        path="labels",
        nested=True,
        keep_extra=True,  # z_<name>, r_<name> (Float32) columns
    ),
    _spec(
        "preds",
        "derived",
        "date, sid",
        "date",
        "year",
        {"date": _D, "sid": _I32, "score": pl.Float32, "made_at": _TS, "mode": _S},
        path="preds",
        nested=True,
    ),
]

CATALOG: dict[str, DatasetSpec] = {s.name: s for s in (*_RAW_ADOPTED, *_RAW_NEW, *_DERIVED)}


def get_spec(dataset: str) -> DatasetSpec:
    """The catalog entry for `dataset`; DataValidationError if unknown."""
    try:
        return CATALOG[dataset]
    except KeyError:
        raise DataValidationError(f"unknown dataset: {dataset!r}") from None
