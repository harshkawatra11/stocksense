"""Event features (PLAN T4.3j): dividends, board meetings, results, announcements, corporate actions.

`FeatureContext.raw()` does not cut forward-looking date columns (`corporate_actions.ex_date`,
`board_meetings.meeting_date`). These features therefore read those two datasets without an `as_of` cut
and gate every row on its KNOWLEDGE time: a board meeting by `announced_ts` (known from the session whose
20:00 IST cutoff is at or after it), a corporate action by `ex_date - 7 sessions` (the table has no
first-seen date, so historical rows are assumed known 7 sessions ahead), and then drop every row not
known by `as_of`. Source-start rule: a feature is null before its source's first known record.
"""

from __future__ import annotations

from datetime import time

import polars as pl

from stocksense.data import io
from stocksense.features._calutil import bday_count, bday_shift
from stocksense.features.base import (
    FeatureContext,
    feature,
    log_ret,
    roll_cov,
    roll_mean,
    roll_sum,
    roll_var,
)

CAP_NEAR = 30
CAP_RESULT = 90
KNOWN_LAG = 7  # sessions before the ex-date at which a corporate action is assumed known
RESULT_RE = r"(?i)financial result|outcome of board meeting"
PAIR_SECONDS = 2 * 86400  # an announcement and a results_index filing this close form one results event
DEDUP_DAYS = 3
CA_STRUCTURAL = ("split", "bonus", "rights", "demerger", "bonus+split")
_CUT = time(20, 0)
_MARKET_CLOSE = time(15, 30)
_MARKET_OPEN = time(9, 15)


def _known_date(ts: pl.Expr) -> pl.Expr:
    """Date of the first session whose 20:00 IST cutoff is at or after `ts`."""
    d = ts.dt.date()
    return pl.when(ts.dt.time() > _CUT).then(d + pl.duration(days=1)).otherwise(d)


def _panel(ctx: FeatureContext) -> pl.DataFrame:
    return ctx.panel().collect().sort("sid", "date")


def _null_frame(panel: pl.DataFrame, *names: str) -> pl.DataFrame:
    return panel.select("date", "sid", *[pl.lit(None, dtype=pl.Float64).alias(n) for n in names])


def _ca(ctx: FeatureContext) -> pl.DataFrame:
    """corporate_actions with a `known` date, read without the as_of cut (empty-safe)."""
    ca = io.read(
        ctx.settings, "corporate_actions", columns=["symbol", "ex_date", "action_type", "dividend_amount"]
    ).collect()
    return ca.with_columns(bday_shift(ca["ex_date"], -KNOWN_LAG).alias("known"))


def _gated(df: pl.DataFrame, ctx: FeatureContext, col: str = "known") -> pl.DataFrame:
    return df if ctx.as_of is None else df.filter(pl.col(col) <= ctx.as_of)


def _next_event(panel: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """Panel rows with the next STRICTLY later ex_date of the symbol (and its `known` date)."""
    p = panel.select("date", "sid", "symbol", "close").sort("date")
    return p.join_asof(
        ev.sort("ex_date"),
        left_on="date",
        right_on="ex_date",
        by="symbol",
        strategy="forward",
        check_sortedness=False,
        allow_exact_matches=False,
    )


def _exdiv(ctx: FeatureContext) -> pl.DataFrame:
    panel = _panel(ctx)
    ca = _ca(ctx)
    div = (
        _gated(ca.filter(pl.col("dividend_amount") > 0), ctx)
        .group_by("symbol", "ex_date", "known")
        .agg(pl.col("dividend_amount").sum().alias("amt"))
    )
    j = _next_event(panel, div)
    ok = (pl.col("known") <= pl.col("date")) & pl.col("ex_date").is_not_null()
    days = bday_count(j["date"], j["ex_date"]).clip(upper_bound=CAP_NEAR)
    return j.with_columns(days.alias("_d")).select(
        "date",
        "sid",
        pl.when(ok).then(pl.col("_d")).alias("days_to_exdiv"),
        pl.when(ok).then(pl.col("amt") / pl.col("close")).alias("exdiv_yield"),
    )


@feature(
    "days_to_exdiv",
    group="events",
    kind="xs",
    window=1,
    label="Sessions until the stock's next dividend ex-date (price drops by the dividend), if already known.",
)
def days_to_exdiv(ctx: FeatureContext) -> pl.LazyFrame:
    """Weekdays to the next dividend ex-date known by t (cap 30); null if none known."""
    return _exdiv(ctx).select("date", "sid", "days_to_exdiv").lazy()


@feature(
    "exdiv_yield",
    group="events",
    kind="xs",
    window=1,
    label="Size of the next known dividend as a fraction of today's price.",
)
def exdiv_yield(ctx: FeatureContext) -> pl.LazyFrame:
    """Next known dividend amount divided by the raw close."""
    return _exdiv(ctx).select("date", "sid", "exdiv_yield").lazy()


@feature(
    "corp_event_next_5",
    group="events",
    kind="xs",
    window=1,
    label="1 if a split, bonus, rights issue or demerger takes effect within the next 5 sessions.",
)
def corp_event_next_5(ctx: FeatureContext) -> pl.LazyFrame:
    """1 if a split/bonus/rights/demerger ex-date falls in the next 5 weekdays and is already known."""
    panel = _panel(ctx)
    ca = _ca(ctx)
    if ca.is_empty():
        return _null_frame(panel, "corp_event_next_5").lazy()
    ev = (
        _gated(ca.filter(pl.col("action_type").is_in(CA_STRUCTURAL)), ctx)
        .select("symbol", "ex_date", "known")
        .unique()
    )
    j = _next_event(panel, ev)
    n = bday_count(j["date"], j["ex_date"])
    hit = (n >= 1) & (n <= 5) & (j["known"] <= j["date"])
    return (
        j.with_columns(hit.fill_null(False).cast(pl.Float64).alias("corp_event_next_5"))
        .select("date", "sid", "corp_event_next_5")
        .lazy()
    )


@feature(
    "days_to_board_meeting",
    group="events",
    kind="xs",
    window=1,
    label="Sessions until the next announced board meeting (results, dividends); 30 if none is known.",
)
def days_to_board_meeting(ctx: FeatureContext) -> pl.LazyFrame:
    """Weekdays to the next board meeting announced by t (cap 30; 0 on the day); null before the source."""
    panel = _panel(ctx)
    bm = io.read(ctx.settings, "board_meetings", columns=["symbol", "meeting_date", "announced_ts"]).collect()
    bm = _gated(bm.with_columns(_known_date(pl.col("announced_ts")).alias("known")), ctx)
    if bm.is_empty():
        return _null_frame(panel, "days_to_board_meeting").lazy()
    start = bm["known"].min()
    s0 = pl.max_horizontal(pl.col("known"), pl.col("meeting_date") - pl.duration(days=45))
    exp = (
        bm.with_columns(s0.alias("_s0"))
        .filter(pl.col("_s0") <= pl.col("meeting_date"))
        .select("symbol", "meeting_date", pl.date_ranges("_s0", "meeting_date").alias("date"))
        .explode("date")
    )
    nxt = (
        exp.join(panel.select("symbol", "date", "sid"), on=["symbol", "date"], how="inner")
        .group_by("sid", "date")
        .agg(pl.col("meeting_date").min().alias("_m"))
    )
    j = panel.select("date", "sid").join(nxt, on=["sid", "date"], how="left")
    days = bday_count(j["date"], j["_m"]).clip(upper_bound=CAP_NEAR)
    return (
        j.with_columns(days.alias("_d"))
        .select(
            "date",
            "sid",
            pl.when(pl.col("date") >= start)
            .then(pl.col("_d").fill_null(CAP_NEAR))
            .alias("days_to_board_meeting"),
        )
        .lazy()
    )


def _ann_counts(ctx: FeatureContext) -> pl.DataFrame:
    panel = _panel(ctx)
    a = ctx.raw("announcements", ["symbol", "ts"]).drop_nulls().collect()
    names = ("ann_count_5", "ann_count_21")
    if a.is_empty():
        return _null_frame(panel, *names)
    a = a.with_columns(_known_date(pl.col("ts")).alias("adate"))
    start = a["adate"].min()
    cnt = a.group_by("symbol", "adate").agg(pl.len().alias("n")).sort("adate")
    keys = panel.select("symbol", "sid", "date").sort("date")
    m = cnt.join_asof(
        keys, left_on="adate", right_on="date", by="symbol", strategy="forward", check_sortedness=False
    )
    m = m.drop_nulls("date").group_by("sid", "date").agg(pl.col("n").sum().cast(pl.Float64))
    p = (
        panel.select("date", "sid", "sx")
        .join(m, on=["sid", "date"], how="left")
        .with_columns(pl.col("n").fill_null(0.0))
        .sort("sid", "date")
    )
    return p.select(
        "date",
        "sid",
        pl.when(pl.col("date") >= start).then(roll_sum("n", 5)).alias("ann_count_5"),
        pl.when(pl.col("date") >= start).then(roll_sum("n", 21)).alias("ann_count_21"),
    )


@feature(
    "ann_count_5",
    group="events",
    kind="xs",
    window=5,
    label="Number of exchange announcements by the company over the last 5 sessions (news flow).",
)
def ann_count_5(ctx: FeatureContext) -> pl.LazyFrame:
    """Announcements assigned to a session by their 20:00 IST cutoff, summed over 5 rows."""
    return _ann_counts(ctx).select("date", "sid", "ann_count_5").lazy()


@feature(
    "ann_count_21",
    group="events",
    kind="xs",
    window=21,
    label="Number of exchange announcements by the company over the last month (news flow).",
)
def ann_count_21(ctx: FeatureContext) -> pl.LazyFrame:
    """Announcements assigned to a session by their 20:00 IST cutoff, summed over 21 rows."""
    return _ann_counts(ctx).select("date", "sid", "ann_count_21").lazy()


# ---- results events ----------------------------------------------------------------------------


def _result_events(ctx: FeatureContext) -> pl.DataFrame:
    """(symbol, ev_ts, known_ts): announcement + results_index filing within 2 days, deduplicated causally."""
    ann = (
        ctx.raw("announcements", ["symbol", "ts", "category"])
        .filter(pl.col("category").str.contains(RESULT_RE))
        .select("symbol", "ts")
        .collect()
    )
    ri = ctx.raw("results_index", ["symbol", "filing_ts"]).drop_nulls().collect()
    pairs = ann.join(ri, on="symbol", how="inner").filter(
        (pl.col("ts") - pl.col("filing_ts")).dt.total_seconds().abs() <= PAIR_SECONDS
    )
    ev = pairs.select(
        "symbol",
        pl.min_horizontal("ts", "filing_ts").alias("ev_ts"),
        pl.max_horizontal("ts", "filing_ts").alias("known_ts"),
    ).sort("symbol", "known_ts", "ev_ts")
    keep, last_sym, last_known = [], None, None
    for sym, known in zip(ev["symbol"].to_list(), ev["known_ts"].to_list(), strict=True):
        dup = sym == last_sym and (known - last_known).total_seconds() < DEDUP_DAYS * 86400
        keep.append(not dup)
        if not dup:
            last_sym, last_known = sym, known
    return ev.filter(pl.Series(keep, dtype=pl.Boolean)) if keep else ev


def _results(ctx: FeatureContext) -> pl.DataFrame:
    """days_since_result, result_abret, result_abvol, result_gap per (date, sid); memoised on the context."""
    cache = ctx.__dict__.setdefault("_results_cache", {})
    if "df" in cache:
        return cache["df"]
    panel = _panel(ctx)
    names = ("days_since_result", "result_abret", "result_abvol", "result_gap")
    ev = _result_events(ctx)
    if ev.is_empty():
        cache["df"] = _null_frame(panel, *names)
        return cache["df"]
    mk = ctx.market().select("date", "ew500_ret").collect()
    p = (
        panel.join(mk, on="date", how="left")
        .sort("sid", "date")
        .with_columns(log_ret("ret").alias("_la"), log_ret("ew500_ret").alias("_lm"))
        .with_columns((roll_cov("_la", "_lm", 126) / roll_var("_lm", 126)).alias("_beta"))
        .with_columns(
            pl.col("_beta").shift(1).over("sid").alias("_beta_prev"),
            roll_mean("turnover_inr", 60).shift(1).over("sid").alias("_v60"),
            pl.int_range(pl.len()).over("sid").alias("rn"),
        )
        .with_columns((pl.col("tret") - pl.col("_beta_prev") * pl.col("ew500_ret")).alias("_ab"))
    )
    e = ev.with_columns(
        pl.int_range(pl.len()).alias("eid"),
        _known_date(pl.col("known_ts")).alias("K"),
        (
            pl.col("ev_ts").dt.date()
            + pl.duration(days=1) * (pl.col("ev_ts").dt.time() >= _MARKET_CLOSE).cast(pl.Int64)
        ).alias("R1"),
        (
            pl.col("ev_ts").dt.date()
            + pl.duration(days=1) * (pl.col("ev_ts").dt.time() >= _MARKET_OPEN).cast(pl.Int64)
        ).alias("RG"),
    )
    ev_min = e["K"].min()
    keys = p.select("symbol", "date", "sid", "rn", "sx").sort("date")

    def locate(col: str, rn_name: str, date_name: str) -> pl.DataFrame:
        j = e.sort(col).join_asof(
            keys, left_on=col, right_on="date", by="symbol", strategy="forward", check_sortedness=False
        )
        return j.select(
            "eid",
            "K",
            pl.col("sid").alias(f"sid_{rn_name}"),
            pl.col("rn").alias(rn_name),
            pl.col("date").alias(date_name),
        )

    lookup = p.select("sid", "rn", "date", "_ab", "turnover_inr", "_v60", "ret_co")
    # reaction window: the first two sessions on/after the results timestamp
    r1 = locate("R1", "rn1", "date1").drop_nulls("rn1")
    r1 = r1.join(
        lookup.select(
            pl.col("sid").alias("sid_rn1"),
            pl.col("rn").alias("rn1"),
            pl.col("_ab").alias("ab1"),
            pl.col("turnover_inr").alias("t1"),
            pl.col("_v60").alias("v60"),
        ),
        on=["sid_rn1", "rn1"],
        how="left",
    ).with_columns((pl.col("rn1") + 1).alias("rn2"))
    r1 = r1.join(
        lookup.select(
            pl.col("sid").alias("sid_rn1"),
            pl.col("rn").alias("rn2"),
            pl.col("_ab").alias("ab2"),
            pl.col("turnover_inr").alias("t2"),
            pl.col("date").alias("date2"),
        ),
        on=["sid_rn1", "rn2"],
        how="inner",
    )
    ab_events = r1.select(
        pl.col("sid_rn1").alias("sid"),
        pl.max_horizontal("date2", "K").alias("avail"),
        pl.col("K").alias("evK"),
        (pl.col("ab1") + pl.col("ab2")).alias("result_abret"),
        (((pl.col("t1") + pl.col("t2")) / 2.0) / pl.col("v60")).alias("result_abvol"),
    )
    rg = (
        locate("RG", "rng", "dateg")
        .drop_nulls("rng")
        .join(
            lookup.select(
                pl.col("sid").alias("sid_rng"), pl.col("rn").alias("rng"), pl.col("ret_co").alias("gap")
            ),
            on=["sid_rng", "rng"],
            how="inner",
        )
    )
    gap_events = rg.select(
        pl.col("sid_rng").alias("sid"),
        pl.max_horizontal("dateg", "K").alias("avail"),
        pl.col("K").alias("evK"),
        pl.col("gap").alias("result_gap"),
    )
    rk = (
        locate("K", "rnk", "datek")
        .drop_nulls("rnk")
        .join(
            keys.select(pl.col("sid").alias("sid_rnk"), pl.col("rn").alias("rnk"), pl.col("sx").alias("sxk")),
            on=["sid_rnk", "rnk"],
            how="left",
        )
    )
    since_events = rk.select(pl.col("sid_rnk").alias("sid"), pl.col("datek").alias("avail"), pl.col("sxk"))

    base = p.select("date", "sid", "sx").sort("date")

    def asof(events: pl.DataFrame) -> pl.DataFrame:
        return base.join_asof(
            events.sort("avail"),
            left_on="date",
            right_on="avail",
            by="sid",
            strategy="backward",
            check_sortedness=False,
        )

    s_df = asof(since_events).select(
        "date",
        "sid",
        pl.when(pl.col("date") >= ev_min)
        .then((pl.col("sx") - pl.col("sxk")).clip(upper_bound=CAP_RESULT).fill_null(CAP_RESULT))
        .alias("days_since_result"),
    )

    # events carry their K as a calendar date; map it to the sx of the sid's first row on/after K
    def with_sx(events: pl.DataFrame) -> pl.DataFrame:
        j = events.sort("evK").join_asof(
            keys.select("sid", pl.col("date").alias("_kd"), pl.col("sx").alias("sx_evK")).sort("_kd"),
            left_on="evK",
            right_on="_kd",
            by="sid",
            strategy="forward",
            check_sortedness=False,
        )
        return j.drop("_kd")

    def staleness(events: pl.DataFrame, *cols: str) -> pl.DataFrame:
        j = asof(with_sx(events)).select("date", "sid", "sx", "sx_evK", *cols)
        ok = (pl.col("sx") - pl.col("sx_evK")) <= CAP_RESULT
        return j.select("date", "sid", *[pl.when(ok).then(pl.col(c)).alias(c) for c in cols])

    a_df = staleness(ab_events, "result_abret", "result_abvol")
    g_df = staleness(gap_events, "result_gap")
    out = s_df.join(a_df, on=["date", "sid"], how="left").join(g_df, on=["date", "sid"], how="left")
    cache["df"] = out
    return out


def _result_feature(name: str, label: str, window: int):
    def fn(ctx: FeatureContext) -> pl.LazyFrame:
        return _results(ctx).select("date", "sid", name).lazy()

    fn.__doc__ = f"{name} from the shared results-event frame (null before the first results event)."
    feature(name, group="events", kind="xs", window=window, label=label)(fn)


_result_feature(
    "days_since_result", "Sessions since the company's last results announcement (capped at 90).", 1
)
_result_feature(
    "result_abret",
    "How much the stock beat or missed the market over the 2 sessions after its last results.",
    1,
)
_result_feature(
    "result_abvol", "How much trading volume jumped over the 2 sessions after its last results.", 1
)
_result_feature("result_gap", "The overnight price gap the stock opened with after its last results.", 1)
