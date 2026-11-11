"""Overnight gap factors from the corporate-action table checked against price evidence (PLAN T3.2).

NSE's `prev_close` is NOT adjusted for splits or bonuses (PLAN B.3 Finding 1), so the factor that turns
the raw overnight ratio `open / base` into a real return has to be inferred: the table says what should
have happened, the price says what did, and each step below trusts one of them.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_FACTOR_TYPES = ["split", "bonus", "bonus+split", "consolidation"]
_NOFACTOR_TYPES = ["demerger", "scheme_of_arrangement", "capital_reduction", "amalgamation"]
_FACTOR_OR_RIGHTS = [*_FACTOR_TYPES, "rights"]

SUSPECT_DOWN = (1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 10, 1 / 20, 2 / 3, 3 / 4, 3 / 5, 2 / 5, 4 / 5, 5 / 6)
SUSPECT_UP = (2.0, 5.0, 10.0)

_OUT = {
    "sid": pl.Int32,
    "date": pl.Date,
    "gap_factor": pl.Float64,
    "gap_src": pl.Utf8,
    "r_obs": pl.Float64,
    "f_ca": pl.Float64,
    "ca_types": pl.Utf8,
    "note": pl.Utf8,
}

_RE_RATIO = r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)"
_RE_PREMIUM = r"(?i)premium\s*(?:of\s*)?(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)"
_RE_AT_PRICE = r"(?i)@\s*(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)"
_DEFAULT_FACE = 10.0


def snap(r: np.ndarray) -> np.ndarray:
    """Each ratio replaced by the nearest 'suspected' ratio if within 4% (else unchanged)."""
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return r
    cs = np.array([*SUSPECT_DOWN, *SUSPECT_UP])
    rel = np.abs(r[:, None] / cs[None, :] - 1.0)
    j = rel.argmin(axis=1)
    ok = rel[np.arange(len(r)), j] <= 0.04
    return np.where(ok, cs[j], r)


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_OUT)


def _shape(df: pl.DataFrame) -> pl.DataFrame:
    """Add missing output columns as typed nulls and order/cast to the output schema."""
    return df.select(
        [pl.col(c).cast(t) if c in df.columns else pl.lit(None, dtype=t).alias(c) for c, t in _OUT.items()]
    )


def prepare_rows(rows: pl.DataFrame) -> pl.DataFrame:
    """rows -> (sid, date, session_idx, base, r_obs, is_first, lc), sorted by sid, date.

    `base` is the exchange prev_close (the previous raw close when it is missing) and
    `r_obs = open / base` the observed overnight ratio; both are null on a sid's first row.
    """
    r = rows.select("sid", "date", "session_idx", "open", "close", "prev_close").sort("sid", "date")
    r = r.with_columns(
        pl.col("close").shift(1).over("sid").alias("lc"),
        (pl.int_range(pl.len()).over("sid") == 0).alias("is_first"),
    )
    base = pl.when(pl.col("prev_close") > 0).then(pl.col("prev_close")).otherwise(pl.col("lc"))
    r = r.with_columns(base.alias("base"))
    r = r.with_columns(
        pl.when((pl.col("open") > 0) & (pl.col("base") > 0))
        .then(pl.col("open") / pl.col("base"))
        .otherwise(None)
        .alias("r_obs")
    )
    return r.select("sid", "date", "session_idx", "base", "r_obs", "is_first", "lc")


def _ex_idx(events: pl.DataFrame, sess: pl.DataFrame) -> pl.DataFrame:
    """Add `ex_idx`: session index of the first session on or after ex_date (null past the end)."""
    return (
        events.sort("ex_date")
        .join_asof(
            sess.rename({"date": "_sd", "session_idx": "ex_idx"}),
            left_on="ex_date",
            right_on="_sd",
            strategy="forward",
        )
        .drop("_sd")
    )


def _first_row_in_window(events: pl.DataFrame, r: pl.DataFrame, n_sessions: int) -> pl.DataFrame:
    """For each (sid, ex_date) the first non-first row with ex_idx <= session_idx <= ex_idx + n."""
    cand = events.join(r.filter(~pl.col("is_first")), on="sid").filter(
        (pl.col("session_idx") >= pl.col("ex_idx")) & (pl.col("session_idx") <= pl.col("ex_idx") + n_sessions)
    )
    return cand.sort("date").group_by(["sid", "ex_date"], maintain_order=True).first()


def _anti(df: pl.DataFrame, assigned: list[pl.DataFrame]) -> pl.DataFrame:
    for a in assigned:
        df = df.join(a.select("sid", "date"), on=["sid", "date"], how="anti")
    return df


def _rights_inputs(ca: pl.DataFrame) -> pl.DataFrame:
    """rights rows with A (new shares), B (held shares) and S (issue price, face + premium if premium)."""
    rt = ca.filter(pl.col("action_type") == "rights")
    if rt.is_empty():
        return rt.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("_A"),
            pl.lit(None, dtype=pl.Float64).alias("_B"),
            pl.lit(None, dtype=pl.Float64).alias("_S"),
        )
    subj = pl.col("subject_raw").fill_null("")
    face_fb = pl.col("face_fallback") if "face_fallback" in rt.columns else pl.lit(None, dtype=pl.Float64)
    face = pl.coalesce(pl.col("face_before"), face_fb, pl.lit(_DEFAULT_FACE))
    prem = subj.str.extract(_RE_PREMIUM, 1).cast(pl.Float64)
    at_price = subj.str.extract(_RE_AT_PRICE, 1).cast(pl.Float64)
    return rt.with_columns(
        pl.coalesce(pl.col("ratio_num"), subj.str.extract(_RE_RATIO, 1).cast(pl.Float64)).alias("_A"),
        pl.coalesce(pl.col("ratio_den"), subj.str.extract(_RE_RATIO, 2).cast(pl.Float64)).alias("_B"),
        pl.coalesce(
            pl.col("rights_price"),
            pl.when(prem.is_not_null()).then(face + prem).otherwise(at_price),
        ).alias("_S"),
    )


def compute_gap_factors(rows: pl.DataFrame, ca: pl.DataFrame) -> pl.DataFrame:
    """Sparse audit frame [sid, date, gap_factor, gap_src, r_obs, f_ca, ca_types, note].

    rows: sid, date, session_idx, open, close, prev_close (all sessions of all sids, any order).
    ca: corporate actions with `sid` already joined (symbol -> sid) plus the raw columns; an optional
    `face_fallback` column supplies the face value for premium-quoted rights issues.
    One row per overnight gap that needs a factor (gap_src in ca, ca_shifted, implied_at_ca, rights,
    implied_event, suspected; gap_factor = the factor) and one per ignored table factor
    (gap_src = ca_ignored, gap_factor = 1.0, dated at the ex-date). Rows not listed need no factor.
    A sid's first row never gets a factor (a listing-day move is not a return).
    """
    r = prepare_rows(rows)
    rn = (
        r.filter(~pl.col("is_first"))
        .select("sid", "date", "session_idx", "base", "r_obs", "is_first")
        .join(rows.select("sid", "date", "close"), on=["sid", "date"], how="left")
    )
    sess = r.select("date", "session_idx").unique().sort("date")
    ca = ca.filter(pl.col("sid").is_not_null()).with_columns(pl.col("sid").cast(pl.Int32))
    parts: list[pl.DataFrame] = []
    ignored: list[pl.DataFrame] = []

    # ---- step 2: table factors against price evidence ---------------------------------------
    f_ev = (
        ca.filter(
            pl.col("action_type").is_in(_FACTOR_TYPES)
            & pl.col("factor_price").is_not_null()
            & (pl.col("factor_price") > 0)
            & ((pl.col("factor_price") - 1.0).abs() > 1e-9)
        )
        .group_by(["sid", "ex_date"])
        .agg(
            pl.col("factor_price").log().sum().exp().alias("f"),
            pl.col("action_type").unique().sort().str.join(",").alias("ca_types"),
        )
    )
    if f_ev.height:
        cand = (
            f_ev.join(rn.filter(pl.col("r_obs").is_not_null()), on="sid")
            .filter(
                (pl.col("date") >= pl.col("ex_date") - pl.duration(days=5))
                & (pl.col("date") <= pl.col("ex_date") + pl.duration(days=15))
            )
            .with_columns((pl.col("r_obs") / pl.col("f")).log().abs().alias("err"))
        )
        best = cand.sort("err").group_by(["sid", "ex_date"], maintain_order=True).first()
        ok = best.filter(pl.col("err") <= 0.20)
        ok = ok.sort("err").unique(subset=["sid", "date"], keep="first", maintain_order=True)
        ok_out = ok.select(
            "sid",
            "date",
            pl.col("f").alias("gap_factor"),
            pl.when(pl.col("date") == pl.col("ex_date"))
            .then(pl.lit("ca"))
            .otherwise(pl.lit("ca_shifted"))
            .alias("gap_src"),
            "r_obs",
            pl.col("f").alias("f_ca"),
            "ca_types",
            pl.lit("").alias("note"),
        )
        parts.append(_shape(ok_out))
        rest = f_ev.join(ok.select("sid", "ex_date"), on=["sid", "ex_date"], how="anti")
        if rest.height:
            fb = (
                cand.filter(pl.col("date") >= pl.col("ex_date"))
                .sort("date")
                .group_by(["sid", "ex_date"], maintain_order=True)
                .first()
            )
            rest = rest.join(
                fb.select("sid", "ex_date", pl.col("date").alias("fb_date"), pl.col("r_obs").alias("fb_r")),
                on=["sid", "ex_date"],
                how="left",
            )
            ln_r = pl.col("fb_r").log()
            take = (ln_r.abs() > 0.15) & (ln_r.sign() == pl.col("f").log().sign())
            imp = rest.filter(take.fill_null(False))
            if imp.height:
                imp_out = imp.select(
                    "sid",
                    pl.col("fb_date").alias("date"),
                    pl.Series("gap_factor", snap(imp["fb_r"].to_numpy())),
                    pl.lit("implied_at_ca").alias("gap_src"),
                    pl.col("fb_r").alias("r_obs"),
                    pl.col("f").alias("f_ca"),
                    "ca_types",
                    pl.lit("table ratio disagrees with price; price used").alias("note"),
                )
                imp_out = _anti(_shape(imp_out), parts)
                parts.append(imp_out)
            ign = rest.filter(~take.fill_null(False))
            if ign.height:
                ignored.append(
                    _shape(
                        ign.select(
                            "sid",
                            pl.col("ex_date").alias("date"),
                            pl.lit(1.0).alias("gap_factor"),
                            pl.lit("ca_ignored").alias("gap_src"),
                            pl.col("fb_r").alias("r_obs"),
                            pl.col("f").alias("f_ca"),
                            "ca_types",
                            pl.lit("no price evidence for the table factor").alias("note"),
                        )
                    )
                )

    # ---- step 3: rights ----------------------------------------------------------------------
    rt = _rights_inputs(ca)
    if rt.height:
        rt = rt.group_by(["sid", "ex_date"]).first()
        rt = _ex_idx(rt.drop_nulls("ex_date"), sess).filter(pl.col("ex_idx").is_not_null())
        row = _first_row_in_window(rt, rn, 5)
        row = _anti(row, parts)
        row = row.filter(
            (pl.col("_S") < pl.col("base"))
            & (pl.col("_A") > 0)
            & (pl.col("_B") > 0)
            & pl.col("_S").is_not_null()
        )
        if row.height:
            f = (pl.col("_B") * pl.col("base") + pl.col("_A") * pl.col("_S")) / (
                (pl.col("_A") + pl.col("_B")) * pl.col("base")
            )
            parts.append(
                _shape(
                    row.select(
                        "sid",
                        "date",
                        f.alias("gap_factor"),
                        pl.lit("rights").alias("gap_src"),
                        "r_obs",
                        f.alias("f_ca"),
                        pl.lit("rights").alias("ca_types"),
                        pl.lit("").alias("note"),
                    )
                )
            )

    # ---- step 4: events without a factor -----------------------------------------------------
    nf = ca.filter(
        pl.col("action_type").is_in(_NOFACTOR_TYPES)
        | (
            pl.col("action_type").is_in(_FACTOR_OR_RIGHTS)
            & (
                pl.col("factor_price").is_null()
                | ((pl.col("factor_price") - 1.0).abs() <= 1e-9)
                | (pl.col("parse_status") == "unparsed")
            )
        )
    )
    if nf.height:
        nf = nf.group_by(["sid", "ex_date"]).agg(
            pl.col("action_type").unique().sort().str.join(",").alias("ca_types")
        )
        nf = _ex_idx(nf, sess).filter(pl.col("ex_idx").is_not_null())
        row = _anti(_first_row_in_window(nf, rn, 5), parts)
        small_open = row.filter(pl.col("r_obs").is_not_null() & (pl.col("r_obs").log().abs() <= 0.12))
        row = row.filter(pl.col("r_obs").is_not_null() & (pl.col("r_obs").log().abs() > 0.12))
        # The event happened inside the session: the open still shows the old price (e.g. ADANIENT
        # 2015-06-03, open -10%, close -83%). The close carries the evidence; the whole day is
        # treated as mechanical (ret 0, ret_co and ret_oc are set to null in the panel).
        intra = small_open.filter(
            (pl.col("close") > 0)
            & (pl.col("base") > 0)
            & ((pl.col("close") / pl.col("base")).log().abs() > 0.5)
        )
        if intra.height:
            parts.append(
                _shape(
                    intra.select(
                        "sid",
                        "date",
                        (pl.col("close") / pl.col("base")).alias("gap_factor"),
                        pl.lit("implied_event_close").alias("gap_src"),
                        "r_obs",
                        pl.lit(None, dtype=pl.Float64).alias("f_ca"),
                        "ca_types",
                        pl.lit("event inside the session; close ratio treated as mechanical").alias("note"),
                    )
                )
            )
        if row.height:
            parts.append(
                _shape(
                    row.select(
                        "sid",
                        "date",
                        pl.col("r_obs").alias("gap_factor"),
                        pl.lit("implied_event").alias("gap_src"),
                        "r_obs",
                        pl.lit(None, dtype=pl.Float64).alias("f_ca"),
                        "ca_types",
                        pl.lit("overnight move treated as mechanical").alias("note"),
                    )
                )
            )

    # ---- step 5: suspected, no table row ------------------------------------------------------
    maybe = _anti(
        rn.filter(
            pl.col("r_obs").is_not_null()
            & (pl.col("base") >= 5)
            & ((pl.col("r_obs").log() < -0.28) | (pl.col("r_obs").log() > 0.60))
        ),
        parts,
    )
    if maybe.height:
        rv = maybe["r_obs"].to_numpy()
        cs = np.array([*SUSPECT_DOWN, *SUSPECT_UP])
        rel = np.abs(rv[:, None] / cs[None, :] - 1.0)
        down = np.array([True] * len(SUSPECT_DOWN) + [False] * len(SUSPECT_UP))
        allowed = np.where(np.log(rv)[:, None] < -0.28, down[None, :], ~down[None, :])
        rel = np.where(allowed & (rel <= 0.03), rel, np.inf)
        j = rel.argmin(axis=1)
        hit = np.isfinite(rel[np.arange(len(rv)), j])
        if hit.any():
            sus = maybe.filter(pl.Series(hit)).with_columns(pl.Series("gap_factor", cs[j][hit]))
            parts.append(
                _shape(
                    sus.select(
                        "sid",
                        "date",
                        "gap_factor",
                        pl.lit("suspected").alias("gap_src"),
                        "r_obs",
                        pl.lit(None, dtype=pl.Float64).alias("f_ca"),
                        pl.lit(None, dtype=pl.Utf8).alias("ca_types"),
                        pl.lit("no table row; ratio inferred from price").alias("note"),
                    )
                )
            )

    out = pl.concat([*parts, *ignored]) if (parts or ignored) else _empty()
    return out.sort("sid", "date", "gap_src")
