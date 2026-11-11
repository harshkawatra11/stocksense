from __future__ import annotations

import json
from datetime import date, datetime

import polars as pl
import pytest

from stocksense.core.clock import IST, FixedClock
from stocksense.core.errors import LockboxError
from stocksense.data import io
from stocksense.features import base
from stocksense.features.base import FeatureContext, FeatureSpec, feature
from stocksense.features.build import (
    build_features,
    cross_section_transform,
    current_version,
    load_features,
)
from stocksense.features.registry import REGISTRY, load_all
from tests.fixtures.synth import _frame, _settings_for, make_market, write_market

# ---- helpers: hand-made series -----------------------------------------------------------------


def _eval(exprs: dict[str, pl.Expr], sx: list[int], x: list, y: list | None = None, sid=None) -> pl.DataFrame:
    n = len(sx)
    data = {"sid": sid if sid is not None else [1] * n, "sx": sx, "x": x}
    if y is not None:
        data["y"] = y
    df = pl.DataFrame(data, schema_overrides={"x": pl.Float64, "sx": pl.Int64, "sid": pl.Int32})
    if y is not None:
        df = df.with_columns(pl.col("y").cast(pl.Float64))
    return df.with_columns(**exprs)


_SX = list(range(6))
_X = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_roll_sum_mean_by_hand():
    df = _eval({"s": base.roll_sum("x", 3), "m": base.roll_mean("x", 3)}, _SX, _X)
    # windows of 3: 1+2+3=6, 2+3+4=9, 12, 15; the first two rows have fewer than 3 values
    assert df["s"].to_list() == [None, None, 6.0, 9.0, 12.0, 15.0]
    assert df["m"].to_list() == [None, None, 2.0, 3.0, 4.0, 5.0]


def test_roll_std_is_sample_std():
    df = _eval({"s": base.roll_std("x", 3)}, _SX, _X)
    # std([1,2,3], ddof=1) = 1 for every consecutive triple
    assert df["s"].to_list() == [None, None, pytest.approx(1.0), 1.0, 1.0, 1.0]
    # a non-trivial window: [1, 3, 8] mean 4 -> (9 + 1 + 16)/2 = 13 -> sqrt(13)
    d2 = _eval({"s": base.roll_std("x", 3)}, [0, 1, 2], [1.0, 3.0, 8.0])
    assert d2["s"][2] == pytest.approx(13**0.5)


def test_roll_min_max_by_hand():
    x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    df = _eval({"lo": base.roll_min("x", 3), "hi": base.roll_max("x", 3)}, _SX, x)
    # windows: [3,1,4] [1,4,1] [4,1,5] [1,5,9]
    assert df["lo"].to_list() == [None, None, 1.0, 1.0, 1.0, 1.0]
    assert df["hi"].to_list() == [None, None, 4.0, 4.0, 5.0, 9.0]


def test_roll_corr_cov_var_by_hand():
    df = _eval(
        {
            "c_pos": base.roll_corr("x", "y", 3),
            "c_neg": base.roll_corr("x", pl.col("y") * -1.0, 3),
            "cov": base.roll_cov("x", "y", 3),
            "var": base.roll_var("x", 3),
        },
        _SX,
        _X,
        [2.0 * v for v in _X],
    )
    # y = 2x: corr 1; sample cov([1,2,3],[2,4,6]) = 2; sample var([1,2,3]) = 1
    assert df["c_pos"][5] == pytest.approx(1.0)
    assert df["c_neg"][5] == pytest.approx(-1.0)
    assert df["cov"][2] == pytest.approx(2.0)
    assert df["var"][2] == pytest.approx(1.0)
    assert df["cov"][0] is None


def test_roll_slope_r2_by_hand():
    slope, r2 = base.roll_slope_r2("x", 3)
    df = _eval({"slope": slope, "r2": r2}, [0, 1, 2, 3], [1.0, 3.0, 2.0, 8.0])
    # window [1,3,2] on rows 0,1,2: mean y 2, cov = ((-1)(-1) + 0 + (1)(0))/2 = 0.5, var(row) = 1,
    # var(y) = (1+1+0)/2 = 1 -> slope 0.5, corr 0.5, r2 0.25
    assert df["slope"][2] == pytest.approx(0.5)
    assert df["r2"][2] == pytest.approx(0.25)
    # window [3,2,8]: mean 13/3; cov = ((-1)(3-13/3) + 0 + (1)(8-13/3))/2 = (4/3 + 11/3)/2 = 2.5 -> slope 2.5
    assert df["slope"][3] == pytest.approx(2.5)
    # exact line y = 2 row + 1 -> r2 1
    d2 = _eval({"slope": slope, "r2": r2}, [0, 1, 2], [1.0, 3.0, 5.0])
    assert d2["slope"][2] == pytest.approx(2.0) and d2["r2"][2] == pytest.approx(1.0)


def test_helpers_are_per_sid():
    df = _eval(
        {"s": base.roll_sum("x", 2)},
        [0, 1, 0, 1],
        [1.0, 2.0, 10.0, 20.0],
        sid=[1, 1, 2, 2],
    )
    # the first row of sid 2 does not borrow from sid 1: window sums are 3 and 30
    assert df["s"].to_list() == [None, 3.0, None, 30.0]


def test_wide_span_rule_nulls_a_window_over_a_suspension():
    # sessions 0..3, a suspension, then 10..12. w=3 allows a span of ceil(3.75)+2 = 6 sessions.
    sx = [0, 1, 2, 3, 10, 11, 12]
    df = _eval({"s": base.roll_sum("x", 3)}, sx, [1.0, 2, 3, 4, 5, 6, 7])
    # row sx=3: rows 1..3 -> 2+3+4 = 9 (span 3); sx=10: rows sx 2,3,10 span 9 > 6 -> null;
    # sx=11: rows 3,10,11 span 9 -> null; sx=12: rows 10,11,12 span 3 -> 5+6+7 = 18
    assert df["s"].to_list() == [None, None, 6.0, 9.0, None, None, 18.0]
    # exactly on the limit: span 6 is still valid
    d2 = _eval({"s": base.roll_sum("x", 3)}, [0, 1, 5], [1.0, 2.0, 3.0])
    assert d2["s"][2] == 6.0  # span 5 - 0 + 1 = 6 <= 6
    d3 = _eval({"s": base.roll_sum("x", 3)}, [0, 1, 6], [1.0, 2.0, 3.0])
    assert d3["s"][2] is None  # span 7 > 6


# ---- cross-sectional transform -----------------------------------------------------------------


def test_rank_scaling_by_hand():
    df = pl.DataFrame(
        {"date": [date(2020, 1, 6)] * 5, "sid": [1, 2, 3, 4, 5], "f": [10.0, 20.0, 20.0, 40.0, None]}
    )
    out = cross_section_transform(df, ["f"], min_valid=2).sort("sid")
    # n_valid = 4, average ranks 1, 2.5, 2.5, 4 -> (r-1)/3 - 0.5 = -0.5, 0.0, 0.0, 0.5; null stays null
    assert out["f"].to_list() == [pytest.approx(-0.5), pytest.approx(0.0), pytest.approx(0.0), 0.5, None]
    assert out["f"].dtype == pl.Float32 and out["f__raw"].dtype == pl.Float32


def test_rank_scaling_all_null_below_min_valid():
    d = date(2020, 1, 6)
    df = pl.DataFrame({"date": [d] * 4, "sid": [1, 2, 3, 4], "f": [1.0, 2.0, 3.0, 4.0]})
    assert cross_section_transform(df, ["f"], min_valid=5)["f"].null_count() == 4
    assert cross_section_transform(df, ["f"], min_valid=4)["f"].null_count() == 0
    # the winsorised raw value is kept even when the rank is null: 1% quantile of [1,2,3,4] is
    # 1 + 0.03 = 1.03 and the 99% quantile is 4 - 0.03 = 3.97
    raw = cross_section_transform(df, ["f"], min_valid=5).sort("sid")["f__raw"].to_list()
    assert raw == [pytest.approx(1.03), 2.0, 3.0, pytest.approx(3.97)]


def test_winsorize_at_1st_and_99th_percentile_and_per_date():
    d1, d2 = date(2020, 1, 6), date(2020, 1, 7)
    vals = [float(i) for i in range(1, 101)] + [10000.0]
    df = pl.DataFrame(
        {"date": [d1] * 101 + [d2] * 3, "sid": list(range(101)) + [0, 1, 2], "f": vals + [5.0, 6.0, 7.0]}
    )
    out = cross_section_transform(df, ["f"], min_valid=50)
    day1 = out.filter(pl.col("date") == d1)
    # 101 sorted values: linear 1% quantile = sorted[1] = 2, 99% quantile = sorted[99] = 100
    assert day1["f__raw"].min() == 2.0
    assert day1["f__raw"].max() == 100.0
    # the ranked column still puts the outlier on top (ranks are on the winsorised values: a tie at 100)
    assert day1.filter(pl.col("sid") == 100)["f"].item() == pytest.approx(0.5 - 0.5 / 100, abs=0.01)
    # date 2 has 3 values (< 50): ranks all null; raw is winsorised on its own date only
    # (1% quantile of [5,6,7] = 5.02, 99% = 6.98)
    day2 = out.filter(pl.col("date") == d2)
    assert day2["f"].null_count() == 3
    assert sorted(day2["f__raw"].to_list()) == [pytest.approx(5.02), 6.0, pytest.approx(6.98)]


# ---- decorator and registry --------------------------------------------------------------------


def _fn(ctx: FeatureContext) -> pl.LazyFrame:
    return ctx.panel().select("date", "sid", pl.col("ret").alias("t_feat"))


def test_decorator_registers_spec_in_given_registry():
    reg: dict[str, FeatureSpec] = {}
    f = feature("t_feat", group="test", kind="xs", window=1, version=2, label="a test", registry=reg)(_fn)
    assert f is _fn
    spec = reg["t_feat"]
    assert (spec.name, spec.group, spec.kind, spec.window, spec.version, spec.label) == (
        "t_feat",
        "test",
        "xs",
        1,
        2,
        "a test",
    )
    assert spec.outputs == ("t_feat",)
    with pytest.raises(ValueError, match="already registered"):
        feature("t_feat", group="test", registry=reg)(_fn)
    with pytest.raises(ValueError, match="kind"):
        feature("t_other", group="test", kind="zz", registry=reg)(_fn)


def test_family_outputs_must_not_collide():
    reg: dict[str, FeatureSpec] = {}
    feature("a", group="g", outputs=("a", "a_2"), registry=reg)(_fn)
    with pytest.raises(ValueError, match="already registered"):
        feature("b", group="g", outputs=("b", "a_2"), registry=reg)(_fn)


def test_global_registry_has_the_seed_features():
    reg = load_all()
    assert reg is REGISTRY
    assert {"ret_1", "vol_21", "mkt_ret_1"} <= set(reg)
    assert reg["ret_1"].kind == "xs" and reg["mkt_ret_1"].kind == "mkt"
    assert all(s.label for s in reg.values())


# ---- context: the as_of choke point ------------------------------------------------------------


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    from stocksense import panel

    root = tmp_path_factory.mktemp("feat_small")
    m = make_market(n=12, n_sessions=120)
    write_market(m, root)
    s = _settings_for(root)
    s = s.model_copy(
        update={
            "universe": s.universe.model_copy(
                update={
                    "min_history_sessions": 5,
                    "min_traded_of_last_60": 5,
                    "min_close": 1.0,
                    "min_med_turnover_inr": 1.0,
                    "u200_min_med_turnover_inr": 1.0,
                }
            )
        }
    )
    panel.build_all(s)
    return s, m


def test_context_panel_has_sx_and_respects_as_of(small):
    s, m = small
    cut = m.sessions[60]
    full = FeatureContext(s).panel().collect()
    cutp = FeatureContext(s, as_of=cut).panel().collect()
    assert full["sx"].dtype == pl.Int64
    # sx is the calendar session index: the Saturday special session is index 120 of the 601-long list,
    # here (120 sessions) plain positions; the truncated frame keeps the same indices
    sess = sorted(m.eq["date"].unique().to_list())
    idx = {d: i for i, d in enumerate(sess)}
    assert (
        full.filter(pl.col("sid") == full["sid"][0])
        .sort("date")
        .select(pl.col("sx") == pl.col("date").replace_strict(idx, return_dtype=pl.Int64))["sx"]
        .all()
    )
    assert cutp["date"].max() == cut
    a = full.filter(pl.col("date") <= cut).sort("sid", "date")
    b = cutp.sort("sid", "date")
    assert a.select("sid", "date", "sx").equals(b.select("sid", "date", "sx"))
    # market and universe are filtered too
    assert FeatureContext(s, as_of=cut).market().collect()["date"].max() <= cut
    assert FeatureContext(s, as_of=cut).universe().collect()["date"].max() <= cut
    # sorted by sid, date
    assert full.select("sid", "date").equals(full.select("sid", "date").sort("sid", "date"))


def test_context_raw_filters_timestamps_at_2000_ist(small):
    s, m = small
    ann = _frame(
        {
            "seq_id": pl.Utf8,
            "symbol": pl.Utf8,
            "ts": pl.Datetime("us", "Asia/Kolkata"),
            "category": pl.Utf8,
        },
        {
            "seq_id": ["a", "b", "c", "d"],
            "symbol": ["S00"] * 4,
            "ts": [
                datetime(2018, 3, 1, 19, 59, tzinfo=IST),
                datetime(2018, 3, 1, 20, 0, tzinfo=IST),
                datetime(2018, 3, 1, 20, 1, tzinfo=IST),
                datetime(2018, 3, 2, 9, 0, tzinfo=IST),
            ],
            "category": ["x"] * 4,
        },
    )
    io.upsert(s, "announcements", ann)
    got = FeatureContext(s, as_of=date(2018, 3, 1)).raw("announcements").collect()
    # 19:59 and exactly 20:00 are in; 20:01 and the next morning are out
    assert sorted(got["seq_id"].to_list()) == ["a", "b"]
    assert FeatureContext(s).raw("announcements").collect().height == 4


def test_context_raw_filters_available_ts_and_date(small):
    s, _ = small
    macro = _frame(
        {
            "series": pl.Utf8,
            "date": pl.Date,
            "close": pl.Float64,
            "available_ts": pl.Datetime("us", "Asia/Kolkata"),
        },
        {
            "series": ["SPX", "SPX"],
            "date": [date(2018, 3, 1), date(2018, 3, 1)],
            # same value date, but the second row became available after the 20:00 IST cutoff of the 1st
            "close": [1.0, 2.0],
            "available_ts": [
                datetime(2018, 3, 1, 19, 0, tzinfo=IST),
                datetime(2018, 3, 2, 2, 30, tzinfo=IST),
            ],
        },
    )
    io.upsert(s, "macro_daily", macro.head(1))
    io.upsert(s, "macro_daily", macro.tail(1).with_columns(pl.lit("SPX2").alias("series")))
    got = FeatureContext(s, as_of=date(2018, 3, 1)).raw("macro_daily").collect()
    assert got["series"].to_list() == ["SPX"]
    # a date-keyed dataset is filtered on date
    got = FeatureContext(s, as_of=date(2018, 1, 10)).raw("bhavcopy_eq", columns=["symbol", "date"]).collect()
    assert got["date"].max() <= date(2018, 1, 10)


# ---- build and load ----------------------------------------------------------------------------


def _clock() -> FixedClock:
    return FixedClock(datetime(2026, 11, 12, 10, 0, tzinfo=IST))


@pytest.fixture
def built(tmp_path):
    from stocksense import panel

    m = make_market(n=12, n_sessions=120)
    write_market(m, tmp_path)
    s = _settings_for(tmp_path)
    s = s.model_copy(
        update={
            "universe": s.universe.model_copy(
                update={
                    "min_history_sessions": 5,
                    "min_traded_of_last_60": 5,
                    "min_close": 1.0,
                    "min_med_turnover_inr": 1.0,
                    "u200_min_med_turnover_inr": 1.0,
                }
            )
        }
    )
    panel.build_all(s)
    return s, m


def test_build_writes_parts_manifest_and_current(built):
    s, m = built
    v = build_features(s, min_valid=5, clock=_clock())
    assert v == 1 and current_version(s) == 1
    root = s.paths.data_root / "features"
    assert (root / "CURRENT").read_text(encoding="utf-8").strip() == "1"
    assert (root / "v1" / "part-2018.parquet").is_file()
    man = json.loads((root / "v1" / "manifest.json").read_text(encoding="utf-8"))
    assert man["version"] == 1 and man["created"].startswith("2026-11-12")
    names = {f["name"] for f in man["features"]}
    # every registered output is listed, with its metadata
    assert names == {o for sp in load_all().values() for o in sp.outputs}
    by = {f["name"]: f for f in man["features"]}
    assert by["ret_1"]["group"] and by["ret_1"]["kind"] == "xs" and by["mkt_ret_1"]["kind"] == "mkt"
    assert by["ret_1"]["label"] and by["ret_1"]["version"] == 1
    assert 0.0 < by["vol_21"]["coverage"] <= 1.0
    # vol_21 needs 21 rows: its first valid date is the 21st session (index 20) at the earliest
    assert date.fromisoformat(by["vol_21"]["first_valid_date"]) >= sorted(m.eq["date"].unique())[20]
    df = io.read(s, "features", sub="v1").collect()
    assert df.select("date", "sid").is_duplicated().sum() == 0
    assert {"ret_1", "ret_1__raw", "vol_21", "vol_21__raw", "mkt_ret_1"} <= set(df.columns)
    assert df["ret_1"].dtype == pl.Float32 and df["mkt_ret_1"].dtype == pl.Float32
    # ranks in [-0.5, 0.5]; market feature identical for every sid on a date; no rows outside u500
    assert df["ret_1"].drop_nulls().min() >= -0.5 and df["ret_1"].drop_nulls().max() <= 0.5
    assert (
        df.group_by("date").agg(pl.col("mkt_ret_1").n_unique().alias("n")).filter(pl.col("n") > 1).is_empty()
    )
    u = io.read(s, "panel_universe").filter(pl.col("u500")).select("date", "sid").collect()
    assert df.join(u, on=["date", "sid"], how="anti").is_empty()


def test_build_values_match_panel_by_hand(built):
    s, _ = built
    build_features(s, min_valid=5, clock=_clock())
    f = io.read(s, "features", sub="v1").collect()
    p = io.read(s, "panel_daily", columns=["sid", "date", "ret"]).collect()
    j = f.join(p, on=["sid", "date"])
    # ret_1 = ln(1 + ret), kept as the winsorised raw value; compare where not at a winsorised tail
    # away from the per-date winsorised extremes, the stored raw value is exactly ln(1 + ret)
    j = j.filter(pl.col("ret_1__raw").is_not_null()).with_columns(
        pl.col("ret_1__raw").min().over("date").alias("_lo"),
        pl.col("ret_1__raw").max().over("date").alias("_hi"),
    )
    mid = j.filter((pl.col("ret_1__raw") > pl.col("_lo")) & (pl.col("ret_1__raw") < pl.col("_hi")))
    assert mid.height > 500
    assert ((mid["ret_1__raw"].cast(pl.Float64) - (1.0 + mid["ret"]).log()).abs() < 1e-6).all()


def test_changed_feature_version_bumps_directory(built):
    s, _ = built
    reg = dict(load_all())
    assert build_features(s, min_valid=5, registry=reg, clock=_clock()) == 1
    # same set of (name, version): rebuild stays in v1
    assert build_features(s, min_valid=5, registry=reg, clock=_clock()) == 1
    old = reg["ret_1"]
    reg["ret_1"] = FeatureSpec(
        old.name, old.group, old.kind, old.fn, old.window, old.version + 1, old.label, old.outputs
    )
    assert build_features(s, min_valid=5, registry=reg, clock=_clock()) == 2
    root = s.paths.data_root / "features"
    assert current_version(s) == 2 and (root / "v1").is_dir() and (root / "v2" / "manifest.json").is_file()
    man = json.loads((root / "v2" / "manifest.json").read_text(encoding="utf-8"))
    assert {f["name"]: f["version"] for f in man["features"]}["ret_1"] == 2
    # a dropped feature is also a changed set
    del reg["vol_21"]
    assert build_features(s, min_valid=5, registry=reg, clock=_clock()) == 3


def test_incremental_build_rewrites_recent_dates_only(built):
    s, m = built
    build_features(s, min_valid=5, clock=_clock())
    path = s.paths.data_root / "features" / "v1" / "part-2018.parquet"
    full = io.read(s, "features", sub="v1").collect().sort("date", "sid")
    start = m.sessions[100]
    # corrupt the tail of the stored file, then an incremental build from `start` must repair it
    bad = full.with_columns(
        pl.when(pl.col("date") >= start)
        .then(0.123)
        .otherwise(pl.col("ret_1__raw"))
        .cast(pl.Float32)
        .alias("ret_1__raw")
    )
    io.write_partition(s, "features", "2018", bad.filter(pl.col("date").dt.year() == 2018), sub="v1")
    assert path.is_file()
    build_features(s, start=start, min_valid=5, clock=_clock())
    again = io.read(s, "features", sub="v1").collect().sort("date", "sid")
    assert again.select("date", "sid").equals(full.select("date", "sid"))
    assert again["ret_1__raw"].to_list() == full["ret_1__raw"].to_list()


def test_end_argument_truncates_build(built):
    s, m = built
    build_features(s, end=m.sessions[80], min_valid=5, clock=_clock())
    assert io.read(s, "features", sub="v1").collect()["date"].max() <= m.sessions[80]


def test_load_features_purposes(built):
    s, m = built
    build_features(s, min_valid=5, clock=_clock())
    cut = m.sessions[70]
    s2 = s.model_copy(update={"research": s.research.model_copy(update={"lockbox_start": cut})})
    prod = load_features(s2, "production").collect()
    res = load_features(s2, "research").collect()
    # research stops the day before lockbox_start; production sees everything
    assert prod["date"].max() > cut
    assert res["date"].max() == m.sessions[69]
    with pytest.raises(LockboxError):
        load_features(s2, "lockbox")
    sub = load_features(
        s2, "production", start=m.sessions[10], end=m.sessions[20], columns=["ret_1"]
    ).collect()
    assert set(sub.columns) == {"date", "sid", "ret_1"}
    assert sub["date"].min() >= m.sessions[10] and sub["date"].max() <= m.sessions[20]
