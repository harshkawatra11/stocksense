"""G2-C5: the g2-s1cost grid plumbing (policy options, family, 16 trials, CLI)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from stocksense.validation.g1 import FAMILY_S1, G1Config, ModelTrial, _spec_s1, plan_grid
from tests.fixtures.engine_panel import settings, zero_slippage

S = settings()
CFG = G1Config(capitals=(5e5,), reference_capital=5e5, start=None)


def test_policy_is_a_spec_key_only_when_set():
    base = _spec_s1(CFG, "y_c2o_10", 10, False, [])
    assert "policy" not in base
    spec = _spec_s1(replace(CFG, s1_policy={"exit_rank": 50, "min_hold": 5}), "y_c2o_10", 10, False, [])
    assert spec["policy"] == {"exit_rank": 50, "min_hold": 5}


def test_family_s1_defaults_to_the_g1_family_and_is_used_by_plan_grid():
    m = [ModelTrial("s1-gbm-y_c2o_10-x", "s1-gbm", "y_c2o_10", {}, 0.07, 20.0)]
    assert CFG.family_s1 == FAMILY_S1
    plans = plan_grid(replace(CFG, family_s1="g2-s1cost"), {"y_c2o_10": m}, [])
    assert plans and {p.family for p in plans} == {"g2-s1cost"}
    assert {p.family for p in plan_grid(CFG, {"y_c2o_10": m}, [])} == {FAMILY_S1}


def test_the_grid_is_16_distinct_trials_in_family_g2_s1cost():
    from stocksense.validation.g1 import plan_s1cost

    models = {t: [ModelTrial(f"s1-gbm-{t}-x", "s1-gbm", t, {}, 0.07, 20.0)] for t in ("y_c2o_10", "y_c2o_20")}
    plans = plan_s1cost(CFG, models, k_max=10, overlay=False)
    assert len(plans) == 16
    assert {p.plan.family for p in plans} == {"g2-s1cost"}
    assert len({repr(p.plan.spec) for p in plans}) == 16
    by = {(p.variant, p.plan.target): p for p in plans}
    v1 = by[("V1", "y_c2o_10")]
    assert v1.plan.spec["policy"] == {"exit_rank": 50, "min_hold": 5}  # h//2 for h = 10
    assert by[("V1", "y_c2o_20")].plan.spec["policy"]["min_hold"] == 10
    for p in plans:
        assert p.plan.spec["universe_flag"] is True and p.plan.spec["auto_resume_sessions"] == 20
        assert p.plan.spec["k_max"] == 10 and p.plan.spec["overlay"] is False
    v2 = by[("V2", "y_c2o_10")].plan.spec
    assert v2["policy"] == {"exit_rank": 50, "min_hold": 5, "free_extension": True, "rebuy_block": 5}
    assert "score_ema" not in v2 and "er_map" not in v2
    assert by[("V3", "y_c2o_10")].plan.spec["score_ema"] == 3
    assert by[("V4", "y_c2o_10")].plan.spec["score_ema"] == 5
    v5 = by[("V5", "y_c2o_10")].plan.spec
    assert v5["policy"]["switch_hurdle"] is True and v5["score_ema"] == 5 and "er_map" not in v5
    v6 = by[("V6", "y_c2o_10")].plan.spec
    assert v6["er_map"] == "decile" and v6["policy"] == v5["policy"]
    assert by[("V7", "y_c2o_10")].plan.spec["policy"]["exit_rank"] == 75
    v8 = by[("V8", "y_c2o_20")].plan.spec["policy"]
    assert v8["max_fixed_cost_bps"] == 10 and v8["exit_rank"] == 50
    # the variant's own config carries the G1Config fields the spec records
    assert by[("V6", "y_c2o_10")].cfg.er_map == "decile" and by[("V6", "y_c2o_10")].cfg.score_ema == 5


def test_a_target_without_models_contributes_nothing():
    from stocksense.validation.g1 import plan_s1cost

    assert plan_s1cost(CFG, {"y_c2o_10": [], "y_c2o_20": []}, k_max=10, overlay=False) == []


def test_policy_option_max_fixed_cost_bps_overrides_risk():
    from stocksense.portfolio.construct import S1Policy

    def pol(**o):
        return S1Policy(
            SimpleNamespace(target="y_c2o_10"), S, None, None, S.costs, zero_slippage(), k_max=10,
            policy_opts=o or None,
        )  # fmt: skip

    base, tight = pol(), pol(max_fixed_cost_bps=10.0, exit_rank=50)
    assert base.cfg.risk.max_fixed_cost_bps == 15.0 and tight.cfg.risk.max_fixed_cost_bps == 10.0
    assert tight.cfg.min_pos > base.cfg.min_pos and tight.cfg.exit_rank == 50
    assert S.risk.max_fixed_cost_bps == 15.0  # settings untouched


def test_s1_cost_command_is_registered():
    from typer.testing import CliRunner

    from stocksense.cli import app

    r = CliRunner().invoke(app, ["research", "s1-cost", "--help"])
    assert r.exit_code == 0 and "--targets" in r.output


def test_strategies_toml_carries_the_s1_policy_table_that_s1config_accepts(tmp_path):
    from pathlib import Path

    from stocksense.core.config import load_settings
    from stocksense.portfolio.construct import S1Config

    real = Path(__file__).resolve().parents[2] / "config"
    s = load_settings(real, tmp_path / "no.env")
    pol = dict(s.strategies.s1.policy)
    assert pol, "[s1.policy] is written after the C5 verdict"
    opts = dict(pol)
    opts.pop("max_fixed_cost_bps", None)  # a Risk override, handled by S1Policy
    cfg = S1Config.from_settings(s, 10, zero_slippage(), **opts)
    assert cfg.exit_rank == pol["exit_rank"]
