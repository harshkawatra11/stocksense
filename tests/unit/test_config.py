from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from stocksense.core.config import load_settings
from stocksense.core.errors import ConfigError

REAL_CONFIG = Path(__file__).resolve().parents[2] / "config"
SECRET_VARS = [
    "STOCKSENSE_ANGEL_API_KEY",
    "STOCKSENSE_ANGEL_CLIENT_CODE",
    "STOCKSENSE_ANGEL_PASSWORD",
    "STOCKSENSE_ANGEL_TOTP_SECRET",
    "STOCKSENSE_UPSTOX_ANALYTICS_TOKEN",
    "STOCKSENSE_UPSTOX_ACCESS_TOKEN",
    "STOCKSENSE_TELEGRAM_BOT_TOKEN",
    "STOCKSENSE_TELEGRAM_ALLOWED_CHAT_ID",
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for v in SECRET_VARS:
        monkeypatch.delenv(v, raising=False)
    return tmp_path / "no.env"  # does not exist


def test_defaults_load_from_real_config(clean_env):
    s = load_settings(REAL_CONFIG, clean_env)
    # values copied from PLAN D.7
    assert s.research.lockbox_start == date(2025, 10, 1)
    assert s.costs.delivery.stt_buy == 0.001
    assert s.costs.slippage.half_spread_bps_by_rank[0] == (50, 1.5)
    assert s.universe.u500_size == 500
    assert s.strategies.s1.k_max == 20
    assert s.risk.max_gross == 1.0
    assert s.time.tz == "Asia/Kolkata"
    assert s.runtime.threads == 8
    assert all(v is None for v in s.secrets.model_dump().values())


def test_settings_frozen(clean_env):
    s = load_settings(REAL_CONFIG, clean_env)
    with pytest.raises(ValidationError):
        s.risk = None  # type: ignore[misc]


def test_env_alias_overrides_and_first_nonempty_wins(clean_env, monkeypatch):
    monkeypatch.setenv("STOCKSENSE_ANGEL_CLIENT_CODE", "C123")
    monkeypatch.setenv("STOCKSENSE_UPSTOX_ANALYTICS_TOKEN", "")
    monkeypatch.setenv("STOCKSENSE_UPSTOX_ACCESS_TOKEN", "ACC")
    s = load_settings(REAL_CONFIG, clean_env)
    assert s.secrets.angel_client_id == "C123"
    assert s.secrets.upstox_analytics_token == "ACC"  # analytics empty -> falls back to access token
    monkeypatch.setenv("STOCKSENSE_UPSTOX_ANALYTICS_TOKEN", "ANA")
    assert load_settings(REAL_CONFIG, clean_env).secrets.upstox_analytics_token == "ANA"


def test_env_file_is_read(clean_env, tmp_path):
    f = tmp_path / "t.env"
    f.write_text(
        "STOCKSENSE_TELEGRAM_ALLOWED_CHAT_ID=999\nSTOCKSENSE_ANGEL_PASSWORD=1234\n", encoding="utf-8"
    )
    s = load_settings(REAL_CONFIG, f)
    assert s.secrets.telegram_chat_id == "999"
    assert s.secrets.angel_pin == "1234"


def test_require_lists_all_missing(clean_env, monkeypatch):
    monkeypatch.setenv("STOCKSENSE_ANGEL_API_KEY", "k")
    s = load_settings(REAL_CONFIG, clean_env)
    s.require("angel_api_key")  # present: no error
    with pytest.raises(ConfigError) as e:
        s.require("angel_api_key", "angel_pin", "angel_totp_secret")
    assert "angel_pin" in str(e.value) and "angel_totp_secret" in str(e.value)
    assert "angel_api_key" not in str(e.value)


def test_repr_masks_secrets(clean_env, monkeypatch):
    monkeypatch.setenv("STOCKSENSE_ANGEL_PASSWORD", "supersecretpin")
    monkeypatch.setenv("STOCKSENSE_TELEGRAM_BOT_TOKEN", "tok-abc-987")
    s = load_settings(REAL_CONFIG, clean_env)
    for text in (repr(s.secrets), str(s.secrets), repr(s)):
        assert "supersecretpin" not in text and "tok-abc-987" not in text
    assert "***" in repr(s.secrets)


def test_wrong_type_raises_config_error_naming_key(clean_env, tmp_path):
    cfg = tmp_path / "config"
    shutil.copytree(REAL_CONFIG, cfg)
    p = cfg / "risk.toml"
    p.write_text(
        p.read_text(encoding="utf-8").replace("max_gross = 1.0", 'max_gross = "lots"'), encoding="utf-8"
    )
    with pytest.raises(ConfigError) as e:
        load_settings(cfg, clean_env)
    assert "max_gross" in str(e.value)


def test_gate_loaded_from_frozen_file(clean_env):
    s = load_settings(REAL_CONFIG, clean_env)
    assert s.gate.research.min_dsr == 0.95 and s.gate.research.max_pbo == 0.30
    assert s.gate.research.reference_capital_inr == 500000
    assert s.gate.lockbox.min_net_sharpe == 0.5 and s.gate.paper.min_sessions == 60
    assert s.gate.promotion.min_shadow_sessions == 20 and s.gate.auction.min_strict_capture == 0.5


def test_gate_file_is_required(tmp_path, clean_env):
    cfg = tmp_path / "config"
    shutil.copytree(REAL_CONFIG, cfg)
    (cfg / "gate.toml").unlink()
    with pytest.raises(ConfigError, match="gate.toml"):
        load_settings(cfg, clean_env)


def test_tmp_data_root_fixture_is_settings(tmp_data_root, tmp_data_dir):
    assert tmp_data_root.paths.data_root == tmp_data_dir
    assert (tmp_data_dir / "parquet").is_dir() and (tmp_data_dir / "state").is_dir()
