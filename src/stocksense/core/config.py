from __future__ import annotations

import os
import tomllib
from datetime import date, time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from stocksense.core.errors import ConfigError


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Paths(_Frozen):
    data_root: Path


class Time(_Frozen):
    tz: str
    decision_cutoff: time

    @field_validator("decision_cutoff", mode="before")
    @classmethod
    def _parse_cutoff(cls, v: Any) -> Any:
        if isinstance(v, str):
            return time.fromisoformat(v)
        return v


class Research(_Frozen):
    dev_start: date
    wf_test_start: date
    lockbox_start: date


class Runtime(_Frozen):
    threads: int
    gpu: bool


class Universe(_Frozen):
    series_allowed: tuple[str, ...]
    min_history_sessions: int
    min_traded_of_last_60: int
    min_close: float
    min_med_turnover_inr: float
    u500_size: int
    u500_exit_rank: int
    u200_size: int
    u200_exit_rank: int
    u200_min_med_turnover_inr: float
    etf_name_regex: str


class ProductCosts(_Frozen):
    stt_buy: float
    stt_sell: float
    stamp_buy: float
    exchange: float
    sebi: float
    gst: float
    dp_per_scrip_sell_inr: float
    brokerage_rate: float
    brokerage_cap_inr: float
    brokerage_min_inr: float


class Slippage(_Frozen):
    impact_k: float
    open_penalty_bps: float
    close_penalty_bps: float
    half_spread_bps_by_rank: tuple[tuple[int, float], ...]


class Grid(_Frozen):
    capital_inr: tuple[int, ...]


class Costs(_Frozen):
    verified_on: date
    delivery: ProductCosts
    intraday: ProductCosts
    slippage: Slippage
    grid: Grid


class GateResearch(_Frozen):
    min_mean_rank_ic: float
    min_ic_tstat: float
    min_net_sharpe: float
    min_net_sharpe_2x_slip: float
    min_excess_tstat: float
    min_frac_positive_years: float
    min_dsr: float
    max_pbo: float
    max_drawdown: float
    reference_capital_inr: float


class GateLockbox(_Frozen):
    min_net_sharpe: float
    min_net_return: float


class GatePaper(_Frozen):
    min_sessions: int
    min_net_sharpe: float
    max_drawdown: float
    max_ic_shortfall: float


class GatePromotion(_Frozen):
    min_shadow_sessions: int
    max_p_value: float


class GateAuction(_Frozen):
    min_strict_capture: float


class Gate(_Frozen):
    research: GateResearch
    lockbox: GateLockbox
    paper: GatePaper
    promotion: GatePromotion
    auction: GateAuction


class S1(_Frozen):
    product: str
    entry: str
    exit: str
    targets: tuple[str, ...]
    universe: str
    long_only: bool
    k_max: int


class S2(_Frozen):
    product: str
    entry: str
    exit: str
    targets: tuple[str, ...]
    universe: str
    long_only: bool
    shorts_require_fno: bool
    k_max: int


class S3(_Frozen):
    enabled: bool
    max_capital_frac: float


class Strategies(_Frozen):
    s1: S1
    s2: S2
    s3: S3


class Risk(_Frozen):
    target_vol_annual: float
    max_gross: float
    max_name_weight: float
    max_industry_weight: float
    max_adv_participation: float
    max_fixed_cost_bps: float
    min_position_inr_floor: float
    stop_min: float
    stop_vol_mult: float
    book_dd_half: float
    book_dd_stop: float
    no_entry_sessions_before_results: int
    no_entry_sessions_before_split: int
    s2_free_cash_frac: float


class SpecialSession(_Frozen):
    date: date
    open: time = time(9, 15)
    close: time = time(15, 30)
    tradable: bool = True

    @field_validator("open", "close", mode="before")
    @classmethod
    def _parse_hhmm(cls, v: Any) -> Any:
        if isinstance(v, str):
            return time.fromisoformat(v)
        return v


class Schedule(BaseModel):
    """config/schedule.toml: special sessions plus the raw `[[job]]` rows (parsed by live.schedule)."""

    model_config = ConfigDict(frozen=True, extra="allow")
    special_sessions: tuple[SpecialSession, ...] = ()
    job: tuple[dict[str, Any], ...] = ()


class Secrets(BaseSettings):
    """Credentials. Every set value prints as *** in repr/str."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    angel_api_key: str | None = Field(None, validation_alias=AliasChoices("STOCKSENSE_ANGEL_API_KEY"))
    angel_client_id: str | None = Field(None, validation_alias=AliasChoices("STOCKSENSE_ANGEL_CLIENT_CODE"))
    angel_pin: str | None = Field(None, validation_alias=AliasChoices("STOCKSENSE_ANGEL_PASSWORD"))
    angel_totp_secret: str | None = Field(None, validation_alias=AliasChoices("STOCKSENSE_ANGEL_TOTP_SECRET"))
    upstox_analytics_token: str | None = Field(
        None,
        validation_alias=AliasChoices("STOCKSENSE_UPSTOX_ANALYTICS_TOKEN", "STOCKSENSE_UPSTOX_ACCESS_TOKEN"),
    )
    telegram_bot_token: str | None = Field(
        None, validation_alias=AliasChoices("STOCKSENSE_TELEGRAM_BOT_TOKEN")
    )
    telegram_chat_id: str | None = Field(
        None, validation_alias=AliasChoices("STOCKSENSE_TELEGRAM_ALLOWED_CHAT_ID")
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Only explicit input: load_settings() merges os.environ and the .env file itself.
        return (init_settings,)

    def secret_values(self) -> list[str]:
        """The non-empty secret values, for log scrubbing only. Never log or print the result."""
        return [v for v in self.model_dump().values() if v]

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={'***' if v else 'None'}" for k, v in self.model_dump().items())
        return f"Secrets({inner})"

    __str__ = __repr__


class Settings(_Frozen):
    paths: Paths
    time: Time
    research: Research
    runtime: Runtime
    universe: Universe
    costs: Costs
    gate: Gate
    strategies: Strategies
    risk: Risk
    schedule: Schedule
    secrets: Secrets

    def require(self, *names: str) -> None:
        """Raise ConfigError listing every named secret that is unset."""
        missing = [n for n in names if not getattr(self.secrets, n, None)]
        if missing:
            raise ConfigError("missing secret: " + ", ".join(missing))


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name}: invalid TOML: {e}") from e


def _validate_settings(data: dict[str, Any]) -> Settings:
    try:
        return Settings.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors())
        raise ConfigError(f"bad config: {problems}") from e


def load_settings(config_dir: Path = Path("config"), env_file: Path = Path(".env")) -> Settings:
    """Load every config/*.toml plus secrets from the environment and env_file (env wins)."""
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise ConfigError(f"config directory not found: {config_dir}")
    default = _read_toml(config_dir / "default.toml")
    paths = dict(default.get("paths", {}))
    root = Path(paths.get("data_root", "data_store"))
    if not root.is_absolute():
        root = config_dir.resolve().parent / root
    paths["data_root"] = root
    gate_path = config_dir / "gate.toml"
    if not gate_path.is_file():
        raise ConfigError(f"gate.toml not found in {config_dir} (it is frozen, see T1.7)")
    sched_path = config_dir / "schedule.toml"
    env: dict[str, str] = {}
    if Path(env_file).is_file():
        env.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    env.update(os.environ)
    env = {k: v for k, v in env.items() if k.startswith("STOCKSENSE_") and v.strip() != ""}
    payload = {
        "paths": paths,
        "time": default.get("time", {}),
        "research": default.get("research", {}),
        "runtime": default.get("runtime", {}),
        "universe": _read_toml(config_dir / "universe.toml"),
        "costs": _read_toml(config_dir / "costs.toml"),
        "gate": _read_toml(gate_path),
        "strategies": _read_toml(config_dir / "strategies.toml"),
        "risk": _read_toml(config_dir / "risk.toml"),
        "schedule": _read_toml(sched_path) if sched_path.is_file() else {},
        "secrets": Secrets(**env),
    }
    return _validate_settings(payload)
