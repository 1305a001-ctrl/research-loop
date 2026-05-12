"""Env-driven settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Connectivity ---
    postgres_dsn: str = "postgresql://localhost:5432/control"
    redis_url: str = "redis://localhost:6379/0"

    # --- Measurement window ---
    # Window over which we measure each strategy. 7 days default — long
    # enough to smooth out daily noise, short enough that yesterday's
    # bleed doesn't get forgiven by last month's lucky run.
    window_days: int = 7

    # --- Halt thresholds (passed to should_halt) ---
    min_n_closed_for_halt: int = 30
    min_days_observed_for_halt: int = 7
    sharpe_halt_threshold: float = 0.0
    catastrophic_pnl_floor_usd: float = -200.0

    # --- Cycle cadence ---
    cycle_interval_sec: int = 3_600   # 1 hour — gives strategies room to recover

    # --- Halt actuation ---
    # When False, we LOG the would-halt decision but don't actually add
    # to strategy:halts. Set True to enforce.
    enforce_halts: bool = False

    # --- Halt key ---
    halts_redis_set: str = "strategy:halts"

    # --- Report destinations (stubs — wire in v0.2) ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- HTTP ---
    http_host: str = "0.0.0.0"  # noqa: S104 — bound to 127.0.0.1 in compose
    http_port: int = 8015
    log_level: str = "INFO"


settings = Settings()
