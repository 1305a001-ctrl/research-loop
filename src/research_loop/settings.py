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

    # --- Learning loop ---
    # Master switch. Disabling skips schema-ensure + all sub-loops.
    learning_loop_enabled: bool = True

    # Outcome join cadence — converts open feature_outcomes rows into
    # closed ones by reading positions table. 5 min default; fast enough
    # to catch closes before the next auto-tune cycle.
    outcome_join_interval_sec: int = 300

    # Auto-tuner runtime — OFF by default. When enabled, runs the
    # per-strategy param-nudge cycle on auto_tuner_interval_sec cadence.
    # Use auto_tuner_dry_run=true to log decisions without applying them
    # — recommended for the first 7+ days of operation.
    auto_tuner_enabled: bool = False
    auto_tuner_interval_sec: int = 3600
    auto_tuner_dry_run: bool = True            # default-deny: log only
    auto_tuner_min_samples: int = 30           # gate: need ≥30 closes per strategy
    auto_tuner_days_window: int = 14           # lookback for fetch_recent_closed
    auto_tuner_closed_limit: int = 100         # max rows per strategy per cycle
    # Comma-separated strategy slugs to opt into tuning. Empty = none.
    # Use "all" to opt-in every strategy in auto_tuner_runtime.TUNABLE_PARAMS.
    auto_tuner_strategies: str = ""

    # Training export — nightly Parquet/JSONL snapshots for offline ML.
    training_export_enabled: bool = True
    training_export_interval_sec: int = 86_400   # 24h
    training_export_dir: str = "/srv/data/training"

    # --- Halt-fast trigger (venue-scoped 24h PnL circuit breaker) ---
    # When True, runs the halt-fast loop. Respects `enforce_halts` for
    # the actual SADD — so this can be enabled in dry-run + observed
    # for a few cycles before flipping enforce.
    halt_fast_enabled: bool = True
    # Floor for the 24h realised PnL. Negative — trip when below.
    halt_fast_pnl_floor_usd_24h: float = -50.0
    # Comma-separated list of venues to scope. Comes from env as a CSV.
    halt_fast_venues: str = "polymarket"
    # Sub-1 minute would hammer the DB; 60s is the safe minimum.
    halt_fast_interval_sec: int = 60
    # Don't trip on a single edge-case close; 1 is the cautious default,
    # bump if you see spurious trips from edge-case fills.
    halt_fast_min_n_closed: int = 1


settings = Settings()
