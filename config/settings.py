"""Centralized configuration for Project Titan.

Loads all configuration from environment variables and .env file using
pydantic-settings. Every tunable parameter in the system flows through
this module so there is a single source of truth.

Usage::

    from config.settings import get_settings

    settings = get_settings()
    print(settings.postgres_dsn)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Resolve the project root so the .env file can be located regardless of
# where the process is started (container, pytest, script, etc.).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class IBKRSettings(BaseSettings):
    """Interactive Brokers Gateway connection parameters."""

    model_config = SettingsConfigDict(env_prefix="IBKR_")

    username: str = Field(default="", description="IBKR account username")
    password: SecretStr = Field(
        default=SecretStr(""),
        description="IBKR account password",
    )
    trading_mode: Literal["paper", "live"] = Field(
        default="paper",
        description="Trading mode: 'paper' for simulated, 'live' for real money",
    )
    gateway_host: str = Field(
        default="ib-gateway",
        description="IB Gateway hostname (Docker service name or IP)",
    )
    gateway_port: int = Field(
        default=4004,
        description="IB Gateway API port (gnzsnz/ib-gateway paper default)",
    )
    client_id: int = Field(
        default=1,
        description="Unique client identifier for the API session",
    )


class PostgresSettings(BaseSettings):
    """PostgreSQL connection parameters."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = Field(default="postgres", description="PostgreSQL hostname")
    port: int = Field(default=5432, description="PostgreSQL port")
    db: str = Field(default="titan", description="Database name")
    user: str = Field(default="titan", description="Database user")
    password: SecretStr = Field(default=SecretStr(""), description="Database password")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn(self) -> str:
        """Return an asyncpg-compatible DSN string."""
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn_sync(self) -> str:
        """Return a psycopg2-compatible DSN string."""
        pwd = self.password.get_secret_value()
        return (
            f"postgresql+psycopg2://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"
        )


class QuestDBSettings(BaseSettings):
    """QuestDB connection parameters for time-series storage."""

    model_config = SettingsConfigDict(env_prefix="QUESTDB_")

    host: str = Field(default="questdb", description="QuestDB hostname")
    http_port: int = Field(default=9000, description="QuestDB HTTP/REST port")
    pg_port: int = Field(
        default=8812,
        description="QuestDB PostgreSQL wire protocol port",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn(self) -> str:
        """Return a PostgreSQL wire protocol DSN for QuestDB."""
        return f"postgresql://admin:quest@{self.host}:{self.pg_port}/qdb"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def http_url(self) -> str:
        """Return the HTTP base URL for QuestDB REST queries."""
        return f"http://{self.host}:{self.http_port}"


class RedisSettings(BaseSettings):
    """Redis connection parameters."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(default="redis", description="Redis hostname")
    port: int = Field(default=6379, description="Redis port")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> str:
        """Return a redis:// connection URL."""
        return f"redis://{self.host}:{self.port}/0"


class APIKeySettings(BaseSettings):
    """Third-party API keys.

    All keys default to empty strings so the application can start
    without every service configured.  Individual subsystems must
    check for a non-empty key before attempting API calls.
    """

    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic Claude API key",
    )
    polygon_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Polygon.io API key for market data",
    )
    unusual_whales_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Unusual Whales API key for options flow",
    )
    finnhub_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Finnhub API key for news and calendar data",
    )
    quiver_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Quiver Quantitative API key for alternative data",
    )
    fred_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="FRED API key for macroeconomic data",
    )


class NotificationSettings(BaseSettings):
    """Notification channel configuration."""

    telegram_bot_token: SecretStr = Field(
        default=SecretStr(""),
        description="Telegram Bot API token",
    )
    telegram_chat_id: str = Field(
        default="",
        description="Telegram chat/channel ID for notifications",
    )
    twilio_account_sid: str = Field(
        default="",
        description="Twilio account SID",
    )
    twilio_auth_token: SecretStr = Field(
        default=SecretStr(""),
        description="Twilio auth token",
    )
    twilio_from_number: str = Field(
        default="",
        description="Twilio originating phone number (E.164 format)",
    )
    twilio_to_number: str = Field(
        default="",
        description="Destination phone number for SMS alerts (E.164 format)",
    )

    @property
    def telegram_enabled(self) -> bool:
        """Return True when Telegram is fully configured."""
        return bool(
            self.telegram_bot_token.get_secret_value() and self.telegram_chat_id
        )

    @property
    def twilio_enabled(self) -> bool:
        """Return True when Twilio SMS is fully configured."""
        return bool(
            self.twilio_account_sid
            and self.twilio_auth_token.get_secret_value()
            and self.twilio_from_number
            and self.twilio_to_number
        )


class TradingSettings(BaseSettings):
    """Core trading parameters that govern risk and position management."""

    account_size: float = Field(
        default=150_000.0,
        description="Total account equity in USD",
    )
    max_drawdown_pct: float = Field(
        default=0.15,
        description="Maximum allowable drawdown as a fraction of account equity",
    )
    per_trade_risk_pct: float = Field(
        default=0.02,
        description="Maximum risk per trade as a fraction of account equity",
    )
    max_concurrent_positions: int = Field(
        default=8,
        description="Maximum number of open positions at any time",
    )
    confidence_threshold: float = Field(
        default=0.78,
        description="Minimum ensemble confidence score required to enter a trade",
    )


class ClaudeAISettings(BaseSettings):
    """Claude AI model and thinking budget configuration."""

    claude_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model identifier for API calls",
    )
    claude_analysis_thinking_budget: int = Field(
        default=16384,
        description="Token budget for extended thinking in the Analysis Agent",
    )
    claude_risk_thinking_budget: int = Field(
        default=8192,
        description="Token budget for extended thinking in the Risk Agent",
    )


class Settings(BaseSettings):
    """Root configuration for Project Titan.

    Aggregates all sub-configurations and provides convenience properties
    for database connection strings.  Values are loaded from environment
    variables and a ``.env`` file located at the project root.

    Instantiate via :func:`get_settings` to benefit from caching::

        settings = get_settings()
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Sub-configurations ───────────────────────────────────────────────
    ibkr: IBKRSettings = Field(default_factory=IBKRSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    questdb: QuestDBSettings = Field(default_factory=QuestDBSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api_keys: APIKeySettings = Field(default_factory=APIKeySettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    claude: ClaudeAISettings = Field(default_factory=ClaudeAISettings)

    # ── Grafana ──────────────────────────────────────────────────────────
    grafana_password: str = Field(
        default="admin",
        description="Grafana admin password",
    )

    # ── Paths ────────────────────────────────────────────────────────────
    project_root: Path = Field(
        default=_PROJECT_ROOT,
        description="Absolute path to the project root directory",
    )
    config_dir: Path = Field(
        default=_PROJECT_ROOT / "config",
        description="Absolute path to the configuration directory",
    )
    models_dir: Path = Field(
        default=_PROJECT_ROOT / "models",
        description="Directory for serialized ML model artifacts",
    )
    data_dir: Path = Field(
        default=_PROJECT_ROOT / "data",
        description="Local data cache directory",
    )
    logs_dir: Path = Field(
        default=_PROJECT_ROOT / "logs",
        description="Structured log output directory",
    )

    # ── Convenience properties ───────────────────────────────────────────

    @property
    def postgres_dsn(self) -> str:
        """Async-compatible PostgreSQL DSN."""
        return self.postgres.dsn

    @property
    def questdb_dsn(self) -> str:
        """QuestDB PostgreSQL wire protocol DSN."""
        return self.questdb.dsn

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        return self.redis.url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings.

    The first call builds the settings from environment variables and
    the ``.env`` file.  Subsequent calls return the same instance.
    """
    return Settings()
