"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LocalEndpoint(BaseSettings):
    """Configuration for a local Ollama endpoint."""

    name: str = Field(description="Human-readable endpoint name")
    host: str = Field(default="localhost", description="Hostname or IP")
    port: int = Field(default=11434, description="Ollama port")
    model: str = Field(description="Model to use on this endpoint")
    priority: int = Field(default=1, description="Lower = higher priority")
    enabled: bool = Field(default=True, description="Whether endpoint is active")

    @property
    def base_url(self) -> str:
        """Construct the base URL for this endpoint."""
        return f"http://{self.host}:{self.port}"


class LocalBackendsConfig(BaseSettings):
    """Configuration for local inference backends."""

    endpoints: list[LocalEndpoint] = Field(default_factory=list)
    selection_strategy: Literal["priority", "round_robin", "latency_best"] = Field(
        default="priority"
    )
    health_check_interval_seconds: int = Field(default=30)
    failover_enabled: bool = Field(default=True)
    timeout_seconds: float = Field(default=120.0)


class CloudBackendsConfig(BaseSettings):
    """Configuration for cloud inference backends."""

    primary: Literal["anthropic", "google"] = Field(default="anthropic")
    fallback: Literal["anthropic", "google", "none"] = Field(default="google")
    anthropic_api_key: str | None = Field(default=None)
    google_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-sonnet-4-20250514")
    google_model: str = Field(default="gemini-1.5-flash")
    timeout_seconds: float = Field(default=60.0)


class TelemetryConfig(BaseSettings):
    """Configuration for observability."""

    enabled: bool = Field(default=True)
    otlp_endpoint: str = Field(default="http://localhost:4317")
    service_name: str = Field(default="inference-sentinel")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    metrics_port: int = Field(default=9090)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    env: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Backends
    local: LocalBackendsConfig = Field(default_factory=LocalBackendsConfig)
    cloud: CloudBackendsConfig = Field(default_factory=CloudBackendsConfig)

    # Telemetry
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @field_validator("debug", mode="before")
    @classmethod
    def set_debug_from_env(cls, v: bool | str) -> bool:
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
