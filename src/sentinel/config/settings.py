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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    primary: Literal["anthropic", "google"] = Field(default="anthropic")
    fallback: Literal["anthropic", "google", "none"] = Field(default="google")
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
    )
    google_api_key: str | None = Field(
        default=None,
        validation_alias="GOOGLE_API_KEY",
    )
    anthropic_model: str = Field(default="claude-sonnet-4-20250514")
    google_model: str = Field(default="gemini-2.0-flash")
    timeout_seconds: float = Field(default=60.0)


class TelemetryConfig(BaseSettings):
    """Configuration for observability."""

    enabled: bool = Field(default=True)
    otlp_endpoint: str = Field(default="http://localhost:4317")
    service_name: str = Field(default="inference-sentinel")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    metrics_port: int = Field(default=9090)


class NERConfig(BaseSettings):
    """Configuration for NER-based classification."""
    
    enabled: bool = Field(default=False, description="Enable NER classification")
    model: Literal["fast", "accurate", "multilingual"] = Field(
        default="fast",
        description="NER model: fast (BERT), accurate (RoBERTa), multilingual"
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device for NER inference"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for NER entities"
    )
    skip_if_regex_tier_gte: int = Field(
        default=3,
        description="Skip NER if regex tier >= this value"
    )


class ShadowConfig(BaseSettings):
    """Configuration for shadow mode (A/B comparison)."""
    
    enabled: bool = Field(default=False, description="Enable shadow mode")
    shadow_tiers: list[int] = Field(
        default=[0, 1],
        description="Privacy tiers to shadow"
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of requests to shadow"
    )
    similarity_enabled: bool = Field(
        default=True,
        description="Enable similarity scoring"
    )
    similarity_model: Literal["fast", "balanced", "accurate"] = Field(
        default="fast",
        description="Embedding model for similarity"
    )
    similarity_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device for similarity computation"
    )
    local_timeout_seconds: float = Field(
        default=60.0,
        description="Timeout for shadow local inference"
    )
    store_responses: bool = Field(
        default=False,
        description="Store full responses (memory heavy)"
    )
    max_stored_results: int = Field(
        default=1000,
        description="Maximum shadow results to store"
    )


class CloudSelectionConfig(BaseSettings):
    """Configuration for cloud backend selection strategy."""
    
    strategy: Literal["primary_fallback", "round_robin"] = Field(
        default="round_robin",
        description="How to select between cloud backends"
    )
    primary: Literal["anthropic", "google"] = Field(
        default="anthropic",
        description="Primary cloud backend (for primary_fallback strategy)"
    )
    fallback: Literal["anthropic", "google", "none"] = Field(
        default="google",
        description="Fallback cloud backend (for primary_fallback strategy)"
    )
    max_retries: int = Field(default=2, description="Max retries before fallback")
    retry_delay_ms: int = Field(default=500, description="Delay between retries")
    fallback_to_local_on_timeout: bool = Field(
        default=True,
        description="Fall back to local if all cloud fails"
    )


class ControllerSettings(BaseSettings):
    """Configuration for closed-loop controller."""
    
    enabled: bool = Field(default=False, description="Enable the controller")
    mode: Literal["observe", "auto"] = Field(
        default="observe",
        description="Controller mode: observe (log only) or auto (apply changes)"
    )
    evaluation_interval_seconds: int = Field(
        default=60,
        description="How often to evaluate metrics"
    )
    window_seconds: int = Field(
        default=300,
        description="Rolling window for metric aggregation (5 min default)"
    )
    drift_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Alert if similarity drops by this percentage"
    )
    cost_savings_threshold_usd: float = Field(
        default=50.0,
        description="Minimum cost savings to recommend routing change"
    )


class SessionConfig(BaseSettings):
    """Configuration for session management (one-way trapdoor)."""
    
    enabled: bool = Field(
        default=False,
        description="Enable session-based routing stickiness"
    )
    ttl_seconds: int = Field(
        default=900,  # 15 minutes
        ge=60,
        le=86400,
        description="Session TTL in seconds (15 min default)"
    )
    max_sessions: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum concurrent sessions"
    )
    lock_threshold_tier: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Minimum tier to trigger LOCAL_LOCKED (2 = CONFIDENTIAL)"
    )
    buffer_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Rolling buffer size (number of interactions to remember)"
    )
    scrub_buffer: bool = Field(
        default=True,
        description="Run classifier on buffer entries before storing"
    )
    inject_context_on_handoff: bool = Field(
        default=True,
        description="Inject conversation context when switching to local"
    )
    capability_guardrail: bool = Field(
        default=True,
        description="Inject 'no external tools' warning to local models"
    )


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
    cloud_selection: CloudSelectionConfig = Field(default_factory=CloudSelectionConfig)

    # Classification
    ner: NERConfig = Field(default_factory=NERConfig)
    
    # Shadow mode
    shadow: ShadowConfig = Field(default_factory=ShadowConfig)
    
    # Closed-loop controller
    controller: ControllerSettings = Field(default_factory=ControllerSettings)
    
    # Session management (one-way trapdoor)
    session: SessionConfig = Field(default_factory=SessionConfig)

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
