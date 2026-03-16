"""API request and response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation."""

    role: Literal["user", "assistant", "system"] = Field(description="Message role")
    content: str = Field(description="Message content")


class InferenceRequest(BaseModel):
    """Request body for inference endpoint."""

    model: str = Field(
        default="auto",
        description="Model to use: 'auto', 'local', 'cloud', or specific model name",
    )
    messages: list[Message] = Field(description="Conversation messages")
    max_tokens: int = Field(default=1024, ge=1, le=16384)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False, description="Whether to stream the response")
    routing_override: Literal["local", "cloud"] | None = Field(
        default=None,
        description="Force routing to local or cloud (bypasses classification)",
    )


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class SentinelMetadata(BaseModel):
    """Sentinel-specific metadata about the request."""

    route: Literal["local", "cloud"]
    backend: str = Field(description="Backend used (e.g., 'ollama', 'anthropic')")
    endpoint: str | None = Field(default=None, description="Specific endpoint name")
    model: str = Field(description="Model used for inference")
    privacy_tier: int = Field(default=0, description="Privacy classification tier (0-3)")
    privacy_tier_label: str = Field(default="PUBLIC")
    entities_detected: list[str] = Field(default_factory=list)
    classification_latency_ms: float = Field(default=0.0)
    routing_latency_ms: float = Field(default=0.0)
    inference_latency_ms: float = Field(default=0.0)
    ttft_ms: float | None = Field(default=None, description="Time to first token")
    itl_p50_ms: float | None = Field(default=None, description="Inter-token latency p50")
    itl_p95_ms: float | None = Field(default=None, description="Inter-token latency p95")
    tpot_ms: float | None = Field(default=None, description="Time per output token")
    tokens_per_second: float | None = Field(default=None)
    cost_usd: float = Field(default=0.0)
    cost_savings_usd: float = Field(default=0.0)


class Choice(BaseModel):
    """A single completion choice."""

    message: Message
    finish_reason: Literal["stop", "length", "error"] = Field(default="stop")


class InferenceResponse(BaseModel):
    """Response body for inference endpoint."""

    id: str = Field(description="Unique request ID")
    model: str = Field(description="Model that generated the response")
    choices: list[Choice]
    usage: Usage
    sentinel: SentinelMetadata


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    backends: dict[str, bool] = Field(
        default_factory=dict,
        description="Health status of each backend",
    )


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str
    detail: str | None = None
    request_id: str | None = None
