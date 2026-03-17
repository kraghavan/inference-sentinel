"""Abstract base classes for inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal


@dataclass
class InferenceResult:
    """Result from an inference call."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float | None = None
    itl_values_ms: list[float] | None = None
    total_latency_ms: float = 0.0
    finish_reason: str = "stop"
    error: str | None = None
    cost_usd: float | None = None  # For cloud backends


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    content: str
    token_index: int | None = None
    is_first: bool = False
    is_last: bool = False
    finish_reason: str | None = None
    timestamp_ms: float = 0.0
    # Final chunk stats
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_ms: float | None = None
    itl_values_ms: list[float] | None = None
    total_latency_ms: float | None = None
    cost_usd: float | None = None
    error: str | None = None


class BaseBackend(ABC):
    """Abstract base class for all inference backends.
    
    This is the root of the backend hierarchy:
    
    BaseBackend
    ├── LocalBackend  - for local inference (Ollama, vLLM, MLX, etc.)
    └── CloudBackend  - for cloud APIs (Anthropic, Google, OpenAI, etc.)
    """

    @property
    @abstractmethod
    def endpoint_name(self) -> str:
        """Return the endpoint name (e.g., 'mac-mini', 'anthropic')."""
        ...

    @property
    @abstractmethod
    def backend_type(self) -> Literal["local", "cloud"]:
        """Return the backend type."""
        ...

    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """Return current health status."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend (connect, validate, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup resources."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy and update status."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> InferenceResult:
        """Generate a completion (non-streaming)."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a completion (streaming)."""
        ...


class LocalBackend(BaseBackend):
    """Abstract base class for local inference backends.
    
    Local backends run inference on local hardware (CPU, GPU, Apple Silicon).
    Examples: Ollama, vLLM, llama.cpp, MLX, TensorRT-LLM
    
    Characteristics:
    - No per-token cost
    - Lower latency for small requests
    - Hardware-dependent throughput
    - Can list available models
    """

    @property
    def backend_type(self) -> Literal["local", "cloud"]:
        return "local"

    @property
    def cost_per_token(self) -> float:
        """Local inference has no per-token cost."""
        return 0.0

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models on this local backend."""
        ...

    @abstractmethod
    async def pull_model(self, model: str) -> bool:
        """Pull/download a model (if supported)."""
        ...

    @abstractmethod
    async def get_model_info(self, model: str) -> dict:
        """Get information about a specific model."""
        ...


class CloudBackend(BaseBackend):
    """Abstract base class for cloud inference backends.
    
    Cloud backends call external APIs for inference.
    Examples: Anthropic (Claude), Google (Gemini), OpenAI, Cohere, Mistral
    
    Characteristics:
    - Per-token pricing
    - Higher latency (network round-trip)
    - Unlimited throughput (within rate limits)
    - Fixed model list per provider
    """

    @property
    def backend_type(self) -> Literal["local", "cloud"]:
        return "cloud"

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the cloud provider name (e.g., 'anthropic', 'google')."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this backend."""
        ...

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """Return list of supported model identifiers."""
        ...

    @abstractmethod
    def get_pricing(self, model: str) -> dict[str, float]:
        """Get pricing per 1M tokens for a model.
        
        Returns:
            Dict with 'input' and 'output' prices per 1M tokens.
        """
        ...

    def calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate cost for a request.
        
        Args:
            model: Model identifier.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            
        Returns:
            Cost in USD.
        """
        pricing = self.get_pricing(model)
        return (
            (prompt_tokens / 1_000_000) * pricing["input"]
            + (completion_tokens / 1_000_000) * pricing["output"]
        )
