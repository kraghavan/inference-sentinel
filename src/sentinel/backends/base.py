"""Abstract base class for inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class InferenceResult:
    """Result from an inference call."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float | None = None
    itl_values_ms: list[float] | None = None  # Raw inter-token latencies
    total_latency_ms: float = 0.0
    finish_reason: str = "stop"
    error: str | None = None


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    content: str
    is_first: bool = False
    is_last: bool = False
    timestamp_ms: float = 0.0


class BaseBackend(ABC):
    """Abstract base class for all inference backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> InferenceResult:
        """Generate a completion (non-streaming)."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a completion (streaming)."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models on this backend."""
        ...
