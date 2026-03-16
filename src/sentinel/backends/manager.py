"""Backend manager for handling multiple inference endpoints."""

import asyncio
from typing import Literal

import structlog

from sentinel.backends.base import BaseBackend, InferenceResult
from sentinel.backends.ollama import OllamaBackend
from sentinel.config import LocalBackendsConfig, LocalEndpoint

logger = structlog.get_logger()


class BackendManager:
    """Manages multiple inference backends with health checking and selection."""

    def __init__(self, config: LocalBackendsConfig):
        self._config = config
        self._backends: dict[str, OllamaBackend] = {}
        self._health_status: dict[str, bool] = {}
        self._round_robin_index: int = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize all configured backends."""
        for endpoint in self._config.endpoints:
            if endpoint.enabled:
                backend = OllamaBackend(endpoint, timeout=self._config.timeout_seconds)
                self._backends[endpoint.name] = backend
                self._health_status[endpoint.name] = False

        # Initial health check
        await self.refresh_health()

        logger.info(
            "Backend manager initialized",
            endpoints=list(self._backends.keys()),
            healthy=[k for k, v in self._health_status.items() if v],
        )

    async def close(self) -> None:
        """Close all backend connections."""
        for backend in self._backends.values():
            await backend.close()

    async def refresh_health(self) -> dict[str, bool]:
        """Refresh health status for all backends."""
        tasks = {
            name: backend.health_check()
            for name, backend in self._backends.items()
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                self._health_status[name] = False
                logger.warning("Health check exception", endpoint=name, error=str(result))
            else:
                self._health_status[name] = result

        return self._health_status.copy()

    def get_healthy_backends(self) -> list[OllamaBackend]:
        """Get list of healthy backends."""
        return [
            self._backends[name]
            for name, healthy in self._health_status.items()
            if healthy
        ]

    async def select_backend(
        self,
        strategy: Literal["priority", "round_robin", "latency_best"] | None = None,
    ) -> OllamaBackend | None:
        """Select a backend based on the configured strategy."""
        strategy = strategy or self._config.selection_strategy
        healthy = self.get_healthy_backends()

        if not healthy:
            logger.warning("No healthy backends available")
            return None

        if strategy == "priority":
            # Sort by priority (lower = higher priority)
            sorted_backends = sorted(
                healthy,
                key=lambda b: next(
                    e.priority for e in self._config.endpoints if e.name == b.endpoint_name
                ),
            )
            return sorted_backends[0]

        elif strategy == "round_robin":
            async with self._lock:
                backend = healthy[self._round_robin_index % len(healthy)]
                self._round_robin_index += 1
                return backend

        elif strategy == "latency_best":
            # For now, just use priority. Real implementation would track latencies.
            # TODO: Implement latency tracking
            return healthy[0]

        return healthy[0]

    async def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        endpoint_name: str | None = None,
    ) -> tuple[InferenceResult, OllamaBackend | None]:
        """Generate a completion using the best available backend.

        Returns:
            Tuple of (result, backend_used). backend_used is None if all backends failed.
        """
        # If specific endpoint requested
        if endpoint_name and endpoint_name in self._backends:
            backend = self._backends[endpoint_name]
            result = await backend.generate(messages, model, max_tokens, temperature)
            return result, backend

        # Select best available backend
        backend = await self.select_backend()
        if backend is None:
            return InferenceResult(
                content="",
                model=model or "unknown",
                prompt_tokens=0,
                completion_tokens=0,
                finish_reason="error",
                error="No healthy backends available",
            ), None

        result = await backend.generate(messages, model, max_tokens, temperature)

        # If failed and failover enabled, try others
        if result.error and self._config.failover_enabled:
            healthy = self.get_healthy_backends()
            for fallback in healthy:
                if fallback.endpoint_name != backend.endpoint_name:
                    logger.info(
                        "Failing over to alternative backend",
                        from_endpoint=backend.endpoint_name,
                        to_endpoint=fallback.endpoint_name,
                    )
                    result = await fallback.generate(messages, model, max_tokens, temperature)
                    if not result.error:
                        return result, fallback

        return result, backend

    def get_backend(self, endpoint_name: str) -> OllamaBackend | None:
        """Get a specific backend by name."""
        return self._backends.get(endpoint_name)

    @property
    def health_status(self) -> dict[str, bool]:
        """Get current health status of all backends."""
        return self._health_status.copy()

    async def list_all_models(self) -> dict[str, list[str]]:
        """List models available on all endpoints."""
        result = {}
        for name, backend in self._backends.items():
            result[name] = await backend.list_models()
        return result
