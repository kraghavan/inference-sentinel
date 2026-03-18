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

    def __init__(
        self,
        config: LocalBackendsConfig,
        cloud_selection_strategy: Literal["primary_fallback", "round_robin"] = "primary_fallback",
        cloud_primary: str = "anthropic",
        cloud_fallback: str = "google",
    ):
        self._config = config
        self._local_backends: dict[str, OllamaBackend] = {}
        self._cloud_backends: dict[str, BaseBackend] = {}
        self._health_status: dict[str, bool] = {}
        
        # Cloud selection config
        self._cloud_selection_strategy = cloud_selection_strategy
        self._cloud_primary = cloud_primary
        self._cloud_fallback = cloud_fallback
        
        # Round-robin state
        self._cloud_rr_index: int = 0
        self._local_rr_index: int = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize all configured backends."""
        for endpoint in self._config.endpoints:
            if endpoint.enabled:
                backend = OllamaBackend(endpoint, timeout=self._config.timeout_seconds)
                self._local_backends[endpoint.name] = backend
                self._health_status[endpoint.name] = False

        # Initial health check
        await self.refresh_health()

        logger.info(
            "Backend manager initialized",
            local_endpoints=list(self._local_backends.keys()),
            cloud_endpoints=list(self._cloud_backends.keys()),
            healthy=[k for k, v in self._health_status.items() if v],
        )

    def add_cloud_backend(self, name: str, backend: BaseBackend) -> None:
        """Add a cloud backend to the manager.

        Args:
            name: Name for the backend (e.g., "anthropic", "google").
            backend: The backend instance.
        """
        self._cloud_backends[name] = backend
        self._health_status[name] = False
        logger.info("Added cloud backend", name=name)

    async def initialize_cloud_backends(self) -> None:
        """Initialize all cloud backends."""
        for name, backend in self._cloud_backends.items():
            try:
                await backend.initialize()
                self._health_status[name] = True
                logger.info("Cloud backend initialized", name=name)
            except Exception as e:
                logger.error("Failed to initialize cloud backend", name=name, error=str(e))
                self._health_status[name] = False

    async def close(self) -> None:
        """Close all backend connections."""
        for backend in self._local_backends.values():
            await backend.close()
        for backend in self._cloud_backends.values():
            await backend.close()

    async def refresh_health(self) -> dict[str, bool]:
        """Refresh health status for all backends."""
        # Check local backends
        local_tasks = {
            name: backend.health_check()
            for name, backend in self._local_backends.items()
        }

        cloud_tasks = {
            name: backend.health_check()
            for name, backend in self._cloud_backends.items()
        }

        all_tasks = {**local_tasks, **cloud_tasks}
        results = await asyncio.gather(*all_tasks.values(), return_exceptions=True)

        for name, result in zip(all_tasks.keys(), results):
            if isinstance(result, Exception):
                self._health_status[name] = False
                logger.warning("Health check exception", endpoint=name, error=str(result))
            else:
                self._health_status[name] = result

        return self._health_status.copy()

    def get_healthy_local_backends(self) -> list[OllamaBackend]:
        """Get list of healthy local backends."""
        return [
            self._local_backends[name]
            for name, healthy in self._health_status.items()
            if healthy and name in self._local_backends
        ]

    def get_healthy_cloud_backends(self) -> list[BaseBackend]:
        """Get list of healthy cloud backends."""
        return [
            self._cloud_backends[name]
            for name, healthy in self._health_status.items()
            if healthy and name in self._cloud_backends
        ]

    async def select_local_backend(
        self,
        strategy: Literal["priority", "round_robin", "latency_best"] | None = None,
    ) -> OllamaBackend | None:
        """Select a local backend based on the configured strategy."""
        strategy = strategy or self._config.selection_strategy
        healthy = self.get_healthy_local_backends()

        if not healthy:
            logger.warning("No healthy local backends available")
            return None

        if strategy == "priority":
            sorted_backends = sorted(
                healthy,
                key=lambda b: next(
                    e.priority for e in self._config.endpoints if e.name == b.endpoint_name
                ),
            )
            return sorted_backends[0]

        elif strategy == "round_robin":
            async with self._lock:
                backend = healthy[self._local_rr_index % len(healthy)]
                self._local_rr_index += 1
                return backend

        elif strategy == "latency_best":
            return healthy[0]

        return healthy[0]

    def select_cloud_backend(self, preferred: str | None = None) -> BaseBackend | None:
        """Select a cloud backend based on configured strategy.

        Strategies:
        - primary_fallback: Try primary, then fallback
        - round_robin: Alternate between healthy backends

        Args:
            preferred: Override to use specific backend (bypasses strategy).

        Returns:
            A healthy cloud backend or None.
        """
        healthy = self.get_healthy_cloud_backends()

        if not healthy:
            logger.warning("No healthy cloud backends available")
            return None

        # If preferred backend specified and healthy, use it
        if preferred and preferred in self._cloud_backends:
            backend = self._cloud_backends[preferred]
            if self._health_status.get(preferred, False):
                logger.debug("Using preferred cloud backend", backend=preferred)
                return backend

        # Apply selection strategy
        if self._cloud_selection_strategy == "round_robin":
            return self._select_cloud_round_robin(healthy)
        else:
            # primary_fallback (default)
            return self._select_cloud_primary_fallback(healthy)

    def _select_cloud_primary_fallback(
        self, healthy: list[BaseBackend]
    ) -> BaseBackend | None:
        """Select cloud backend using primary/fallback strategy."""
        # Check if primary is healthy
        if self._cloud_primary in self._cloud_backends:
            primary = self._cloud_backends[self._cloud_primary]
            if primary in healthy:
                logger.debug("Using primary cloud backend", backend=self._cloud_primary)
                return primary

        # Check if fallback is healthy
        if self._cloud_fallback in self._cloud_backends:
            fallback = self._cloud_backends[self._cloud_fallback]
            if fallback in healthy:
                logger.debug(
                    "Primary unavailable, using fallback",
                    primary=self._cloud_primary,
                    fallback=self._cloud_fallback,
                )
                return fallback

        # Return first healthy as last resort
        return healthy[0] if healthy else None

    def _select_cloud_round_robin(
        self, healthy: list[BaseBackend]
    ) -> BaseBackend | None:
        """Select cloud backend using round-robin strategy."""
        if not healthy:
            return None

        # Get next backend in rotation
        selected = healthy[self._cloud_rr_index % len(healthy)]
        
        # Advance index for next call
        self._cloud_rr_index = (self._cloud_rr_index + 1) % len(healthy)
        
        logger.debug(
            "Round-robin cloud selection",
            selected=selected.endpoint_name,
            index=self._cloud_rr_index,
            healthy_count=len(healthy),
        )
        
        return selected

    async def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        endpoint_name: str | None = None,
    ) -> tuple[InferenceResult, BaseBackend | None]:
        """Generate a completion using the best available local backend.

        Returns:
            Tuple of (result, backend_used). backend_used is None if all backends failed.
        """
        # If specific endpoint requested
        if endpoint_name and endpoint_name in self._local_backends:
            backend = self._local_backends[endpoint_name]
            result = await backend.generate(messages, model, max_tokens, temperature)
            return result, backend

        # Select best available local backend
        backend = await self.select_local_backend()
        if backend is None:
            return InferenceResult(
                content="",
                model=model or "unknown",
                prompt_tokens=0,
                completion_tokens=0,
                finish_reason="error",
                error="No healthy local backends available",
            ), None

        result = await backend.generate(messages, model, max_tokens, temperature)

        # If failed and failover enabled, try others
        if result.error and self._config.failover_enabled:
            healthy = self.get_healthy_local_backends()
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

    async def generate_cloud(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        preferred_backend: str | None = None,
    ) -> tuple[InferenceResult, BaseBackend | None]:
        """Generate a completion using a cloud backend.

        Args:
            messages: Conversation messages.
            model: Model name override.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            preferred_backend: Preferred cloud provider (e.g., "anthropic", "google").

        Returns:
            Tuple of (result, backend_used).
        """
        backend = self.select_cloud_backend(preferred_backend)
        if backend is None:
            return InferenceResult(
                content="",
                model=model or "unknown",
                prompt_tokens=0,
                completion_tokens=0,
                finish_reason="error",
                error="No healthy cloud backends available",
            ), None

        result = await backend.generate(messages, model, max_tokens, temperature)

        # Failover to other cloud backends if primary fails
        if result.error:
            healthy = self.get_healthy_cloud_backends()
            for fallback in healthy:
                if fallback.endpoint_name != backend.endpoint_name:
                    logger.info(
                        "Failing over to alternative cloud backend",
                        from_endpoint=backend.endpoint_name,
                        to_endpoint=fallback.endpoint_name,
                    )
                    result = await fallback.generate(messages, model, max_tokens, temperature)
                    if not result.error:
                        return result, fallback

        return result, backend

    async def generate_routed(
        self,
        messages: list[dict],
        route: Literal["local", "cloud"],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        preferred_cloud_backend: str | None = None,
    ) -> tuple[InferenceResult, BaseBackend | None, Literal["local", "cloud"]]:
        """Generate a completion using the specified route.

        Args:
            messages: Conversation messages.
            route: "local" or "cloud".
            model: Model name override.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            preferred_cloud_backend: Preferred cloud provider for cloud route.

        Returns:
            Tuple of (result, backend_used, actual_route).
        """
        if route == "cloud":
            result, backend = await self.generate_cloud(
                messages, model, max_tokens, temperature, preferred_cloud_backend
            )
            if result.error and self._config.failover_enabled:
                # Fallback to local if cloud fails
                logger.info("Cloud failed, falling back to local")
                result, backend = await self.generate(
                    messages, model, max_tokens, temperature
                )
                return result, backend, "local"
            return result, backend, "cloud"
        else:
            result, backend = await self.generate(
                messages, model, max_tokens, temperature
            )
            return result, backend, "local"

    def get_backend(self, endpoint_name: str) -> BaseBackend | None:
        """Get a specific backend by name."""
        if endpoint_name in self._local_backends:
            return self._local_backends[endpoint_name]
        return self._cloud_backends.get(endpoint_name)

    @property
    def health_status(self) -> dict[str, bool]:
        """Get current health status of all backends."""
        return self._health_status.copy()

    @property
    def has_cloud_backends(self) -> bool:
        """Check if any cloud backends are configured."""
        return len(self._cloud_backends) > 0

    @property
    def has_healthy_cloud_backends(self) -> bool:
        """Check if any cloud backends are healthy."""
        return len(self.get_healthy_cloud_backends()) > 0

    async def list_all_models(self) -> dict[str, list[str]]:
        """List models available on all endpoints."""
        result = {}
        for name, backend in self._local_backends.items():
            result[name] = await backend.list_models()
        # Cloud backends have fixed model lists
        for name in self._cloud_backends:
            result[name] = ["(cloud models)"]
        return result
