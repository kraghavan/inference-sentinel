"""Unit tests for cloud backend selection strategies."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import AsyncIterator, Literal

from sentinel.backends.manager import BackendManager
from sentinel.backends.base import BaseBackend, InferenceResult, StreamChunk
from sentinel.config import LocalBackendsConfig


class MockBackend(BaseBackend):
    """Mock backend for testing."""
    
    def __init__(self, name: str):
        self._name = name
        self._healthy = True
    
    @property
    def endpoint_name(self) -> str:
        return self._name
    
    @property
    def backend_type(self) -> Literal["local", "cloud"]:
        return "cloud"
    
    @property
    def is_healthy(self) -> bool:
        return self._healthy
    
    async def initialize(self) -> None:
        pass
    
    async def close(self) -> None:
        pass
    
    async def health_check(self) -> bool:
        return self._healthy
    
    async def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> InferenceResult:
        return InferenceResult(
            content="mock response",
            model=model or "mock-model",
            prompt_tokens=10,
            completion_tokens=20,
        )
    
    async def generate_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content="mock", is_first=True, is_last=True)


class TestCloudSelectionStrategies:
    """Test cloud backend selection strategies."""
    
    @pytest.fixture
    def config(self):
        """Create minimal local config."""
        return LocalBackendsConfig(endpoints=[])
    
    @pytest.fixture
    def anthropic_backend(self):
        """Create mock Anthropic backend."""
        return MockBackend("anthropic")
    
    @pytest.fixture
    def google_backend(self):
        """Create mock Google backend."""
        return MockBackend("google")
    
    def test_round_robin_alternates(self, config, anthropic_backend, google_backend):
        """Round-robin should alternate between backends."""
        manager = BackendManager(
            config=config,
            cloud_selection_strategy="round_robin",
        )
        
        # Add both backends
        manager.add_cloud_backend("anthropic", anthropic_backend)
        manager.add_cloud_backend("google", google_backend)
        
        # Mark both as healthy
        manager._health_status["anthropic"] = True
        manager._health_status["google"] = True
        
        # Select multiple times - should alternate
        selections = []
        for _ in range(6):
            backend = manager.select_cloud_backend()
            selections.append(backend.endpoint_name)
        
        # Should see both backends used
        assert "anthropic" in selections
        assert "google" in selections
        
        # Should alternate (not all same)
        assert len(set(selections)) == 2
    
    def test_round_robin_skips_unhealthy(self, config, anthropic_backend, google_backend):
        """Round-robin should skip unhealthy backends."""
        manager = BackendManager(
            config=config,
            cloud_selection_strategy="round_robin",
        )
        
        manager.add_cloud_backend("anthropic", anthropic_backend)
        manager.add_cloud_backend("google", google_backend)
        
        # Only anthropic is healthy
        manager._health_status["anthropic"] = True
        manager._health_status["google"] = False
        
        # All selections should be anthropic
        for _ in range(5):
            backend = manager.select_cloud_backend()
            assert backend.endpoint_name == "anthropic"
    
    def test_primary_fallback_uses_primary_first(self, config, anthropic_backend, google_backend):
        """Primary/fallback should use primary when healthy."""
        manager = BackendManager(
            config=config,
            cloud_selection_strategy="primary_fallback",
            cloud_primary="anthropic",
            cloud_fallback="google",
        )
        
        manager.add_cloud_backend("anthropic", anthropic_backend)
        manager.add_cloud_backend("google", google_backend)
        
        # Both healthy
        manager._health_status["anthropic"] = True
        manager._health_status["google"] = True
        
        # Should always use primary
        for _ in range(5):
            backend = manager.select_cloud_backend()
            assert backend.endpoint_name == "anthropic"
    
    def test_primary_fallback_uses_fallback_when_primary_unhealthy(
        self, config, anthropic_backend, google_backend
    ):
        """Primary/fallback should use fallback when primary is down."""
        manager = BackendManager(
            config=config,
            cloud_selection_strategy="primary_fallback",
            cloud_primary="anthropic",
            cloud_fallback="google",
        )
        
        manager.add_cloud_backend("anthropic", anthropic_backend)
        manager.add_cloud_backend("google", google_backend)
        
        # Primary unhealthy
        manager._health_status["anthropic"] = False
        manager._health_status["google"] = True
        
        # Should use fallback
        backend = manager.select_cloud_backend()
        assert backend.endpoint_name == "google"
    
    def test_preferred_override_bypasses_strategy(
        self, config, anthropic_backend, google_backend
    ):
        """Preferred parameter should bypass selection strategy."""
        manager = BackendManager(
            config=config,
            cloud_selection_strategy="round_robin",
        )
        
        manager.add_cloud_backend("anthropic", anthropic_backend)
        manager.add_cloud_backend("google", google_backend)
        
        manager._health_status["anthropic"] = True
        manager._health_status["google"] = True
        
        # Force google even in round-robin
        backend = manager.select_cloud_backend(preferred="google")
        assert backend.endpoint_name == "google"
        
        # Force anthropic
        backend = manager.select_cloud_backend(preferred="anthropic")
        assert backend.endpoint_name == "anthropic"
    
    def test_no_healthy_backends_returns_none(self, config, anthropic_backend, google_backend):
        """Should return None when no backends are healthy."""
        manager = BackendManager(
            config=config,
            cloud_selection_strategy="round_robin",
        )
        
        manager.add_cloud_backend("anthropic", anthropic_backend)
        manager.add_cloud_backend("google", google_backend)
        
        # Both unhealthy
        manager._health_status["anthropic"] = False
        manager._health_status["google"] = False
        
        backend = manager.select_cloud_backend()
        assert backend is None
    
    def test_single_backend_round_robin(self, config, anthropic_backend):
        """Round-robin with single backend should work."""
        manager = BackendManager(
            config=config,
            cloud_selection_strategy="round_robin",
        )
        
        manager.add_cloud_backend("anthropic", anthropic_backend)
        manager._health_status["anthropic"] = True
        
        # Should return the same backend every time
        for _ in range(5):
            backend = manager.select_cloud_backend()
            assert backend.endpoint_name == "anthropic"
