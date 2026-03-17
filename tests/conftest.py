"""Pytest configuration and fixtures."""

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from sentinel.backends import BackendManager
from sentinel.config import LocalBackendsConfig, LocalEndpoint


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def local_endpoint() -> LocalEndpoint:
    """Create a test local endpoint configuration."""
    return LocalEndpoint(
        name="test-endpoint",
        host="localhost",
        port=11434,
        model="gemma3:4b",
        priority=1,
        enabled=True,
    )


@pytest.fixture
def local_backends_config(local_endpoint: LocalEndpoint) -> LocalBackendsConfig:
    """Create a test local backends configuration."""
    return LocalBackendsConfig(
        endpoints=[local_endpoint],
        selection_strategy="priority",
        health_check_interval_seconds=30,
        failover_enabled=True,
        timeout_seconds=60.0,
    )


@pytest_asyncio.fixture
async def backend_manager(
    local_backends_config: LocalBackendsConfig,
) -> AsyncGenerator[BackendManager, None]:
    """Create a test backend manager."""
    manager = BackendManager(local_backends_config)
    # Don't initialize (would require running Ollama)
    yield manager
    await manager.close()


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app.
    
    Note: This fixture is for integration tests only.
    It requires all dependencies (including prometheus_client, etc.)
    """
    # Lazy import to avoid loading full app for unit tests
    from sentinel.main import create_app
    
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
