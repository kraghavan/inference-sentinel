"""Integration tests for API endpoints.

Note: These tests require the full stack to be running.
Run with: pytest tests/integration/ -v (requires docker-compose up)
Skip with: pytest tests/unit/ -v (unit tests only)
"""

import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    async def test_health_returns_status(self, async_client: AsyncClient) -> None:
        """Test health endpoint returns valid response structure."""
        response = await async_client.get("/health")
        # Note: Will be unhealthy if Ollama isn't running, but structure should be valid
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "backends" in data
        assert data["version"] == "0.1.0"


class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""

    async def test_models_returns_dict(self, async_client: AsyncClient) -> None:
        """Test models endpoint returns valid structure."""
        response = await async_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], dict)


@pytest.mark.ollama
class TestInferenceEndpoint:
    """Tests for /v1/inference endpoint.
    
    These tests require Ollama to be running.
    Run with: pytest -m ollama
    """

    async def test_inference_basic(self, async_client: AsyncClient) -> None:
        """Test basic inference request."""
        response = await async_client.post(
            "/v1/inference",
            json={
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "id" in data
        assert data["id"].startswith("req_")
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "usage" in data
        assert "sentinel" in data
        
        # Check sentinel metadata
        sentinel = data["sentinel"]
        assert sentinel["route"] == "local"
        assert sentinel["backend"] == "ollama"
        assert "ttft_ms" in sentinel
        assert "inference_latency_ms" in sentinel

    async def test_inference_with_system_message(self, async_client: AsyncClient) -> None:
        """Test inference with system message."""
        response = await async_client.post(
            "/v1/inference",
            json={
                "messages": [
                    {"role": "system", "content": "You are a pirate."},
                    {"role": "user", "content": "Say ahoy."},
                ],
                "max_tokens": 20,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    async def test_inference_custom_temperature(self, async_client: AsyncClient) -> None:
        """Test inference with custom temperature."""
        response = await async_client.post(
            "/v1/inference",
            json={
                "messages": [{"role": "user", "content": "Count to 3."}],
                "max_tokens": 20,
                "temperature": 0.1,
            },
        )
        assert response.status_code == 200

    async def test_inference_invalid_request(self, async_client: AsyncClient) -> None:
        """Test inference with invalid request."""
        response = await async_client.post(
            "/v1/inference",
            json={
                "messages": [],  # Empty messages
            },
        )
        # Should still succeed or return validation error
        # Depends on whether we validate empty messages
        assert response.status_code in [200, 422]

    async def test_inference_metrics_populated(self, async_client: AsyncClient) -> None:
        """Test that inference metrics are populated."""
        response = await async_client.post(
            "/v1/inference",
            json={
                "messages": [{"role": "user", "content": "Say yes or no."}],
                "max_tokens": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        
        sentinel = data["sentinel"]
        # These should always be present
        assert sentinel["classification_latency_ms"] >= 0
        assert sentinel["routing_latency_ms"] >= 0
        assert sentinel["inference_latency_ms"] > 0
        
        # TTFT should be present for successful requests
        assert sentinel["ttft_ms"] is not None
        assert sentinel["ttft_ms"] > 0
