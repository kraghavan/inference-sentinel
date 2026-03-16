"""Unit tests for API schemas."""

import pytest
from pydantic import ValidationError

from sentinel.api.schemas import (
    Choice,
    ErrorResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    Message,
    SentinelMetadata,
    Usage,
)


class TestMessage:
    """Tests for Message schema."""

    def test_valid_message(self) -> None:
        """Test valid message creation."""
        msg = Message(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_all_roles(self) -> None:
        """Test all valid roles."""
        for role in ["user", "assistant", "system"]:
            msg = Message(role=role, content="test")  # type: ignore
            assert msg.role == role

    def test_invalid_role(self) -> None:
        """Test invalid role raises error."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="test")  # type: ignore


class TestInferenceRequest:
    """Tests for InferenceRequest schema."""

    def test_minimal_request(self) -> None:
        """Test request with only required fields."""
        request = InferenceRequest(
            messages=[Message(role="user", content="Hello")]
        )
        assert request.model == "auto"
        assert request.max_tokens == 1024
        assert request.temperature == 0.7
        assert request.stream is False
        assert request.routing_override is None

    def test_full_request(self) -> None:
        """Test request with all fields."""
        request = InferenceRequest(
            model="gemma3:4b",
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello"),
            ],
            max_tokens=2048,
            temperature=0.5,
            stream=True,
            routing_override="local",
        )
        assert request.model == "gemma3:4b"
        assert len(request.messages) == 2
        assert request.max_tokens == 2048
        assert request.temperature == 0.5
        assert request.stream is True
        assert request.routing_override == "local"

    def test_temperature_bounds(self) -> None:
        """Test temperature validation."""
        # Valid bounds
        InferenceRequest(
            messages=[Message(role="user", content="test")],
            temperature=0.0,
        )
        InferenceRequest(
            messages=[Message(role="user", content="test")],
            temperature=2.0,
        )

        # Invalid bounds
        with pytest.raises(ValidationError):
            InferenceRequest(
                messages=[Message(role="user", content="test")],
                temperature=-0.1,
            )
        with pytest.raises(ValidationError):
            InferenceRequest(
                messages=[Message(role="user", content="test")],
                temperature=2.1,
            )

    def test_max_tokens_bounds(self) -> None:
        """Test max_tokens validation."""
        with pytest.raises(ValidationError):
            InferenceRequest(
                messages=[Message(role="user", content="test")],
                max_tokens=0,
            )
        with pytest.raises(ValidationError):
            InferenceRequest(
                messages=[Message(role="user", content="test")],
                max_tokens=20000,
            )

    def test_routing_override_values(self) -> None:
        """Test routing_override accepts valid values."""
        for override in ["local", "cloud", None]:
            request = InferenceRequest(
                messages=[Message(role="user", content="test")],
                routing_override=override,
            )
            assert request.routing_override == override


class TestSentinelMetadata:
    """Tests for SentinelMetadata schema."""

    def test_minimal_metadata(self) -> None:
        """Test metadata with minimal fields."""
        meta = SentinelMetadata(
            route="local",
            backend="ollama",
            model="gemma3:4b",
        )
        assert meta.route == "local"
        assert meta.backend == "ollama"
        assert meta.model == "gemma3:4b"
        assert meta.privacy_tier == 0
        assert meta.privacy_tier_label == "PUBLIC"
        assert meta.entities_detected == []
        assert meta.cost_usd == 0.0

    def test_full_metadata(self) -> None:
        """Test metadata with all fields."""
        meta = SentinelMetadata(
            route="local",
            backend="ollama",
            endpoint="mac-mini",
            model="gemma3:4b",
            privacy_tier=2,
            privacy_tier_label="CONFIDENTIAL",
            entities_detected=["email", "phone"],
            classification_latency_ms=5.2,
            routing_latency_ms=1.1,
            inference_latency_ms=1500.0,
            ttft_ms=250.0,
            itl_p50_ms=12.5,
            itl_p95_ms=18.0,
            tpot_ms=14.0,
            tokens_per_second=75.0,
            cost_usd=0.0,
            cost_savings_usd=0.015,
        )
        assert meta.endpoint == "mac-mini"
        assert meta.privacy_tier == 2
        assert meta.ttft_ms == 250.0
        assert meta.itl_p50_ms == 12.5


class TestInferenceResponse:
    """Tests for InferenceResponse schema."""

    def test_valid_response(self) -> None:
        """Test valid response creation."""
        response = InferenceResponse(
            id="req_abc123",
            model="gemma3:4b",
            choices=[
                Choice(
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            sentinel=SentinelMetadata(
                route="local",
                backend="ollama",
                model="gemma3:4b",
            ),
        )
        assert response.id == "req_abc123"
        assert len(response.choices) == 1
        assert response.usage.total_tokens == 15


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy_response(self) -> None:
        """Test healthy status response."""
        response = HealthResponse(
            status="healthy",
            version="0.1.0",
            backends={"mac-mini": True, "macbook": True},
        )
        assert response.status == "healthy"
        assert response.backends["mac-mini"] is True

    def test_degraded_response(self) -> None:
        """Test degraded status response."""
        response = HealthResponse(
            status="degraded",
            version="0.1.0",
            backends={"mac-mini": True, "macbook": False},
        )
        assert response.status == "degraded"


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_minimal_error(self) -> None:
        """Test error with minimal fields."""
        error = ErrorResponse(error="Something went wrong")
        assert error.error == "Something went wrong"
        assert error.detail is None
        assert error.request_id is None

    def test_full_error(self) -> None:
        """Test error with all fields."""
        error = ErrorResponse(
            error="Inference failed",
            detail="Connection timeout to backend",
            request_id="req_xyz789",
        )
        assert error.error == "Inference failed"
        assert error.detail == "Connection timeout to backend"
        assert error.request_id == "req_xyz789"
