"""Unit tests for configuration settings."""

import os

import pytest

from sentinel.config import (
    CloudBackendsConfig,
    LocalBackendsConfig,
    LocalEndpoint,
    Settings,
    TelemetryConfig,
)


class TestLocalEndpoint:
    """Tests for LocalEndpoint configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        endpoint = LocalEndpoint(name="test", model="gemma3:4b")
        assert endpoint.host == "localhost"
        assert endpoint.port == 11434
        assert endpoint.priority == 1
        assert endpoint.enabled is True

    def test_base_url(self) -> None:
        """Test base_url property construction."""
        endpoint = LocalEndpoint(
            name="test",
            host="192.168.1.10",
            port=11434,
            model="gemma3:4b",
        )
        assert endpoint.base_url == "http://192.168.1.10:11434"

    def test_custom_values(self) -> None:
        """Test custom values are preserved."""
        endpoint = LocalEndpoint(
            name="macbook",
            host="192.168.1.20",
            port=11435,
            model="llama3.2:8b",
            priority=2,
            enabled=False,
        )
        assert endpoint.name == "macbook"
        assert endpoint.host == "192.168.1.20"
        assert endpoint.port == 11435
        assert endpoint.model == "llama3.2:8b"
        assert endpoint.priority == 2
        assert endpoint.enabled is False


class TestLocalBackendsConfig:
    """Tests for LocalBackendsConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = LocalBackendsConfig()
        assert config.endpoints == []
        assert config.selection_strategy == "priority"
        assert config.health_check_interval_seconds == 30
        assert config.failover_enabled is True
        assert config.timeout_seconds == 120.0

    def test_with_endpoints(self) -> None:
        """Test configuration with endpoints."""
        endpoint = LocalEndpoint(name="test", model="gemma3:4b")
        config = LocalBackendsConfig(endpoints=[endpoint])
        assert len(config.endpoints) == 1
        assert config.endpoints[0].name == "test"


class TestCloudBackendsConfig:
    """Tests for CloudBackendsConfig."""

    def test_default_values(self) -> None:
        """Test default values (with env vars cleared)."""
        # Clear API key env vars that would be picked up
        env_vars_to_clear = ["ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
        original_values = {}
        for key in env_vars_to_clear:
            if key in os.environ:
                original_values[key] = os.environ.pop(key)

        try:
            config = CloudBackendsConfig()
            assert config.primary == "anthropic"
            assert config.fallback == "google"
            assert config.anthropic_api_key is None
            assert config.google_api_key is None
        finally:
            # Restore original env vars
            os.environ.update(original_values)

    def test_reads_standard_env_vars(self) -> None:
        """Test that config reads from standard API key env vars."""
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
        os.environ["GOOGLE_API_KEY"] = "test-google-key"

        try:
            config = CloudBackendsConfig()
            assert config.anthropic_api_key == "test-anthropic-key"
            assert config.google_api_key == "test-google-key"
        finally:
            del os.environ["ANTHROPIC_API_KEY"]
            del os.environ["GOOGLE_API_KEY"]

    def test_default_models(self) -> None:
        """Test default model names."""
        config = CloudBackendsConfig()
        assert "claude" in config.anthropic_model.lower() or "claude" in config.anthropic_model
        assert "gemini" in config.google_model.lower()


class TestTelemetryConfig:
    """Tests for TelemetryConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = TelemetryConfig()
        assert config.enabled is True
        assert config.service_name == "inference-sentinel"
        assert config.log_level == "INFO"
        assert config.metrics_port == 9090


class TestSettings:
    """Tests for main Settings class."""

    def test_default_values(self) -> None:
        """Test default values."""
        # Clear any existing env vars that might interfere
        env_vars_to_clear = [k for k in os.environ if k.startswith("SENTINEL_")]
        original_values = {k: os.environ.pop(k) for k in env_vars_to_clear}

        try:
            # Create fresh settings without cache
            settings = Settings()
            assert settings.env == "development"
            assert settings.debug is False
            assert settings.host == "0.0.0.0"
            assert settings.port == 8000
        finally:
            # Restore original env vars
            os.environ.update(original_values)

    def test_debug_from_string(self) -> None:
        """Test debug flag parsing from string."""
        os.environ["SENTINEL_DEBUG"] = "true"
        try:
            settings = Settings()
            assert settings.debug is True
        finally:
            del os.environ["SENTINEL_DEBUG"]

    def test_debug_from_bool_string_variations(self) -> None:
        """Test various string representations of boolean."""
        for true_val in ["true", "True", "TRUE", "1", "yes", "Yes"]:
            os.environ["SENTINEL_DEBUG"] = true_val
            try:
                settings = Settings()
                assert settings.debug is True, f"Failed for value: {true_val}"
            finally:
                del os.environ["SENTINEL_DEBUG"]
