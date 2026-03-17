"""Unit tests for shadow mode module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from sentinel.shadow.similarity import (
    SimilarityScorer,
    SimilarityResult,
    get_similarity_scorer,
    configure_similarity,
)
from sentinel.shadow.shadow_runner import (
    ShadowRunner,
    ShadowResult,
    ShadowConfig,
    get_shadow_runner,
    configure_shadow,
)


class TestSimilarityResult:
    """Test SimilarityResult dataclass."""
    
    def test_high_similarity_is_quality_match(self):
        """High similarity score should be a quality match."""
        result = SimilarityResult(
            similarity_score=0.90,
            interpretation="high",
            cloud_response_length=100,
            local_response_length=95,
            length_ratio=0.95,
            latency_ms=10.0,
            model_name="test-model",
        )
        assert result.is_quality_match is True
    
    def test_low_similarity_is_not_quality_match(self):
        """Low similarity score should not be a quality match."""
        result = SimilarityResult(
            similarity_score=0.50,
            interpretation="low",
            cloud_response_length=100,
            local_response_length=50,
            length_ratio=0.5,
            latency_ms=10.0,
            model_name="test-model",
        )
        assert result.is_quality_match is False
    
    def test_threshold_boundary(self):
        """Score at 0.75 threshold should be a quality match."""
        result = SimilarityResult(
            similarity_score=0.75,
            interpretation="medium",
            cloud_response_length=100,
            local_response_length=100,
            length_ratio=1.0,
            latency_ms=10.0,
            model_name="test-model",
        )
        assert result.is_quality_match is True
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        result = SimilarityResult(
            similarity_score=0.85,
            interpretation="high",
            cloud_response_length=100,
            local_response_length=95,
            length_ratio=0.95,
            latency_ms=10.5,
            model_name="test-model",
        )
        
        d = result.to_dict()
        
        assert d["similarity_score"] == 0.85
        assert d["interpretation"] == "high"
        assert d["is_quality_match"] is True


class TestSimilarityScorer:
    """Test SimilarityScorer class."""
    
    def test_scorer_disabled_by_default(self):
        """Scorer can be disabled."""
        scorer = SimilarityScorer(enabled=False)
        assert scorer.enabled is False
    
    def test_model_selection(self):
        """Scorer should map model keys to model names."""
        scorer = SimilarityScorer(model_name="fast")
        assert scorer.model_name == "all-MiniLM-L6-v2"
        
        scorer = SimilarityScorer(model_name="balanced")
        assert scorer.model_name == "all-mpnet-base-v2"
    
    @pytest.mark.asyncio
    async def test_compute_similarity_when_disabled(self):
        """Should return error result when disabled."""
        scorer = SimilarityScorer(enabled=False)
        
        result = await scorer.compute_similarity(
            cloud_response="Hello world",
            local_response="Hello world",
        )
        
        assert result.similarity_score == 0.0
        assert result.error == "Similarity scorer disabled"
    
    @pytest.mark.asyncio
    async def test_compute_similarity_with_mock_model(self):
        """Should compute similarity with mocked model."""
        pytest.importorskip("numpy")
        import numpy as np
        
        scorer = SimilarityScorer(enabled=True)
        
        # Mock the model
        mock_model = MagicMock()
        # Return normalized embeddings (same vector = similarity 1.0)
        normalized = np.array([0.5, 0.5, 0.5, 0.5])
        normalized = normalized / np.linalg.norm(normalized)
        mock_model.encode.return_value = np.array([normalized, normalized])
        
        scorer._model = mock_model
        scorer._initialized = True
        
        result = await scorer.compute_similarity(
            cloud_response="Hello world",
            local_response="Hello world",
        )
        
        assert result.error is None
        assert result.similarity_score >= 0.99  # Should be ~1.0 for identical embeddings
        assert result.interpretation == "high"


class TestSimilarityScorerGlobal:
    """Test global similarity scorer functions."""
    
    def test_get_similarity_scorer_returns_instance(self):
        """get_similarity_scorer should return an instance."""
        scorer = get_similarity_scorer()
        assert isinstance(scorer, SimilarityScorer)
    
    def test_configure_similarity(self):
        """configure_similarity should create configured scorer."""
        scorer = configure_similarity(
            model_name="fast",
            device="cpu",
            enabled=True,
        )
        
        assert scorer.model_name == "all-MiniLM-L6-v2"
        assert scorer.device == "cpu"
        assert scorer.enabled is True


class TestShadowConfig:
    """Test ShadowConfig dataclass."""
    
    def test_default_config(self):
        """Default config should have sane defaults."""
        config = ShadowConfig()
        
        assert config.enabled is False
        assert config.shadow_tiers == [0, 1]
        assert config.sample_rate == 1.0
        assert config.similarity_enabled is True
    
    def test_custom_config(self):
        """Should accept custom values."""
        config = ShadowConfig(
            enabled=True,
            shadow_tiers=[0],
            sample_rate=0.5,
            similarity_enabled=False,
        )
        
        assert config.enabled is True
        assert config.shadow_tiers == [0]
        assert config.sample_rate == 0.5


class TestShadowResult:
    """Test ShadowResult dataclass."""
    
    def test_local_is_faster_when_negative_diff(self):
        """Negative latency diff means local is faster."""
        result = ShadowResult(
            shadow_id="shadow_123",
            request_id="req_456",
            timestamp=datetime.now(timezone.utc).isoformat(),
            cloud_response="",
            local_response="",
            cloud_model="claude-3",
            local_model="gemma3:4b",
            cloud_backend="anthropic",
            local_backend="ollama",
            cloud_latency_ms=2000.0,
            local_latency_ms=500.0,
            latency_diff_ms=-1500.0,  # local - cloud
            cloud_tokens=100,
            local_tokens=100,
            cloud_cost_usd=0.001,
            local_cost_usd=0.0,
            cost_savings_usd=0.001,
        )
        
        assert result.local_is_faster is True
    
    def test_quality_match_with_similarity(self):
        """Should check similarity for quality match."""
        similarity = SimilarityResult(
            similarity_score=0.85,
            interpretation="high",
            cloud_response_length=100,
            local_response_length=100,
            length_ratio=1.0,
            latency_ms=10.0,
            model_name="test",
        )
        
        result = ShadowResult(
            shadow_id="shadow_123",
            request_id="req_456",
            timestamp=datetime.now(timezone.utc).isoformat(),
            cloud_response="",
            local_response="",
            cloud_model="claude-3",
            local_model="gemma3:4b",
            cloud_backend="anthropic",
            local_backend="ollama",
            cloud_latency_ms=2000.0,
            local_latency_ms=500.0,
            latency_diff_ms=-1500.0,
            cloud_tokens=100,
            local_tokens=100,
            cloud_cost_usd=0.001,
            local_cost_usd=0.0,
            cost_savings_usd=0.001,
            similarity=similarity,
        )
        
        assert result.is_quality_match is True


class TestShadowRunner:
    """Test ShadowRunner class."""
    
    def test_should_shadow_when_disabled(self):
        """Should not shadow when disabled."""
        config = ShadowConfig(enabled=False)
        runner = ShadowRunner(config=config)
        
        assert runner.should_shadow(privacy_tier=0) is False
    
    def test_should_shadow_for_tier_0(self):
        """Should shadow tier 0 when enabled."""
        config = ShadowConfig(enabled=True, shadow_tiers=[0, 1])
        runner = ShadowRunner(config=config)
        
        assert runner.should_shadow(privacy_tier=0) is True
    
    def test_should_not_shadow_tier_2(self):
        """Should not shadow tier 2 (not in shadow_tiers)."""
        config = ShadowConfig(enabled=True, shadow_tiers=[0, 1])
        runner = ShadowRunner(config=config)
        
        assert runner.should_shadow(privacy_tier=2) is False
    
    def test_should_not_shadow_tier_3(self):
        """Should never shadow tier 3."""
        config = ShadowConfig(enabled=True, shadow_tiers=[0, 1])
        runner = ShadowRunner(config=config)
        
        assert runner.should_shadow(privacy_tier=3) is False
    
    def test_sampling_rate_respected(self):
        """Should respect sampling rate."""
        config = ShadowConfig(enabled=True, shadow_tiers=[0, 1], sample_rate=0.0)
        runner = ShadowRunner(config=config)
        
        # With 0% sample rate, should never shadow
        assert runner.should_shadow(privacy_tier=0) is False
    
    def test_get_metrics_empty(self):
        """Should return metrics when empty."""
        runner = ShadowRunner()
        metrics = runner.get_metrics()
        
        assert metrics["total_shadows"] == 0
        assert metrics["successful_shadows"] == 0
        assert metrics["quality_match_rate"] == 0.0
    
    def test_get_recent_results_empty(self):
        """Should return empty list when no results."""
        runner = ShadowRunner()
        results = runner.get_recent_results()
        
        assert results == []


class TestShadowRunnerGlobal:
    """Test global shadow runner functions."""
    
    def test_get_shadow_runner_returns_instance(self):
        """get_shadow_runner should return an instance."""
        runner = get_shadow_runner()
        assert isinstance(runner, ShadowRunner)
    
    def test_configure_shadow(self):
        """configure_shadow should create configured runner."""
        runner = configure_shadow(
            enabled=True,
            shadow_tiers=[0],
            sample_rate=0.5,
            similarity_enabled=False,
        )
        
        assert runner.config.enabled is True
        assert runner.config.shadow_tiers == [0]
        assert runner.config.sample_rate == 0.5
        assert runner.config.similarity_enabled is False
