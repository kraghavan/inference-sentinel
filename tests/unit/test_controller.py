"""Unit tests for closed-loop controller."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock

from sentinel.controller import (
    ClosedLoopController,
    ControllerConfig,
    ControllerStatus,
    Recommendation,
    RecommendationType,
    Confidence,
    TierMetrics,
    RuleEngine,
    MetricsReader,
    MetricsSample,
)


class TestTierMetrics:
    """Test TierMetrics dataclass."""
    
    def test_default_metrics(self):
        """Default metrics should have zero values."""
        metrics = TierMetrics(tier=0)
        assert metrics.tier == 0
        assert metrics.sample_count == 0
        assert metrics.avg_similarity == 0.0
        assert metrics.quality_match_rate == 0.0
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = TierMetrics(
            tier=1,
            sample_count=100,
            avg_similarity=0.92,
            quality_match_rate=0.85,
        )
        
        d = metrics.to_dict()
        
        assert d["tier"] == 1
        assert d["sample_count"] == 100
        assert d["avg_similarity"] == 0.92
        assert d["quality_match_rate"] == 0.85


class TestRecommendation:
    """Test Recommendation dataclass."""
    
    def test_create_recommendation(self):
        """Should create a recommendation."""
        rec = Recommendation(
            tier=0,
            recommendation=RecommendationType.ROUTE_TO_LOCAL,
            reason="Quality threshold met",
            confidence=Confidence.HIGH,
            current_similarity=0.92,
            threshold_similarity=0.85,
            sample_count=500,
            min_samples_required=100,
            potential_savings_usd=127.50,
        )
        
        assert rec.tier == 0
        assert rec.recommendation == RecommendationType.ROUTE_TO_LOCAL
        assert rec.confidence == Confidence.HIGH
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        rec = Recommendation(
            tier=0,
            recommendation=RecommendationType.KEEP_ON_CLOUD,
            reason="Quality below threshold",
            confidence=Confidence.MEDIUM,
            current_similarity=0.75,
            threshold_similarity=0.85,
            sample_count=200,
            min_samples_required=100,
        )
        
        d = rec.to_dict()
        
        assert d["tier"] == 0
        assert d["recommendation"] == "keep_on_cloud"
        assert d["confidence"] == "medium"
    
    def test_to_log_dict(self):
        """Should create structured log format."""
        rec = Recommendation(
            tier=1,
            recommendation=RecommendationType.DRIFT_ALERT,
            reason="Quality degraded by 15%",
            confidence=Confidence.HIGH,
            current_similarity=0.72,
            threshold_similarity=0.85,
            sample_count=600,
            min_samples_required=100,
        )
        
        log = rec.to_log_dict()
        
        assert log["event"] == "controller_recommendation"
        assert log["tier"] == 1
        assert log["recommendation"] == "drift_alert"


class TestControllerConfig:
    """Test ControllerConfig dataclass."""
    
    def test_default_config(self):
        """Default config should have sensible values."""
        config = ControllerConfig()
        
        assert config.enabled is False
        assert config.mode == "observe"
        assert config.evaluation_interval_seconds == 60
        assert config.window_seconds == 300
        assert config.drift_threshold == 0.10
    
    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "enabled": True,
            "mode": "observe",
            "evaluation_interval_seconds": 120,
            "window_seconds": 600,
            "thresholds": {
                "tier_0": {"min_similarity": 0.90, "min_samples": 200},
                "tier_1": {"min_similarity": 0.85, "min_samples": 150},
            },
            "alerts": {
                "drift_threshold": 0.15,
                "cost_savings_threshold_usd": 100.0,
            },
        }
        
        config = ControllerConfig.from_dict(data)
        
        assert config.enabled is True
        assert config.evaluation_interval_seconds == 120
        assert config.window_seconds == 600
        assert config.tier_thresholds[0]["min_similarity"] == 0.90
        assert config.tier_thresholds[1]["min_samples"] == 150
        assert config.drift_threshold == 0.15


class TestMetricsReader:
    """Test MetricsReader class."""
    
    def test_default_window(self):
        """Should use default 5 minute window."""
        reader = MetricsReader()
        assert reader.window_seconds == 300
    
    def test_custom_window(self):
        """Should accept custom window size."""
        reader = MetricsReader(window_seconds=600)
        assert reader.window_seconds == 600
    
    def test_add_sample(self):
        """Should add samples to internal storage."""
        reader = MetricsReader()
        
        sample = MetricsSample(
            tier=0,
            similarity_score=0.92,
            latency_diff_ms=-50.0,
            cost_savings_usd=0.01,
            is_quality_match=True,
        )
        reader.add_sample(sample)
        
        metrics = reader.get_tier_metrics(tier=0)
        assert metrics.sample_count == 1
        assert metrics.avg_similarity == 0.92
    
    def test_get_tier_metrics_empty(self):
        """Should return empty metrics when no data."""
        reader = MetricsReader()
        
        metrics = reader.get_tier_metrics(tier=0)
        
        assert metrics.tier == 0
        assert metrics.sample_count == 0
    
    def test_window_filtering(self):
        """Should filter samples by time window."""
        reader = MetricsReader(window_seconds=60)
        
        # Add old sample (outside window)
        old_sample = MetricsSample(
            tier=0,
            similarity_score=0.50,
            latency_diff_ms=100.0,
            cost_savings_usd=0.01,
            is_quality_match=False,
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=120),
        )
        reader.add_sample(old_sample)
        
        # Add recent sample (inside window)
        new_sample = MetricsSample(
            tier=0,
            similarity_score=0.95,
            latency_diff_ms=-50.0,
            cost_savings_usd=0.02,
            is_quality_match=True,
            timestamp=datetime.now(timezone.utc),
        )
        reader.add_sample(new_sample)
        
        metrics = reader.get_tier_metrics(tier=0)
        
        # Should only include the recent sample
        assert metrics.sample_count == 1
        assert metrics.avg_similarity == 0.95
    
    def test_get_all_tier_metrics(self):
        """Should return metrics for all tiers with data."""
        reader = MetricsReader()
        
        reader.add_sample(MetricsSample(
            tier=0, similarity_score=0.90, latency_diff_ms=0, 
            cost_savings_usd=0.01, is_quality_match=True,
        ))
        reader.add_sample(MetricsSample(
            tier=1, similarity_score=0.85, latency_diff_ms=0,
            cost_savings_usd=0.01, is_quality_match=True,
        ))
        
        all_metrics = reader.get_all_tier_metrics()
        
        assert 0 in all_metrics
        assert 1 in all_metrics
        assert all_metrics[0].avg_similarity == 0.90
        assert all_metrics[1].avg_similarity == 0.85
    
    def test_store_previous_metrics(self):
        """Should store and retrieve previous metrics."""
        reader = MetricsReader()
        
        metrics = {
            0: TierMetrics(tier=0, avg_similarity=0.90),
            1: TierMetrics(tier=1, avg_similarity=0.85),
        }
        
        reader.store_current_as_previous(metrics)
        
        prev_0 = reader.get_previous_metrics(0)
        prev_1 = reader.get_previous_metrics(1)
        
        assert prev_0.avg_similarity == 0.90
        assert prev_1.avg_similarity == 0.85


class TestRuleEngine:
    """Test RuleEngine class."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return ControllerConfig(
            tier_thresholds={
                0: {"min_similarity": 0.85, "min_samples": 100},
                1: {"min_similarity": 0.80, "min_samples": 100},
            },
            drift_threshold=0.10,
            cost_savings_threshold_usd=50.0,
        )
    
    @pytest.fixture
    def rule_engine(self, config):
        """Create rule engine with test config."""
        return RuleEngine(config)
    
    def test_insufficient_data(self, rule_engine):
        """Should recommend insufficient data when too few samples."""
        metrics = TierMetrics(tier=0, sample_count=50, avg_similarity=0.92)
        
        rec = rule_engine.evaluate(metrics)
        
        assert rec.recommendation == RecommendationType.INSUFFICIENT_DATA
        assert rec.confidence == Confidence.LOW
    
    def test_route_to_local(self, rule_engine):
        """Should recommend local when quality meets threshold."""
        metrics = TierMetrics(
            tier=0,
            sample_count=500,
            avg_similarity=0.92,
            total_cost_savings_usd=100.0,
        )
        
        rec = rule_engine.evaluate(metrics)
        
        assert rec.recommendation == RecommendationType.ROUTE_TO_LOCAL
        assert rec.confidence == Confidence.HIGH
    
    def test_keep_on_cloud(self, rule_engine):
        """Should recommend cloud when quality is below threshold."""
        metrics = TierMetrics(
            tier=0,
            sample_count=500,
            avg_similarity=0.75,
            total_cost_savings_usd=100.0,
        )
        
        rec = rule_engine.evaluate(metrics)
        
        assert rec.recommendation == RecommendationType.KEEP_ON_CLOUD
    
    def test_drift_alert(self, rule_engine):
        """Should alert on quality drift."""
        previous = TierMetrics(tier=0, sample_count=500, avg_similarity=0.92)
        current = TierMetrics(tier=0, sample_count=500, avg_similarity=0.80)
        
        rec = rule_engine.evaluate(current, previous)
        
        assert rec.recommendation == RecommendationType.DRIFT_ALERT
        assert rec.previous_similarity == 0.92
        assert rec.similarity_delta is not None
    
    def test_no_drift_on_small_change(self, rule_engine):
        """Should not alert on small quality changes."""
        previous = TierMetrics(tier=0, sample_count=500, avg_similarity=0.92)
        current = TierMetrics(
            tier=0, 
            sample_count=500, 
            avg_similarity=0.90,  # Only 2% drop, below threshold
            total_cost_savings_usd=100.0,
        )
        
        rec = rule_engine.evaluate(current, previous)
        
        # Should still recommend local (not drift alert)
        assert rec.recommendation == RecommendationType.ROUTE_TO_LOCAL


class TestClosedLoopController:
    """Test ClosedLoopController class."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return ControllerConfig(
            enabled=True,
            mode="observe",
            evaluation_interval_seconds=60,
            window_seconds=300,
        )
    
    @pytest.fixture
    def controller(self, config):
        """Create controller instance."""
        return ClosedLoopController(config)
    
    def test_init(self, controller, config):
        """Should initialize with config."""
        assert controller.config.enabled is True
        assert controller.config.mode == "observe"
        assert controller.is_running is False
    
    def test_get_status_not_running(self, controller):
        """Should return status when not running."""
        status = controller.get_status()
        
        assert status.enabled is True
        assert status.running is False
        assert status.total_evaluations == 0
        assert status.recommendations == []
    
    @pytest.mark.asyncio
    async def test_start_stop(self, controller):
        """Should start and stop cleanly."""
        # Start
        await controller.start()
        assert controller.is_running is True
        
        # Stop
        await controller.stop()
        assert controller.is_running is False
    
    @pytest.mark.asyncio
    async def test_start_disabled(self):
        """Should not start when disabled."""
        config = ControllerConfig(enabled=False)
        controller = ClosedLoopController(config)
        
        await controller.start()
        
        assert controller.is_running is False
    
    def test_update_config(self, controller):
        """Should update configuration."""
        new_config = ControllerConfig(
            enabled=True,
            mode="observe",
            evaluation_interval_seconds=120,
            window_seconds=600,
        )
        
        controller.update_config(new_config)
        
        assert controller.config.evaluation_interval_seconds == 120
        assert controller.config.window_seconds == 600
    
    def test_get_history_empty(self, controller):
        """Should return empty history initially."""
        history = controller.get_history()
        assert history == []
    
    @pytest.mark.asyncio
    async def test_force_evaluate(self, controller):
        """Should force evaluation."""
        result = await controller.force_evaluate()
        
        assert "evaluation_number" in result
        assert "timestamp" in result
        assert "recommendations" in result
    
    def test_set_shadow_runner(self, controller):
        """Should set shadow runner."""
        mock_runner = MagicMock()
        
        controller.set_shadow_runner(mock_runner)
        
        # Should not raise
        assert True


class TestControllerStatus:
    """Test ControllerStatus dataclass."""
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        status = ControllerStatus(
            enabled=True,
            mode="observe",
            running=True,
            evaluation_interval_seconds=60,
            total_evaluations=10,
        )
        
        d = status.to_dict()
        
        assert d["enabled"] is True
        assert d["mode"] == "observe"
        assert d["running"] is True
        assert d["total_evaluations"] == 10
