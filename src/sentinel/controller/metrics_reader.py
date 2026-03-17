"""Metrics reader for closed-loop controller.

Reads shadow mode results from ShadowRunner internal state.
No Prometheus dependency - pure Python state access.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, median, quantiles
from typing import TYPE_CHECKING

from sentinel.controller.recommendations import TierMetrics

if TYPE_CHECKING:
    from sentinel.shadow import ShadowRunner


@dataclass
class MetricsSample:
    """A single metrics sample from shadow mode."""
    
    tier: int
    similarity_score: float
    latency_diff_ms: float  # Negative = local faster
    cost_savings_usd: float
    is_quality_match: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsReader:
    """Reads and aggregates metrics from ShadowRunner internal state.
    
    Uses timestamped filtering for rolling window calculations.
    Thread-safe access to shadow runner results.
    """
    
    def __init__(self, window_seconds: int = 300):
        """Initialize metrics reader.
        
        Args:
            window_seconds: Rolling window size in seconds (default: 5 min)
        """
        self._window_seconds = window_seconds
        self._shadow_runner: "ShadowRunner | None" = None
        
        # Internal sample storage (for when shadow runner not available)
        self._samples: deque[MetricsSample] = deque(maxlen=10000)
        
        # Previous evaluation results (for drift detection)
        self._previous_tier_metrics: dict[int, TierMetrics] = {}
    
    def set_shadow_runner(self, runner: "ShadowRunner") -> None:
        """Set the shadow runner to read metrics from."""
        self._shadow_runner = runner
    
    def add_sample(self, sample: MetricsSample) -> None:
        """Add a sample directly (for testing or manual input)."""
        self._samples.append(sample)
    
    def get_tier_metrics(
        self,
        tier: int,
        window_seconds: int | None = None,
        quality_threshold: float = 0.85,
    ) -> TierMetrics:
        """Get aggregated metrics for a specific tier.
        
        Args:
            tier: Privacy tier (0-3)
            window_seconds: Override default window (optional)
            quality_threshold: Similarity threshold for quality match
            
        Returns:
            TierMetrics with aggregated statistics
        """
        window = window_seconds or self._window_seconds
        cutoff = datetime.utcnow() - timedelta(seconds=window)
        
        # Collect samples from shadow runner or internal storage
        samples = self._collect_samples(tier, cutoff)
        
        if not samples:
            return TierMetrics(tier=tier)
        
        # Calculate statistics
        similarities = [s.similarity_score for s in samples]
        latency_diffs = [s.latency_diff_ms for s in samples]
        cost_savings = [s.cost_savings_usd for s in samples]
        quality_matches = [s for s in samples if s.similarity_score >= quality_threshold]
        
        # Percentiles
        p50 = median(similarities)
        try:
            p95 = quantiles(similarities, n=20)[18] if len(similarities) >= 20 else max(similarities)
        except Exception:
            p95 = max(similarities)
        
        metrics = TierMetrics(
            tier=tier,
            sample_count=len(samples),
            avg_similarity=mean(similarities),
            min_similarity=min(similarities),
            max_similarity=max(similarities),
            p50_similarity=p50,
            p95_similarity=p95,
            avg_latency_diff_ms=mean(latency_diffs) if latency_diffs else 0.0,
            total_cost_savings_usd=sum(cost_savings),
            quality_match_count=len(quality_matches),
            quality_match_rate=len(quality_matches) / len(samples) if samples else 0.0,
            window_start=min(s.timestamp for s in samples),
            window_end=max(s.timestamp for s in samples),
        )
        
        return metrics
    
    def get_all_tier_metrics(
        self,
        window_seconds: int | None = None,
        quality_threshold: float = 0.85,
    ) -> dict[int, TierMetrics]:
        """Get metrics for all tiers with samples.
        
        Returns:
            Dict mapping tier number to TierMetrics
        """
        window = window_seconds or self._window_seconds
        cutoff = datetime.utcnow() - timedelta(seconds=window)
        
        # Find all tiers with data
        all_samples = self._collect_all_samples(cutoff)
        tiers = set(s.tier for s in all_samples)
        
        return {
            tier: self.get_tier_metrics(tier, window_seconds, quality_threshold)
            for tier in tiers
        }
    
    def get_previous_metrics(self, tier: int) -> TierMetrics | None:
        """Get metrics from previous evaluation (for drift detection)."""
        return self._previous_tier_metrics.get(tier)
    
    def store_current_as_previous(self, metrics: dict[int, TierMetrics]) -> None:
        """Store current metrics for next evaluation's drift detection."""
        self._previous_tier_metrics = metrics.copy()
    
    def _collect_samples(self, tier: int, cutoff: datetime) -> list[MetricsSample]:
        """Collect samples for a tier after cutoff time."""
        samples = []
        
        # Try reading from shadow runner first
        if self._shadow_runner:
            samples.extend(self._read_from_shadow_runner(tier, cutoff))
        
        # Also check internal storage
        samples.extend(
            s for s in self._samples 
            if s.tier == tier and s.timestamp >= cutoff
        )
        
        return samples
    
    def _collect_all_samples(self, cutoff: datetime) -> list[MetricsSample]:
        """Collect all samples after cutoff time."""
        samples = []
        
        # Try reading from shadow runner first
        if self._shadow_runner:
            samples.extend(self._read_all_from_shadow_runner(cutoff))
        
        # Also check internal storage
        samples.extend(s for s in self._samples if s.timestamp >= cutoff)
        
        return samples
    
    def _read_from_shadow_runner(
        self, tier: int, cutoff: datetime
    ) -> list[MetricsSample]:
        """Read samples from shadow runner's internal results."""
        if not self._shadow_runner:
            return []
        
        samples = []
        
        # Access shadow runner's results deque
        # ShadowRunner stores results in _results deque
        results = getattr(self._shadow_runner, "_results", [])
        
        for result in results:
            # Check timestamp
            result_time = getattr(result, "timestamp", None)
            if result_time and result_time < cutoff:
                continue
            
            # Check tier
            result_tier = getattr(result, "tier", None)
            if result_tier != tier:
                continue
            
            # Extract metrics
            similarity = getattr(result, "similarity_score", 0.0)
            latency_diff = getattr(result, "latency_diff_seconds", 0.0) * 1000  # Convert to ms
            cost_savings = getattr(result, "cost_savings_usd", 0.0)
            is_match = getattr(result, "is_quality_match", False)
            
            samples.append(MetricsSample(
                tier=result_tier,
                similarity_score=similarity,
                latency_diff_ms=latency_diff,
                cost_savings_usd=cost_savings,
                is_quality_match=is_match,
                timestamp=result_time or datetime.utcnow(),
            ))
        
        return samples
    
    def _read_all_from_shadow_runner(self, cutoff: datetime) -> list[MetricsSample]:
        """Read all samples from shadow runner after cutoff."""
        if not self._shadow_runner:
            return []
        
        samples = []
        results = getattr(self._shadow_runner, "_results", [])
        
        for result in results:
            result_time = getattr(result, "timestamp", None)
            if result_time and result_time < cutoff:
                continue
            
            result_tier = getattr(result, "tier", None)
            if result_tier is None:
                continue
            
            similarity = getattr(result, "similarity_score", 0.0)
            latency_diff = getattr(result, "latency_diff_seconds", 0.0) * 1000
            cost_savings = getattr(result, "cost_savings_usd", 0.0)
            is_match = getattr(result, "is_quality_match", False)
            
            samples.append(MetricsSample(
                tier=result_tier,
                similarity_score=similarity,
                latency_diff_ms=latency_diff,
                cost_savings_usd=cost_savings,
                is_quality_match=is_match,
                timestamp=result_time or datetime.utcnow(),
            ))
        
        return samples
    
    def clear_samples(self) -> None:
        """Clear internal sample storage."""
        self._samples.clear()
    
    @property
    def window_seconds(self) -> int:
        """Get current window size."""
        return self._window_seconds
    
    @window_seconds.setter
    def window_seconds(self, value: int) -> None:
        """Set window size."""
        self._window_seconds = value
