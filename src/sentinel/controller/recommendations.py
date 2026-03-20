"""Recommendation dataclasses and enums for closed-loop controller."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


class RecommendationType(Enum):
    """Types of routing recommendations."""
    
    ROUTE_TO_LOCAL = "route_to_local"       # Quality good, save money
    KEEP_ON_CLOUD = "keep_on_cloud"         # Quality insufficient
    DRIFT_ALERT = "drift_alert"             # Quality degraded from previous
    INSUFFICIENT_DATA = "insufficient_data"  # Need more samples
    NO_CHANGE = "no_change"                  # Current config is optimal


class Confidence(Enum):
    """Confidence level for recommendations."""
    
    HIGH = "high"      # >500 samples, stable metrics
    MEDIUM = "medium"  # 100-500 samples
    LOW = "low"        # <100 samples


@dataclass
class TierMetrics:
    """Aggregated metrics for a single tier."""
    
    tier: int
    sample_count: int = 0
    avg_similarity: float = 0.0
    min_similarity: float = 1.0
    max_similarity: float = 0.0
    p50_similarity: float = 0.0
    p95_similarity: float = 0.0
    
    avg_latency_diff_ms: float = 0.0  # Negative = local faster
    total_cost_savings_usd: float = 0.0
    
    # Quality match rate (similarity >= threshold)
    quality_match_count: int = 0
    quality_match_rate: float = 0.0
    
    window_start: datetime | None = None
    window_end: datetime | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tier": self.tier,
            "sample_count": self.sample_count,
            "avg_similarity": round(self.avg_similarity, 4),
            "min_similarity": round(self.min_similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "p50_similarity": round(self.p50_similarity, 4),
            "p95_similarity": round(self.p95_similarity, 4),
            "avg_latency_diff_ms": round(self.avg_latency_diff_ms, 2),
            "total_cost_savings_usd": round(self.total_cost_savings_usd, 4),
            "quality_match_count": self.quality_match_count,
            "quality_match_rate": round(self.quality_match_rate, 4),
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
        }


@dataclass
class Recommendation:
    """A single routing recommendation for a tier."""
    
    tier: int
    recommendation: RecommendationType
    reason: str
    confidence: Confidence
    
    # Supporting metrics
    current_similarity: float = 0.0
    threshold_similarity: float = 0.0
    sample_count: int = 0
    min_samples_required: int = 0
    
    # Cost analysis
    potential_savings_usd: float = 0.0
    
    # Drift detection
    previous_similarity: float | None = None
    similarity_delta: float | None = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tier": self.tier,
            "recommendation": self.recommendation.value,
            "reason": self.reason,
            "confidence": self.confidence.value,
            "current_similarity": round(self.current_similarity, 4),
            "threshold_similarity": round(self.threshold_similarity, 4),
            "sample_count": self.sample_count,
            "min_samples_required": self.min_samples_required,
            "potential_savings_usd": round(self.potential_savings_usd, 4),
            "previous_similarity": round(self.previous_similarity, 4) if self.previous_similarity else None,
            "similarity_delta": round(self.similarity_delta, 4) if self.similarity_delta else None,
            "created_at": self.created_at.isoformat(),
        }
    
    def to_log_dict(self) -> dict:
        """Convert to dictionary for structured logging."""
        return {
            "event": "controller_recommendation",
            "tier": self.tier,
            "recommendation": self.recommendation.value,
            "reason": self.reason,
            "confidence": self.confidence.value,
            "similarity": round(self.current_similarity, 4),
            "threshold": round(self.threshold_similarity, 4),
            "samples": self.sample_count,
            "savings_usd": round(self.potential_savings_usd, 4),
        }


@dataclass
class ControllerStatus:
    """Current status of the closed-loop controller."""
    
    enabled: bool
    mode: Literal["observe", "auto"]
    running: bool = False
    
    last_evaluation: datetime | None = None
    next_evaluation: datetime | None = None
    evaluation_interval_seconds: int = 60
    
    total_evaluations: int = 0
    
    # Current recommendations per tier
    recommendations: list[Recommendation] = field(default_factory=list)
    
    # Aggregated metrics per tier
    tier_metrics: dict[int, TierMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "running": self.running,
            "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None,
            "next_evaluation": self.next_evaluation.isoformat() if self.next_evaluation else None,
            "evaluation_interval_seconds": self.evaluation_interval_seconds,
            "total_evaluations": self.total_evaluations,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "tier_metrics": {k: v.to_dict() for k, v in self.tier_metrics.items()},
        }


@dataclass 
class ControllerConfig:
    """Configuration for the closed-loop controller."""
    
    enabled: bool = False
    mode: Literal["observe", "auto"] = "observe"
    evaluation_interval_seconds: int = 60
    window_seconds: int = 300  # 5 minute rolling window
    
    # Per-tier thresholds
    tier_thresholds: dict[int, dict] = field(default_factory=lambda: {
        0: {"min_similarity": 0.85, "min_samples": 100},
        1: {"min_similarity": 0.80, "min_samples": 100},
    })
    
    # Alert thresholds
    drift_threshold: float = 0.10  # Alert if similarity drops by 10%
    cost_savings_threshold_usd: float = 50.0  # Min savings to recommend
    
    @classmethod
    def from_dict(cls, data: dict) -> "ControllerConfig":
        """Create from dictionary (e.g., parsed YAML)."""
        tier_thresholds = {}
        thresholds_data = data.get("thresholds", {})
        
        for key, value in thresholds_data.items():
            if key.startswith("tier_"):
                tier_num = int(key.split("_")[1])
                tier_thresholds[tier_num] = value
        
        return cls(
            enabled=data.get("enabled", False),
            mode=data.get("mode", "observe"),
            evaluation_interval_seconds=data.get("evaluation_interval_seconds", 60),
            window_seconds=data.get("window_seconds", 300),
            tier_thresholds=tier_thresholds if tier_thresholds else cls().tier_thresholds,
            drift_threshold=data.get("alerts", {}).get("drift_threshold", 0.10),
            cost_savings_threshold_usd=data.get("alerts", {}).get("cost_savings_threshold_usd", 50.0),
        )
