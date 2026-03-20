"""Rule definitions and evaluation for closed-loop controller.

Evaluates metrics against thresholds to generate recommendations.
"""

from dataclasses import dataclass
from datetime import datetime

from sentinel.controller.recommendations import (
    Confidence,
    ControllerConfig,
    Recommendation,
    RecommendationType,
    TierMetrics,
)


@dataclass
class RuleContext:
    """Context for rule evaluation."""
    
    current_metrics: TierMetrics
    previous_metrics: TierMetrics | None
    config: ControllerConfig
    tier: int
    
    @property
    def threshold_config(self) -> dict:
        """Get threshold config for this tier."""
        return self.config.tier_thresholds.get(self.tier, {
            "min_similarity": 0.85,
            "min_samples": 100,
        })


class RuleEngine:
    """Evaluates metrics against rules to generate recommendations."""
    
    def __init__(self, config: ControllerConfig):
        self._config = config
    
    def update_config(self, config: ControllerConfig) -> None:
        """Update configuration (for hot reload)."""
        self._config = config
    
    def evaluate(
        self,
        current: TierMetrics,
        previous: TierMetrics | None = None,
    ) -> Recommendation:
        """Evaluate metrics for a tier and generate recommendation.
        
        Args:
            current: Current aggregated metrics
            previous: Previous evaluation's metrics (for drift detection)
            
        Returns:
            Recommendation for this tier
        """
        ctx = RuleContext(
            current_metrics=current,
            previous_metrics=previous,
            config=self._config,
            tier=current.tier,
        )
        
        # Rule priority: check in order
        # 1. Insufficient data
        if self._check_insufficient_data(ctx):
            return self._create_insufficient_data_recommendation(ctx)
        
        # 2. Drift alert (quality degradation)
        if self._check_drift(ctx):
            return self._create_drift_recommendation(ctx)
        
        # 3. Route to local (quality good + cost savings)
        if self._check_route_to_local(ctx):
            return self._create_route_to_local_recommendation(ctx)
        
        # 4. Keep on cloud (quality insufficient)
        if self._check_keep_on_cloud(ctx):
            return self._create_keep_on_cloud_recommendation(ctx)
        
        # 5. No change needed
        return self._create_no_change_recommendation(ctx)
    
    def _check_insufficient_data(self, ctx: RuleContext) -> bool:
        """Check if we have enough samples."""
        min_samples = ctx.threshold_config.get("min_samples", 100)
        return ctx.current_metrics.sample_count < min_samples
    
    def _check_drift(self, ctx: RuleContext) -> bool:
        """Check for quality drift (degradation)."""
        if not ctx.previous_metrics:
            return False
        
        prev_sim = ctx.previous_metrics.avg_similarity
        curr_sim = ctx.current_metrics.avg_similarity
        
        # Only alert on significant drops
        if prev_sim == 0:
            return False
        
        delta = prev_sim - curr_sim  # Positive = degradation
        return delta >= self._config.drift_threshold
    
    def _check_route_to_local(self, ctx: RuleContext) -> bool:
        """Check if quality is good enough to route to local."""
        min_similarity = ctx.threshold_config.get("min_similarity", 0.85)
        
        # Average similarity meets threshold
        if ctx.current_metrics.avg_similarity < min_similarity:
            return False
        
        # Cost savings worth it
        if ctx.current_metrics.total_cost_savings_usd < self._config.cost_savings_threshold_usd:
            # Still recommend if quality is very high
            return ctx.current_metrics.avg_similarity >= 0.95
        
        return True
    
    def _check_keep_on_cloud(self, ctx: RuleContext) -> bool:
        """Check if quality is too low for local routing."""
        min_similarity = ctx.threshold_config.get("min_similarity", 0.85)
        return ctx.current_metrics.avg_similarity < min_similarity
    
    def _get_confidence(self, sample_count: int) -> Confidence:
        """Determine confidence level based on sample count."""
        if sample_count >= 500:
            return Confidence.HIGH
        elif sample_count >= 100:
            return Confidence.MEDIUM
        return Confidence.LOW
    
    def _create_insufficient_data_recommendation(
        self, ctx: RuleContext
    ) -> Recommendation:
        """Create recommendation for insufficient data."""
        min_samples = ctx.threshold_config.get("min_samples", 100)
        
        return Recommendation(
            tier=ctx.tier,
            recommendation=RecommendationType.INSUFFICIENT_DATA,
            reason=f"Only {ctx.current_metrics.sample_count} samples, need {min_samples} for evaluation",
            confidence=Confidence.LOW,
            current_similarity=ctx.current_metrics.avg_similarity,
            threshold_similarity=ctx.threshold_config.get("min_similarity", 0.85),
            sample_count=ctx.current_metrics.sample_count,
            min_samples_required=min_samples,
            potential_savings_usd=ctx.current_metrics.total_cost_savings_usd,
        )
    
    def _create_drift_recommendation(self, ctx: RuleContext) -> Recommendation:
        """Create recommendation for quality drift."""
        prev_sim = ctx.previous_metrics.avg_similarity if ctx.previous_metrics else 0.0
        curr_sim = ctx.current_metrics.avg_similarity
        delta = prev_sim - curr_sim
        
        return Recommendation(
            tier=ctx.tier,
            recommendation=RecommendationType.DRIFT_ALERT,
            reason=f"Quality degraded by {delta:.1%} (from {prev_sim:.2%} to {curr_sim:.2%})",
            confidence=self._get_confidence(ctx.current_metrics.sample_count),
            current_similarity=curr_sim,
            threshold_similarity=ctx.threshold_config.get("min_similarity", 0.85),
            sample_count=ctx.current_metrics.sample_count,
            min_samples_required=ctx.threshold_config.get("min_samples", 100),
            potential_savings_usd=ctx.current_metrics.total_cost_savings_usd,
            previous_similarity=prev_sim,
            similarity_delta=-delta,  # Negative = degradation
        )
    
    def _create_route_to_local_recommendation(
        self, ctx: RuleContext
    ) -> Recommendation:
        """Create recommendation to route to local."""
        min_similarity = ctx.threshold_config.get("min_similarity", 0.85)
        
        return Recommendation(
            tier=ctx.tier,
            recommendation=RecommendationType.ROUTE_TO_LOCAL,
            reason=(
                f"Similarity {ctx.current_metrics.avg_similarity:.1%} exceeds threshold "
                f"{min_similarity:.1%} over {ctx.current_metrics.sample_count} samples"
            ),
            confidence=self._get_confidence(ctx.current_metrics.sample_count),
            current_similarity=ctx.current_metrics.avg_similarity,
            threshold_similarity=min_similarity,
            sample_count=ctx.current_metrics.sample_count,
            min_samples_required=ctx.threshold_config.get("min_samples", 100),
            potential_savings_usd=ctx.current_metrics.total_cost_savings_usd,
        )
    
    def _create_keep_on_cloud_recommendation(
        self, ctx: RuleContext
    ) -> Recommendation:
        """Create recommendation to keep on cloud."""
        min_similarity = ctx.threshold_config.get("min_similarity", 0.85)
        gap = min_similarity - ctx.current_metrics.avg_similarity
        
        return Recommendation(
            tier=ctx.tier,
            recommendation=RecommendationType.KEEP_ON_CLOUD,
            reason=(
                f"Similarity {ctx.current_metrics.avg_similarity:.1%} is {gap:.1%} below "
                f"threshold {min_similarity:.1%}"
            ),
            confidence=self._get_confidence(ctx.current_metrics.sample_count),
            current_similarity=ctx.current_metrics.avg_similarity,
            threshold_similarity=min_similarity,
            sample_count=ctx.current_metrics.sample_count,
            min_samples_required=ctx.threshold_config.get("min_samples", 100),
            potential_savings_usd=ctx.current_metrics.total_cost_savings_usd,
        )
    
    def _create_no_change_recommendation(self, ctx: RuleContext) -> Recommendation:
        """Create recommendation for no change needed."""
        return Recommendation(
            tier=ctx.tier,
            recommendation=RecommendationType.NO_CHANGE,
            reason="Current configuration is optimal",
            confidence=self._get_confidence(ctx.current_metrics.sample_count),
            current_similarity=ctx.current_metrics.avg_similarity,
            threshold_similarity=ctx.threshold_config.get("min_similarity", 0.85),
            sample_count=ctx.current_metrics.sample_count,
            min_samples_required=ctx.threshold_config.get("min_samples", 100),
            potential_savings_usd=ctx.current_metrics.total_cost_savings_usd,
        )
