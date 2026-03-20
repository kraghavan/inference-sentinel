"""Closed-loop controller for adaptive routing optimization.

Observes shadow mode metrics and generates routing recommendations.
Currently observe-only (logs recommendations, no auto-action).

Architecture:
- Runs as asyncio background task within FastAPI lifecycle
- Reads metrics from ShadowRunner internal state (no Prometheus dependency)
- Uses rolling window for metric aggregation
- Emits structured logs for Loki/Grafana visualization
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

from sentinel.controller.metrics_reader import MetricsReader
from sentinel.controller.recommendations import (
    ControllerConfig,
    ControllerStatus,
    Recommendation,
    TierMetrics,
)
from sentinel.controller.rules import RuleEngine
from sentinel.telemetry import get_logger

if TYPE_CHECKING:
    from sentinel.shadow import ShadowRunner

logger = get_logger("sentinel.controller")


class ClosedLoopController:
    """Closed-loop controller for routing optimization.
    
    Observes shadow mode quality metrics and recommends routing changes.
    
    Modes:
    - observe: Log recommendations only (default)
    - auto: Apply routing changes automatically (future)
    
    Example usage:
        controller = ClosedLoopController(config)
        controller.set_shadow_runner(shadow_runner)
        await controller.start()
        
        # Later...
        status = controller.get_status()
        await controller.stop()
    """
    
    def __init__(self, config: ControllerConfig):
        """Initialize controller.
        
        Args:
            config: Controller configuration
        """
        self._config = config
        self._metrics_reader = MetricsReader(window_seconds=config.window_seconds)
        self._rule_engine = RuleEngine(config)
        
        # State
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_evaluation: datetime | None = None
        self._total_evaluations: int = 0
        
        # Current recommendations (per tier)
        self._current_recommendations: dict[int, Recommendation] = {}
        self._current_tier_metrics: dict[int, TierMetrics] = {}
        
        # History for /admin/controller/history endpoint
        self._recommendation_history: deque[dict] = deque(maxlen=100)
    
    def set_shadow_runner(self, runner: "ShadowRunner") -> None:
        """Set the shadow runner to read metrics from."""
        self._metrics_reader.set_shadow_runner(runner)
        logger.info("Controller connected to shadow runner")
    
    def update_config(self, config: ControllerConfig) -> None:
        """Update configuration (for hot reload).
        
        Does not restart the background task - new config takes effect
        on next evaluation cycle.
        """
        self._config = config
        self._metrics_reader.window_seconds = config.window_seconds
        self._rule_engine.update_config(config)
        logger.info(
            "Controller config updated",
            enabled=config.enabled,
            mode=config.mode,
            interval=config.evaluation_interval_seconds,
            window=config.window_seconds,
        )
    
    async def start(self) -> None:
        """Start the controller background task."""
        if not self._config.enabled:
            logger.info("Controller disabled, not starting")
            return
        
        if self._running:
            logger.warning("Controller already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Controller started",
            mode=self._config.mode,
            interval_seconds=self._config.evaluation_interval_seconds,
            window_seconds=self._config.window_seconds,
        )
    
    async def stop(self) -> None:
        """Stop the controller background task."""
        if not self._running:
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info("Controller stopped")
    
    async def _run_loop(self) -> None:
        """Main controller loop - runs evaluation at configured interval."""
        while self._running:
            try:
                await self._evaluate()
            except Exception as e:
                logger.error("Controller evaluation failed", error=str(e))
            
            # Wait for next evaluation
            await asyncio.sleep(self._config.evaluation_interval_seconds)
    
    async def _evaluate(self) -> None:
        """Run a single evaluation cycle."""
        self._total_evaluations += 1
        self._last_evaluation = datetime.utcnow()
        
        # Collect metrics for all tiers
        quality_threshold = min(
            t.get("min_similarity", 0.85) 
            for t in self._config.tier_thresholds.values()
        ) if self._config.tier_thresholds else 0.85
        
        tier_metrics = self._metrics_reader.get_all_tier_metrics(
            quality_threshold=quality_threshold
        )
        
        if not tier_metrics:
            logger.debug("No shadow metrics available for evaluation")
            return
        
        # Evaluate each tier
        recommendations = {}
        for tier, metrics in tier_metrics.items():
            previous = self._metrics_reader.get_previous_metrics(tier)
            recommendation = self._rule_engine.evaluate(metrics, previous)
            recommendations[tier] = recommendation
            
            # Log recommendation (structured for Loki)
            self._log_recommendation(recommendation, metrics)
        
        # Store current metrics for next evaluation's drift detection
        self._metrics_reader.store_current_as_previous(tier_metrics)
        
        # Update current state
        self._current_recommendations = recommendations
        self._current_tier_metrics = tier_metrics
        
        # Add to history
        self._recommendation_history.append({
            "timestamp": self._last_evaluation.isoformat(),
            "evaluation_number": self._total_evaluations,
            "recommendations": {
                tier: rec.to_dict() for tier, rec in recommendations.items()
            },
            "tier_metrics": {
                tier: m.to_dict() for tier, m in tier_metrics.items()
            },
        })
        
        logger.info(
            "Controller evaluation complete",
            evaluation_number=self._total_evaluations,
            tiers_evaluated=len(tier_metrics),
            recommendations_count=len(recommendations),
        )
    
    def _log_recommendation(
        self, 
        recommendation: Recommendation, 
        metrics: TierMetrics
    ) -> None:
        """Log a recommendation in structured format for Loki."""
        # Use different log levels based on recommendation type
        log_data = {
            **recommendation.to_log_dict(),
            "sample_count": metrics.sample_count,
            "quality_match_rate": round(metrics.quality_match_rate, 4),
            "avg_latency_diff_ms": round(metrics.avg_latency_diff_ms, 2),
            "total_cost_savings_usd": round(metrics.total_cost_savings_usd, 4),
        }
        
        if recommendation.recommendation.value == "drift_alert":
            logger.warning("Routing recommendation: drift detected", **log_data)
        elif recommendation.recommendation.value == "route_to_local":
            logger.info("Routing recommendation: route to local", **log_data)
        elif recommendation.recommendation.value == "keep_on_cloud":
            logger.info("Routing recommendation: keep on cloud", **log_data)
        elif recommendation.recommendation.value == "insufficient_data":
            logger.debug("Routing recommendation: insufficient data", **log_data)
        else:
            logger.debug("Routing recommendation: no change", **log_data)
    
    def get_status(self) -> ControllerStatus:
        """Get current controller status."""
        next_eval = None
        if self._running and self._last_evaluation:
            next_eval = self._last_evaluation + timedelta(
                seconds=self._config.evaluation_interval_seconds
            )
        
        return ControllerStatus(
            enabled=self._config.enabled,
            mode=self._config.mode,
            running=self._running,
            last_evaluation=self._last_evaluation,
            next_evaluation=next_eval,
            evaluation_interval_seconds=self._config.evaluation_interval_seconds,
            total_evaluations=self._total_evaluations,
            recommendations=list(self._current_recommendations.values()),
            tier_metrics=self._current_tier_metrics,
        )
    
    def get_history(self, limit: int = 20) -> list[dict]:
        """Get recommendation history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of historical evaluation results (newest first)
        """
        history = list(self._recommendation_history)
        history.reverse()  # Newest first
        return history[:limit]
    
    def get_recommendation(self, tier: int) -> Recommendation | None:
        """Get current recommendation for a specific tier."""
        return self._current_recommendations.get(tier)
    
    def get_tier_metrics(self, tier: int) -> TierMetrics | None:
        """Get current metrics for a specific tier."""
        return self._current_tier_metrics.get(tier)
    
    async def force_evaluate(self) -> dict:
        """Force an immediate evaluation (for testing/debugging).
        
        Returns:
            Dict with evaluation results
        """
        await self._evaluate()
        return {
            "evaluation_number": self._total_evaluations,
            "timestamp": self._last_evaluation.isoformat() if self._last_evaluation else None,
            "recommendations": {
                tier: rec.to_dict() 
                for tier, rec in self._current_recommendations.items()
            },
            "tier_metrics": {
                tier: m.to_dict() 
                for tier, m in self._current_tier_metrics.items()
            },
        }
    
    @property
    def is_running(self) -> bool:
        """Check if controller is running."""
        return self._running
    
    @property
    def config(self) -> ControllerConfig:
        """Get current configuration."""
        return self._config


# Global controller instance
_controller: ClosedLoopController | None = None


def get_controller() -> ClosedLoopController | None:
    """Get the global controller instance."""
    return _controller


def set_controller(controller: ClosedLoopController) -> None:
    """Set the global controller instance."""
    global _controller
    _controller = controller


async def initialize_controller(
    config: ControllerConfig,
    shadow_runner: "ShadowRunner | None" = None,
) -> ClosedLoopController:
    """Initialize and start the controller.
    
    Args:
        config: Controller configuration
        shadow_runner: Shadow runner to read metrics from
        
    Returns:
        Initialized controller instance
    """
    controller = ClosedLoopController(config)
    
    if shadow_runner:
        controller.set_shadow_runner(shadow_runner)
    
    set_controller(controller)
    await controller.start()
    
    return controller
