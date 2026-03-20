"""Closed-loop controller for adaptive routing optimization.

The controller observes shadow mode metrics and generates routing
recommendations. Currently operates in observe-only mode.

Key components:
- ClosedLoopController: Main controller with background task
- RuleEngine: Evaluates metrics against thresholds
- MetricsReader: Reads from ShadowRunner internal state
- Recommendations: Data classes for results
"""

from sentinel.controller.controller import (
    ClosedLoopController,
    get_controller,
    set_controller,
    initialize_controller,
)
from sentinel.controller.recommendations import (
    ControllerConfig,
    ControllerStatus,
    Recommendation,
    RecommendationType,
    Confidence,
    TierMetrics,
)
from sentinel.controller.rules import RuleEngine
from sentinel.controller.metrics_reader import MetricsReader, MetricsSample

__all__ = [
    # Controller
    "ClosedLoopController",
    "get_controller",
    "set_controller",
    "initialize_controller",
    # Config and status
    "ControllerConfig",
    "ControllerStatus",
    # Recommendations
    "Recommendation",
    "RecommendationType",
    "Confidence",
    "TierMetrics",
    # Rules
    "RuleEngine",
    # Metrics
    "MetricsReader",
    "MetricsSample",
]
