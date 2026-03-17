"""Telemetry module for observability.

This module provides:
- Prometheus metrics definitions and helpers
- Structured logging with structlog
- OpenTelemetry tracing integration
"""

from sentinel.telemetry.metrics import (
    # Metric objects
    REQUESTS_TOTAL,
    REQUESTS_IN_PROGRESS,
    TTFT_SECONDS,
    ITL_SECONDS,
    TPOT_SECONDS,
    INFERENCE_LATENCY_SECONDS,
    CLASSIFICATION_LATENCY_SECONDS,
    ROUTING_LATENCY_SECONDS,
    TOKENS_PER_SECOND,
    TOKENS_TOTAL,
    CLASSIFICATIONS_TOTAL,
    ENTITIES_DETECTED_TOTAL,
    COST_USD_TOTAL,
    COST_SAVINGS_USD_TOTAL,
    LOCAL_BACKEND_HEALTHY,
    CLOUD_BACKEND_HEALTHY,
    ERRORS_TOTAL,
    FALLBACK_TOTAL,
    # Shadow metrics
    SHADOW_REQUESTS_TOTAL,
    SHADOW_QUALITY_MATCH,
    SHADOW_SIMILARITY_SCORE,
    SHADOW_LATENCY_DIFF_SECONDS,
    SHADOW_COST_SAVINGS_USD,
    # Helper functions
    get_metrics,
    get_content_type,
    init_app_info,
    record_request,
    record_latencies,
    record_tokens,
    record_classification,
    record_cost,
    record_routing_latency,
    set_backend_health,
    record_error,
    record_fallback,
    record_shadow_result,
)

from sentinel.telemetry.logging import (
    setup_logging,
    get_logger,
    InferenceLogger,
    ClassificationLogger,
    RoutingLogger,
    BackendLogger,
)

from sentinel.telemetry.tracing import (
    setup_tracing,
    get_tracer,
    trace_span,
    trace_inference_request,
    trace_classification,
    trace_routing,
    trace_backend_call,
    end_span_success,
    end_span_error,
    traced,
    extract_trace_context,
    inject_trace_context,
)

__all__ = [
    # Metrics
    "REQUESTS_TOTAL",
    "REQUESTS_IN_PROGRESS",
    "TTFT_SECONDS",
    "ITL_SECONDS",
    "TPOT_SECONDS",
    "INFERENCE_LATENCY_SECONDS",
    "CLASSIFICATION_LATENCY_SECONDS",
    "ROUTING_LATENCY_SECONDS",
    "TOKENS_PER_SECOND",
    "TOKENS_TOTAL",
    "CLASSIFICATIONS_TOTAL",
    "ENTITIES_DETECTED_TOTAL",
    "COST_USD_TOTAL",
    "COST_SAVINGS_USD_TOTAL",
    "LOCAL_BACKEND_HEALTHY",
    "CLOUD_BACKEND_HEALTHY",
    "ERRORS_TOTAL",
    "FALLBACK_TOTAL",
    # Shadow metrics
    "SHADOW_REQUESTS_TOTAL",
    "SHADOW_QUALITY_MATCH",
    "SHADOW_SIMILARITY_SCORE",
    "SHADOW_LATENCY_DIFF_SECONDS",
    "SHADOW_COST_SAVINGS_USD",
    "get_metrics",
    "get_content_type",
    "init_app_info",
    "record_request",
    "record_latencies",
    "record_tokens",
    "record_classification",
    "record_cost",
    "record_routing_latency",
    "set_backend_health",
    "record_error",
    "record_fallback",
    "record_shadow_result",
    # Logging
    "setup_logging",
    "get_logger",
    "InferenceLogger",
    "ClassificationLogger",
    "RoutingLogger",
    "BackendLogger",
    # Tracing
    "setup_tracing",
    "get_tracer",
    "trace_span",
    "trace_inference_request",
    "trace_classification",
    "trace_routing",
    "trace_backend_call",
    "end_span_success",
    "end_span_error",
    "traced",
    "extract_trace_context",
    "inject_trace_context",
]

