"""Prometheus metrics for inference-sentinel."""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST


# =============================================================================
# Application Info
# =============================================================================
APP_INFO = Info(
    "sentinel",
    "Inference Sentinel application information"
)

# =============================================================================
# Request Metrics
# =============================================================================
REQUESTS_TOTAL = Counter(
    "sentinel_requests_total",
    "Total number of inference requests",
    labelnames=["route", "backend", "endpoint", "tier", "status"]
)

REQUESTS_IN_PROGRESS = Gauge(
    "sentinel_requests_in_progress",
    "Number of requests currently being processed",
    labelnames=["backend"]
)

# =============================================================================
# Latency Metrics (Histograms)
# =============================================================================
# Custom buckets for LLM inference latencies
LATENCY_BUCKETS_MS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000]
LATENCY_BUCKETS_SEC = [b / 1000 for b in LATENCY_BUCKETS_MS]

# Time to First Token
TTFT_SECONDS = Histogram(
    "sentinel_ttft_seconds",
    "Time to first token in seconds",
    labelnames=["backend", "endpoint", "model"],
    buckets=LATENCY_BUCKETS_SEC
)

# Inter-Token Latency
ITL_SECONDS = Histogram(
    "sentinel_itl_seconds",
    "Inter-token latency in seconds",
    labelnames=["backend", "endpoint", "model"],
    buckets=[0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200, 0.500]
)

# Time Per Output Token
TPOT_SECONDS = Histogram(
    "sentinel_tpot_seconds",
    "Time per output token in seconds",
    labelnames=["backend", "endpoint", "model"],
    buckets=[0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200, 0.500]
)

# Total inference latency
INFERENCE_LATENCY_SECONDS = Histogram(
    "sentinel_inference_latency_seconds",
    "Total inference latency in seconds",
    labelnames=["backend", "endpoint", "model"],
    buckets=LATENCY_BUCKETS_SEC
)

# Classification latency
CLASSIFICATION_LATENCY_SECONDS = Histogram(
    "sentinel_classification_latency_seconds",
    "Privacy classification latency in seconds",
    labelnames=["detection_method"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
)

# Routing latency
ROUTING_LATENCY_SECONDS = Histogram(
    "sentinel_routing_latency_seconds",
    "Routing decision latency in seconds",
    buckets=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
)

# =============================================================================
# Throughput Metrics
# =============================================================================
TOKENS_PER_SECOND = Gauge(
    "sentinel_tokens_per_second",
    "Current tokens per second throughput",
    labelnames=["backend", "endpoint", "model"]
)

TOKENS_TOTAL = Counter(
    "sentinel_tokens_total",
    "Total tokens processed",
    labelnames=["backend", "endpoint", "type"]  # type: prompt, completion
)

# =============================================================================
# Privacy/Classification Metrics
# =============================================================================
CLASSIFICATIONS_TOTAL = Counter(
    "sentinel_classifications_total",
    "Total classification operations",
    labelnames=["tier", "tier_label"]
)

ENTITIES_DETECTED_TOTAL = Counter(
    "sentinel_entities_detected_total",
    "Total entities detected by type",
    labelnames=["entity_type", "tier"]
)

PRIVACY_TIER_DISTRIBUTION = Gauge(
    "sentinel_privacy_tier_current",
    "Current request privacy tier (last request)",
    labelnames=["tier_label"]
)

# =============================================================================
# Cost Metrics
# =============================================================================
COST_USD_TOTAL = Counter(
    "sentinel_cost_usd_total",
    "Total inference cost in USD",
    labelnames=["backend", "model"]
)

COST_SAVINGS_USD_TOTAL = Counter(
    "sentinel_cost_savings_usd_total",
    "Total cost savings from local routing in USD"
)

# =============================================================================
# Backend Health Metrics
# =============================================================================
LOCAL_BACKEND_HEALTHY = Gauge(
    "sentinel_local_backend_healthy",
    "Local backend health status (1=healthy, 0=unhealthy)",
    labelnames=["endpoint", "model"]
)

CLOUD_BACKEND_HEALTHY = Gauge(
    "sentinel_cloud_backend_healthy",
    "Cloud backend health status (1=healthy, 0=unhealthy)",
    labelnames=["provider"]
)

# =============================================================================
# Error Metrics
# =============================================================================
ERRORS_TOTAL = Counter(
    "sentinel_errors_total",
    "Total errors by type",
    labelnames=["error_type", "backend"]
)

FALLBACK_TOTAL = Counter(
    "sentinel_fallback_total",
    "Total fallback events (cloud to local, or primary to secondary)",
    labelnames=["from_backend", "to_backend", "reason"]
)

# =============================================================================
# Shadow Mode Metrics
# =============================================================================
SHADOW_REQUESTS_TOTAL = Counter(
    "sentinel_shadow_requests_total",
    "Total shadow mode comparisons",
    labelnames=["status"]  # success, timeout, error
)

SHADOW_QUALITY_MATCH = Counter(
    "sentinel_shadow_quality_match_total",
    "Shadow comparisons where local quality matched cloud",
    labelnames=["tier"]
)

SHADOW_SIMILARITY_SCORE = Histogram(
    "sentinel_shadow_similarity_score",
    "Semantic similarity score between cloud and local outputs",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
)

SHADOW_LATENCY_DIFF_SECONDS = Histogram(
    "sentinel_shadow_latency_diff_seconds",
    "Latency difference (local - cloud) in seconds, negative = local faster",
    buckets=[-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 30]
)

SHADOW_COST_SAVINGS_USD = Counter(
    "sentinel_shadow_cost_savings_potential_usd",
    "Potential cost savings if routed to local (from shadow comparisons)"
)


# =============================================================================
# Helper Functions
# =============================================================================
def get_metrics() -> bytes:
    """Generate current metrics in Prometheus format."""
    return generate_latest()


def get_content_type() -> str:
    """Get the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST


def init_app_info(version: str, env: str) -> None:
    """Initialize application info metric."""
    APP_INFO.info({
        "version": version,
        "environment": env
    })


def record_request(
    route: str,
    backend: str,
    endpoint: str,
    tier: int,
    status: str = "success"
) -> None:
    """Record a completed request."""
    REQUESTS_TOTAL.labels(
        route=route,
        backend=backend,
        endpoint=endpoint,
        tier=str(tier),
        status=status
    ).inc()


def record_latencies(
    backend: str,
    endpoint: str,
    model: str,
    ttft_ms: float | None,
    itl_ms: float | None,
    tpot_ms: float | None,
    total_ms: float
) -> None:
    """Record all latency metrics for a request."""
    labels = {"backend": backend, "endpoint": endpoint, "model": model}
    
    if ttft_ms is not None:
        TTFT_SECONDS.labels(**labels).observe(ttft_ms / 1000)
    
    if itl_ms is not None:
        ITL_SECONDS.labels(**labels).observe(itl_ms / 1000)
    
    if tpot_ms is not None:
        TPOT_SECONDS.labels(**labels).observe(tpot_ms / 1000)
    
    INFERENCE_LATENCY_SECONDS.labels(**labels).observe(total_ms / 1000)


def record_tokens(
    backend: str,
    endpoint: str,
    prompt_tokens: int,
    completion_tokens: int,
    tokens_per_second: float | None = None,
    model: str = ""
) -> None:
    """Record token counts and throughput."""
    TOKENS_TOTAL.labels(backend=backend, endpoint=endpoint, type="prompt").inc(prompt_tokens)
    TOKENS_TOTAL.labels(backend=backend, endpoint=endpoint, type="completion").inc(completion_tokens)
    
    if tokens_per_second is not None:
        TOKENS_PER_SECOND.labels(backend=backend, endpoint=endpoint, model=model).set(tokens_per_second)


def record_classification(
    tier: int,
    tier_label: str,
    entities: list[dict],
    latency_ms: float,
    detection_method: str = "regex"
) -> None:
    """Record classification metrics."""
    CLASSIFICATIONS_TOTAL.labels(tier=str(tier), tier_label=tier_label).inc()
    CLASSIFICATION_LATENCY_SECONDS.labels(detection_method=detection_method).observe(latency_ms / 1000)
    
    for entity in entities:
        ENTITIES_DETECTED_TOTAL.labels(
            entity_type=entity.get("entity_type", "unknown"),
            tier=str(entity.get("tier", tier))
        ).inc()
    
    # Update current tier gauge (for dashboard)
    PRIVACY_TIER_DISTRIBUTION.labels(tier_label=tier_label).set(1)


def record_cost(
    backend: str,
    model: str,
    cost_usd: float,
    savings_usd: float = 0.0
) -> None:
    """Record cost metrics."""
    if cost_usd > 0:
        COST_USD_TOTAL.labels(backend=backend, model=model).inc(cost_usd)
    if savings_usd > 0:
        COST_SAVINGS_USD_TOTAL.inc(savings_usd)


def record_routing_latency(latency_ms: float) -> None:
    """Record routing decision latency."""
    ROUTING_LATENCY_SECONDS.observe(latency_ms / 1000)


def set_backend_health(endpoint: str, healthy: bool, is_cloud: bool = False, model: str = "") -> None:
    """Update backend health status."""
    if is_cloud:
        CLOUD_BACKEND_HEALTHY.labels(provider=endpoint).set(1 if healthy else 0)
    else:
        LOCAL_BACKEND_HEALTHY.labels(endpoint=endpoint, model=model).set(1 if healthy else 0)


def record_error(error_type: str, backend: str = "unknown") -> None:
    """Record an error."""
    ERRORS_TOTAL.labels(error_type=error_type, backend=backend).inc()


def record_fallback(from_backend: str, to_backend: str, reason: str) -> None:
    """Record a fallback event."""
    FALLBACK_TOTAL.labels(
        from_backend=from_backend,
        to_backend=to_backend,
        reason=reason
    ).inc()


def record_shadow_result(
    status: str,
    tier: int,
    similarity_score: float | None = None,
    latency_diff_ms: float | None = None,
    cost_savings_usd: float | None = None,
    is_quality_match: bool = False,
) -> None:
    """Record shadow mode comparison results.
    
    Args:
        status: "success", "timeout", or "error"
        tier: Privacy tier of the shadowed request
        similarity_score: Semantic similarity (0.0 to 1.0)
        latency_diff_ms: local_latency - cloud_latency (negative = local faster)
        cost_savings_usd: Potential savings if routed to local
        is_quality_match: Whether local quality matched cloud
    """
    SHADOW_REQUESTS_TOTAL.labels(status=status).inc()
    
    if status == "success":
        if similarity_score is not None:
            SHADOW_SIMILARITY_SCORE.observe(similarity_score)
        
        if latency_diff_ms is not None:
            SHADOW_LATENCY_DIFF_SECONDS.observe(latency_diff_ms / 1000)
        
        if cost_savings_usd is not None and cost_savings_usd > 0:
            SHADOW_COST_SAVINGS_USD.inc(cost_savings_usd)
        
        if is_quality_match:
            SHADOW_QUALITY_MATCH.labels(tier=str(tier)).inc()
