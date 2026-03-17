"""API routes for inference-sentinel."""

import time
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from sentinel.api.schemas import (
    Choice,
    ErrorResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    Message,
    SentinelMetadata,
    Usage,
)
from sentinel.backends import BackendManager
from sentinel.classification import (
    classify_messages,
    ClassificationResult,
    get_hybrid_classifier,
    classify_messages_hybrid,
)
from sentinel.controller import (
    get_controller,
    ControllerConfig,
)
from sentinel.routing import route as route_request, RoutingDecision
from sentinel.shadow import ShadowRunner, get_shadow_runner
from sentinel.telemetry import (
    get_logger,
    record_request,
    record_latencies,
    record_tokens,
    record_classification,
    record_cost,
    record_routing_latency,
    record_error,
    record_fallback,
    REQUESTS_IN_PROGRESS,
)

logger = get_logger("sentinel.api")

router = APIRouter()


# Dependency to get backend manager (will be set during app startup)
_backend_manager: BackendManager | None = None
_shadow_runner: ShadowRunner | None = None


def get_backend_manager() -> BackendManager:
    """Dependency to get the backend manager."""
    if _backend_manager is None:
        raise HTTPException(status_code=503, detail="Backend manager not initialized")
    return _backend_manager


def set_backend_manager(manager: BackendManager) -> None:
    """Set the backend manager instance."""
    global _backend_manager
    _backend_manager = manager


def set_shadow_runner(runner: ShadowRunner) -> None:
    """Set the shadow runner instance."""
    global _shadow_runner
    _shadow_runner = runner


@router.get("/health", response_model=HealthResponse)
async def health_check(
    manager: Annotated[BackendManager, Depends(get_backend_manager)],
) -> HealthResponse:
    """Check the health of the service and all backends."""
    health_status = manager.health_status
    any_healthy = any(health_status.values())

    return HealthResponse(
        status="healthy" if any_healthy else "unhealthy",
        version="0.1.0",
        backends=health_status,
    )


@router.post(
    "/v1/inference",
    response_model=InferenceResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def inference(
    request: InferenceRequest,
    manager: Annotated[BackendManager, Depends(get_backend_manager)],
) -> InferenceResponse:
    """Run inference on the best available backend."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    start_time = time.perf_counter()

    logger.info(
        "Inference request received",
        request_id=request_id,
        model=request.model,
        routing_override=request.routing_override,
        message_count=len(request.messages),
    )

    # Convert messages to dict format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Phase 1: Classify the request for privacy-sensitive content
    classification_start = time.perf_counter()
    classification = classify_messages(messages)
    classification_latency_ms = (time.perf_counter() - classification_start) * 1000
    
    # Record classification metrics
    record_classification(
        tier=classification.tier,
        tier_label=classification.tier_label,
        entities=[{"entity_type": e.entity_type, "tier": e.tier} for e in classification.entities_detected],
        latency_ms=classification_latency_ms,
        detection_method=classification.detection_method,
    )

    logger.info(
        "Classification result",
        request_id=request_id,
        tier=classification.tier,
        tier_label=classification.tier_label,
        entity_types=classification.entity_types,
        latency_ms=round(classification_latency_ms, 2),
    )

    # Phase 2: Route based on classification
    routing_start = time.perf_counter()
    override: Literal["local", "cloud"] | None = None
    if request.routing_override in ("local", "cloud"):
        override = request.routing_override  # type: ignore
    
    routing_decision = route_request(classification, override)
    routing_latency_ms = (time.perf_counter() - routing_start) * 1000
    
    # Record routing metrics
    record_routing_latency(routing_latency_ms)

    logger.info(
        "Routing decision",
        request_id=request_id,
        route=routing_decision.route,
        reason=routing_decision.reason,
        override_applied=routing_decision.override_applied,
    )

    # Generate response using routed backend
    inference_start = time.perf_counter()
    
    # Track in-progress requests
    backend_type = "local" if routing_decision.route == "local" else "cloud"
    REQUESTS_IN_PROGRESS.labels(backend=backend_type).inc()
    
    try:
        # Determine if we can actually route to cloud
        actual_route = routing_decision.route
        if routing_decision.route == "cloud" and not manager.has_healthy_cloud_backends:
            logger.warning(
                "Cloud route requested but no cloud backends available, falling back to local",
                request_id=request_id,
            )
            actual_route = "local"
            record_fallback("cloud", "local", "no_healthy_backends")

        if actual_route == "cloud":
            result, backend, final_route = await manager.generate_routed(
                messages=messages,
                route="cloud",
                model=request.model if request.model not in ("auto", "local", "cloud") else None,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        else:
            result, backend = await manager.generate(
                messages=messages,
                model=request.model if request.model not in ("auto", "local", "cloud") else None,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            final_route = "local"
    finally:
        REQUESTS_IN_PROGRESS.labels(backend=backend_type).dec()
    
    inference_latency_ms = (time.perf_counter() - inference_start) * 1000

    if result.error:
        logger.error(
            "Inference failed",
            request_id=request_id,
            error=result.error,
        )
        record_error("inference_failed", backend.endpoint_name if backend else "unknown")
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="Inference failed",
                detail=result.error,
                request_id=request_id,
            ).model_dump(),
        )

    # Calculate metrics
    itl_p50 = None
    itl_p95 = None
    tpot_ms = None
    tokens_per_second = None

    if result.itl_values_ms and len(result.itl_values_ms) > 0:
        sorted_itl = sorted(result.itl_values_ms)
        itl_p50 = sorted_itl[len(sorted_itl) // 2]
        itl_p95 = sorted_itl[int(len(sorted_itl) * 0.95)] if len(sorted_itl) > 1 else itl_p50

    if result.completion_tokens > 0 and result.ttft_ms is not None:
        decode_time_ms = result.total_latency_ms - result.ttft_ms
        if decode_time_ms > 0:
            tpot_ms = decode_time_ms / result.completion_tokens
            tokens_per_second = (result.completion_tokens / decode_time_ms) * 1000

    total_latency_ms = (time.perf_counter() - start_time) * 1000

    # Determine backend type for response
    endpoint_name = backend.endpoint_name if backend else "unknown"
    backend_type_str = "ollama" if final_route == "local" else endpoint_name

    # Record all telemetry metrics
    record_request(
        route=final_route,
        backend=backend_type_str,
        endpoint=endpoint_name,
        tier=classification.tier,
        status="success",
    )
    
    record_latencies(
        backend=backend_type_str,
        endpoint=endpoint_name,
        model=result.model,
        ttft_ms=result.ttft_ms,
        itl_ms=itl_p50,
        tpot_ms=tpot_ms,
        total_ms=inference_latency_ms,
    )
    
    record_tokens(
        backend=backend_type_str,
        endpoint=endpoint_name,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        tokens_per_second=tokens_per_second,
        model=result.model,
    )
    
    if result.cost_usd:
        record_cost(
            backend=backend_type_str,
            model=result.model,
            cost_usd=result.cost_usd,
            savings_usd=0.0,  # TODO: Calculate vs cloud baseline
        )

    # =========================================================================
    # Shadow Mode: Run local inference in background for cloud requests
    # =========================================================================
    if (
        _shadow_runner is not None
        and final_route == "cloud"
        and _shadow_runner.should_shadow(classification.tier)
    ):
        logger.debug(
            "Triggering shadow mode",
            request_id=request_id,
            tier=classification.tier,
        )
        await _shadow_runner.run_shadow(
            request_id=request_id,
            messages=messages,
            cloud_result=result,
            cloud_backend_name=endpoint_name,
            cloud_latency_ms=inference_latency_ms,
            privacy_tier=classification.tier,
            backend_manager=manager,
        )

    logger.info(
        "Inference completed",
        request_id=request_id,
        endpoint=endpoint_name,
        model=result.model,
        ttft_ms=result.ttft_ms,
        total_latency_ms=total_latency_ms,
        tokens=result.completion_tokens,
    )

    return InferenceResponse(
        id=request_id,
        model=result.model,
        choices=[
            Choice(
                message=Message(role="assistant", content=result.content),
                finish_reason=result.finish_reason,  # type: ignore
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
        sentinel=SentinelMetadata(
            route=final_route,
            backend=backend_type_str,
            endpoint=endpoint_name,
            model=result.model,
            privacy_tier=classification.tier,
            privacy_tier_label=classification.tier_label,
            entities_detected=classification.entity_types,
            classification_latency_ms=classification_latency_ms,
            routing_latency_ms=routing_latency_ms,
            inference_latency_ms=inference_latency_ms,
            ttft_ms=result.ttft_ms,
            itl_p50_ms=itl_p50,
            itl_p95_ms=itl_p95,
            tpot_ms=tpot_ms,
            tokens_per_second=tokens_per_second,
            cost_usd=result.cost_usd if result.cost_usd else 0.0,
            cost_savings_usd=0.0,  # TODO: Calculate vs cloud baseline
        ),
    )


@router.get("/v1/models")
async def list_models(
    manager: Annotated[BackendManager, Depends(get_backend_manager)],
) -> dict:
    """List all available models across all endpoints."""
    models = await manager.list_all_models()
    return {"models": models}


# ============== CLASSIFICATION ENDPOINT ==============

class ClassifyRequest(BaseModel):
    """Request body for classification endpoint."""

    text: str | None = Field(default=None, description="Text to classify")
    messages: list[Message] | None = Field(
        default=None, description="Messages to classify (alternative to text)"
    )


class DetectedEntityResponse(BaseModel):
    """A detected entity in the response."""

    entity_type: str
    tier: int
    start_pos: int
    end_pos: int
    confidence: float


class ClassifyResponse(BaseModel):
    """Response body for classification endpoint."""

    tier: int = Field(description="Highest privacy tier detected (0-3)")
    tier_label: str = Field(description="Tier label (PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED)")
    is_sensitive: bool = Field(description="True if tier > 0")
    requires_local: bool = Field(description="True if tier >= 2 (should route locally)")
    entities_detected: list[DetectedEntityResponse] = Field(default_factory=list)
    entity_types: list[str] = Field(default_factory=list)
    entity_count: int = Field(default=0)
    detection_method: str = Field(default="regex")
    detection_latency_ms: float = Field(default=0.0)


@router.post("/v1/classify", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest) -> ClassifyResponse:
    """Classify text or messages for privacy-sensitive content.

    This endpoint detects PII, credentials, and other sensitive data
    and returns the privacy tier classification.

    Tiers:
    - 0 (PUBLIC): No sensitive content
    - 1 (INTERNAL): Business context, non-regulated
    - 2 (CONFIDENTIAL): PII, credentials - should route locally
    - 3 (RESTRICTED): Regulated data - must route locally
    """
    if request.text is None and request.messages is None:
        raise HTTPException(
            status_code=400,
            detail="Either 'text' or 'messages' must be provided",
        )

    # Classify based on input type
    if request.messages:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        result = classify_messages(messages)
    else:
        from sentinel.classification import classify
        result = classify(request.text or "")

    logger.info(
        "Classification completed",
        tier=result.tier,
        tier_label=result.tier_label,
        entity_count=result.entity_count,
        latency_ms=round(result.detection_latency_ms, 2),
    )

    return ClassifyResponse(
        tier=result.tier,
        tier_label=result.tier_label,
        is_sensitive=result.is_sensitive,
        requires_local=result.requires_local,
        entities_detected=[
            DetectedEntityResponse(
                entity_type=e.entity_type,
                tier=e.tier,
                start_pos=e.start_pos,
                end_pos=e.end_pos,
                confidence=e.confidence,
            )
            for e in result.entities_detected
        ],
        entity_types=result.entity_types,
        entity_count=result.entity_count,
        detection_method=result.detection_method,
        detection_latency_ms=result.detection_latency_ms,
    )


# ============== ADMIN ENDPOINTS ==============

class ShadowMetricsResponse(BaseModel):
    """Response for shadow mode metrics."""
    
    enabled: bool = Field(description="Whether shadow mode is enabled")
    total_shadows: int = Field(description="Total shadow comparisons run")
    successful_shadows: int = Field(description="Successful shadow comparisons")
    quality_matches: int = Field(description="Comparisons where local matched cloud quality")
    quality_match_rate: float = Field(description="Quality match rate (0.0 to 1.0)")
    total_cost_savings_usd: float = Field(description="Potential cost savings from shadow data")
    pending_tasks: int = Field(description="Currently running shadow tasks")
    stored_results: int = Field(description="Number of stored shadow results")


class ShadowResultResponse(BaseModel):
    """A single shadow comparison result."""
    
    shadow_id: str
    request_id: str
    timestamp: str
    cloud_model: str
    local_model: str
    cloud_latency_ms: float
    local_latency_ms: float
    latency_diff_ms: float
    local_is_faster: bool
    similarity_score: float | None
    is_quality_match: bool
    privacy_tier: int
    cost_savings_usd: float


@router.get("/admin/shadow/metrics", response_model=ShadowMetricsResponse)
async def get_shadow_metrics() -> ShadowMetricsResponse:
    """Get shadow mode metrics and statistics.
    
    Returns aggregate statistics about shadow mode comparisons,
    including quality match rates and cost savings.
    """
    if _shadow_runner is None:
        return ShadowMetricsResponse(
            enabled=False,
            total_shadows=0,
            successful_shadows=0,
            quality_matches=0,
            quality_match_rate=0.0,
            total_cost_savings_usd=0.0,
            pending_tasks=0,
            stored_results=0,
        )
    
    metrics = _shadow_runner.get_metrics()
    
    return ShadowMetricsResponse(
        enabled=_shadow_runner.config.enabled,
        total_shadows=metrics["total_shadows"],
        successful_shadows=metrics["successful_shadows"],
        quality_matches=metrics["quality_matches"],
        quality_match_rate=metrics["quality_match_rate"],
        total_cost_savings_usd=metrics["total_cost_savings_usd"],
        pending_tasks=metrics["pending_tasks"],
        stored_results=metrics["stored_results"],
    )


@router.get("/admin/shadow/results", response_model=list[ShadowResultResponse])
async def get_shadow_results(limit: int = 10) -> list[ShadowResultResponse]:
    """Get recent shadow comparison results.
    
    Returns the most recent shadow comparisons with detailed
    metrics about quality and latency differences.
    
    Args:
        limit: Maximum number of results to return (default: 10, max: 100)
    """
    if _shadow_runner is None:
        return []
    
    limit = min(limit, 100)  # Cap at 100
    results = _shadow_runner.get_recent_results(limit=limit)
    
    return [
        ShadowResultResponse(
            shadow_id=r["shadow_id"],
            request_id=r["request_id"],
            timestamp=r["timestamp"],
            cloud_model=r["cloud_model"],
            local_model=r["local_model"],
            cloud_latency_ms=r["cloud_latency_ms"],
            local_latency_ms=r["local_latency_ms"],
            latency_diff_ms=r["latency_diff_ms"],
            local_is_faster=r["local_is_faster"],
            similarity_score=r["similarity_score"],
            is_quality_match=r["is_quality_match"],
            privacy_tier=r["privacy_tier"],
            cost_savings_usd=r.get("cost_savings_usd", 0.0),
        )
        for r in results
    ]


# =============================================================================
# Controller Admin Endpoints
# =============================================================================

class ControllerStatusResponse(BaseModel):
    """Response model for controller status."""
    
    enabled: bool
    mode: str
    running: bool
    last_evaluation: str | None
    next_evaluation: str | None
    evaluation_interval_seconds: int
    total_evaluations: int
    recommendations: list[dict]
    tier_metrics: dict[int, dict]


class ControllerHistoryResponse(BaseModel):
    """Response model for controller history."""
    
    entries: list[dict]
    count: int


class ReloadResponse(BaseModel):
    """Response model for config reload."""
    
    success: bool
    message: str
    reloaded_components: list[str]


@router.get("/admin/controller/status", response_model=ControllerStatusResponse)
async def get_controller_status() -> ControllerStatusResponse:
    """Get current controller status and recommendations.
    
    Returns the controller's current state including:
    - Whether it's running
    - Current recommendations per tier
    - Aggregated metrics per tier
    - Next evaluation time
    """
    controller = get_controller()
    
    if controller is None:
        return ControllerStatusResponse(
            enabled=False,
            mode="observe",
            running=False,
            last_evaluation=None,
            next_evaluation=None,
            evaluation_interval_seconds=60,
            total_evaluations=0,
            recommendations=[],
            tier_metrics={},
        )
    
    status = controller.get_status()
    return ControllerStatusResponse(
        enabled=status.enabled,
        mode=status.mode,
        running=status.running,
        last_evaluation=status.last_evaluation.isoformat() if status.last_evaluation else None,
        next_evaluation=status.next_evaluation.isoformat() if status.next_evaluation else None,
        evaluation_interval_seconds=status.evaluation_interval_seconds,
        total_evaluations=status.total_evaluations,
        recommendations=[r.to_dict() for r in status.recommendations],
        tier_metrics={k: v.to_dict() for k, v in status.tier_metrics.items()},
    )


@router.get("/admin/controller/history", response_model=ControllerHistoryResponse)
async def get_controller_history(limit: int = 20) -> ControllerHistoryResponse:
    """Get controller recommendation history.
    
    Returns the most recent evaluation results with recommendations
    and metrics. Useful for debugging and trend analysis.
    
    Args:
        limit: Maximum number of entries to return (default: 20, max: 100)
    """
    controller = get_controller()
    
    if controller is None:
        return ControllerHistoryResponse(entries=[], count=0)
    
    limit = min(limit, 100)
    history = controller.get_history(limit=limit)
    
    return ControllerHistoryResponse(
        entries=history,
        count=len(history),
    )


@router.post("/admin/controller/evaluate")
async def force_controller_evaluate() -> dict:
    """Force an immediate controller evaluation.
    
    Triggers the controller to evaluate metrics and generate
    recommendations immediately, without waiting for the next
    scheduled evaluation.
    
    Returns the evaluation results.
    """
    controller = get_controller()
    
    if controller is None:
        raise HTTPException(
            status_code=503,
            detail="Controller not initialized"
        )
    
    result = await controller.force_evaluate()
    return result


@router.post("/admin/reload", response_model=ReloadResponse)
async def reload_config() -> ReloadResponse:
    """Hot-reload configuration from routing.yaml.
    
    Reloads configuration for:
    - Controller thresholds
    - Shadow mode settings
    - Cloud selection strategy
    
    Does NOT reload:
    - Local backend endpoints (requires restart)
    - API keys (requires restart)
    
    Note: This endpoint reads from config/routing.yaml in the
    current working directory.
    """
    import yaml
    from pathlib import Path
    
    reloaded = []
    
    try:
        # Load routing.yaml
        config_path = Path("config/routing.yaml")
        if not config_path.exists():
            return ReloadResponse(
                success=False,
                message="config/routing.yaml not found",
                reloaded_components=[],
            )
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Reload controller config
        controller = get_controller()
        if controller and "controller" in config:
            controller_data = config["controller"]
            new_config = ControllerConfig.from_dict(controller_data)
            controller.update_config(new_config)
            reloaded.append("controller")
            logger.info("Controller config reloaded", config=controller_data)
        
        # Reload shadow config
        shadow_runner = _shadow_runner
        if shadow_runner and "shadow" in config:
            shadow_data = config["shadow"]
            # Update sample rate and tiers
            if "sample_rate" in shadow_data:
                shadow_runner._sample_rate = shadow_data["sample_rate"]
            if "shadow_tiers" in shadow_data:
                shadow_runner._shadow_tiers = set(shadow_data["shadow_tiers"])
            reloaded.append("shadow")
            logger.info("Shadow config reloaded", config=shadow_data)
        
        # Reload cloud selection (if backend manager available)
        if _backend_manager and "cloud" in config:
            cloud_data = config["cloud"]
            if "selection_strategy" in cloud_data:
                _backend_manager._cloud_selection_strategy = cloud_data["selection_strategy"]
            if "primary" in cloud_data:
                _backend_manager._cloud_primary = cloud_data["primary"]
            if "fallback" in cloud_data:
                _backend_manager._cloud_fallback = cloud_data["fallback"]
            reloaded.append("cloud_selection")
            logger.info("Cloud selection config reloaded", config=cloud_data)
        
        return ReloadResponse(
            success=True,
            message=f"Reloaded {len(reloaded)} component(s)",
            reloaded_components=reloaded,
        )
    
    except Exception as e:
        logger.error("Config reload failed", error=str(e))
        return ReloadResponse(
            success=False,
            message=f"Reload failed: {str(e)}",
            reloaded_components=reloaded,
        )
