"""API routes for inference-sentinel."""

import time
import uuid
from typing import Annotated

import structlog
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
from sentinel.classification import classify_messages, ClassificationResult

logger = structlog.get_logger()

router = APIRouter()


# Dependency to get backend manager (will be set during app startup)
_backend_manager: BackendManager | None = None


def get_backend_manager() -> BackendManager:
    """Dependency to get the backend manager."""
    if _backend_manager is None:
        raise HTTPException(status_code=503, detail="Backend manager not initialized")
    return _backend_manager


def set_backend_manager(manager: BackendManager) -> None:
    """Set the backend manager instance."""
    global _backend_manager
    _backend_manager = manager


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

    logger.info(
        "Classification result",
        request_id=request_id,
        tier=classification.tier,
        tier_label=classification.tier_label,
        entity_types=classification.entity_types,
        latency_ms=round(classification_latency_ms, 2),
    )

    # Routing decision (Phase 1: always local, but log what we would do)
    routing_start = time.perf_counter()
    # TODO Phase 2: Implement actual routing based on classification
    # For now, just determine what route WOULD be taken
    intended_route = "local" if classification.requires_local else "cloud"
    if request.routing_override:
        intended_route = request.routing_override
    routing_latency_ms = (time.perf_counter() - routing_start) * 1000

    # Generate response
    inference_start = time.perf_counter()
    result, backend = await manager.generate(
        messages=messages,
        model=request.model if request.model not in ("auto", "local", "cloud") else None,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    inference_latency_ms = (time.perf_counter() - inference_start) * 1000

    if result.error:
        logger.error(
            "Inference failed",
            request_id=request_id,
            error=result.error,
        )
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

    logger.info(
        "Inference completed",
        request_id=request_id,
        endpoint=backend.endpoint_name if backend else None,
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
            route="local",  # Phase 1: still always local, but classification is active
            backend="ollama",
            endpoint=backend.endpoint_name if backend else None,
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
            cost_usd=0.0,  # Local inference is free
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
