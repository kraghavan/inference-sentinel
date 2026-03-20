"""Structured logging configuration for inference-sentinel."""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import EventDict, Processor


def add_log_level(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add log level to event dict."""
    event_dict["level"] = method_name
    return event_dict


def add_timestamp(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add ISO timestamp."""
    from datetime import datetime, timezone
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    log_file: str | None = None
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_logs: If True, output JSON format (for Loki). If False, console format.
        log_file: Optional file path for logging
    """
    # Shared processors
    shared_processors: list[Processor] = [
        structlog.stdlib.add_logger_name,
        add_log_level,
        add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_logs:
        # JSON format for production/Loki
        processors: list[Processor] = shared_processors + [
            structlog.processors.JSONRenderer()
        ]
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# =============================================================================
# Logging Helpers for Common Events
# =============================================================================

class InferenceLogger:
    """Helper for logging inference-related events."""
    
    def __init__(self, logger_name: str = "sentinel.inference"):
        self.logger = get_logger(logger_name)
    
    def request_started(
        self,
        request_id: str,
        route: str,
        backend: str,
        tier: int,
        tier_label: str,
        entities: list[str]
    ) -> None:
        """Log when an inference request starts."""
        self.logger.info(
            "Inference request started",
            request_id=request_id,
            route=route,
            backend=backend,
            privacy_tier=tier,
            privacy_tier_label=tier_label,
            entities_detected=entities
        )
    
    def request_completed(
        self,
        request_id: str,
        backend: str,
        model: str,
        total_tokens: int,
        latency_ms: float,
        cost_usd: float,
        tokens_per_second: float | None = None
    ) -> None:
        """Log when an inference request completes."""
        self.logger.info(
            "Inference request completed",
            request_id=request_id,
            backend=backend,
            model=model,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost_usd, 6),
            tokens_per_second=round(tokens_per_second, 2) if tokens_per_second else None
        )
    
    def request_failed(
        self,
        request_id: str,
        backend: str,
        error: str,
        error_type: str
    ) -> None:
        """Log when an inference request fails."""
        self.logger.error(
            "Inference request failed",
            request_id=request_id,
            backend=backend,
            error=error,
            error_type=error_type
        )
    
    def fallback_triggered(
        self,
        request_id: str,
        from_backend: str,
        to_backend: str,
        reason: str
    ) -> None:
        """Log when a fallback is triggered."""
        self.logger.warning(
            "Fallback triggered",
            request_id=request_id,
            from_backend=from_backend,
            to_backend=to_backend,
            reason=reason
        )


class ClassificationLogger:
    """Helper for logging classification events."""
    
    def __init__(self, logger_name: str = "sentinel.classification"):
        self.logger = get_logger(logger_name)
    
    def classified(
        self,
        tier: int,
        tier_label: str,
        entity_count: int,
        entity_types: list[str],
        latency_ms: float
    ) -> None:
        """Log a classification result."""
        self.logger.info(
            "Content classified",
            privacy_tier=tier,
            privacy_tier_label=tier_label,
            entity_count=entity_count,
            entity_types=entity_types,
            latency_ms=round(latency_ms, 4)
        )
    
    def sensitive_content_detected(
        self,
        tier: int,
        tier_label: str,
        entities: list[dict]
    ) -> None:
        """Log when sensitive content is detected (tier >= 2)."""
        if tier >= 2:
            # Don't log the actual content, just metadata
            self.logger.warning(
                "Sensitive content detected",
                privacy_tier=tier,
                privacy_tier_label=tier_label,
                entity_types=[e.get("entity_type") for e in entities],
                action="routing_to_local"
            )


class RoutingLogger:
    """Helper for logging routing decisions."""
    
    def __init__(self, logger_name: str = "sentinel.routing"):
        self.logger = get_logger(logger_name)
    
    def routed(
        self,
        route: str,
        backend: str,
        endpoint: str,
        tier: int,
        override_applied: bool = False
    ) -> None:
        """Log a routing decision."""
        self.logger.info(
            "Request routed",
            route=route,
            backend=backend,
            endpoint=endpoint,
            privacy_tier=tier,
            override_applied=override_applied
        )
    
    def override_denied(
        self,
        tier: int,
        tier_label: str,
        requested_route: str
    ) -> None:
        """Log when a route override is denied."""
        self.logger.warning(
            "Route override denied",
            privacy_tier=tier,
            privacy_tier_label=tier_label,
            requested_route=requested_route,
            reason="RESTRICTED tier cannot be overridden"
        )


class BackendLogger:
    """Helper for logging backend events."""
    
    def __init__(self, logger_name: str = "sentinel.backends"):
        self.logger = get_logger(logger_name)
    
    def health_check(
        self,
        endpoint: str,
        healthy: bool,
        latency_ms: float | None = None,
        error: str | None = None
    ) -> None:
        """Log a backend health check result."""
        if healthy:
            self.logger.debug(
                "Backend health check passed",
                endpoint=endpoint,
                latency_ms=round(latency_ms, 2) if latency_ms else None
            )
        else:
            self.logger.warning(
                "Backend health check failed",
                endpoint=endpoint,
                error=error
            )
    
    def backend_initialized(
        self,
        backend_type: str,
        endpoint: str,
        model: str | None = None
    ) -> None:
        """Log when a backend is initialized."""
        self.logger.info(
            "Backend initialized",
            backend_type=backend_type,
            endpoint=endpoint,
            model=model
        )
