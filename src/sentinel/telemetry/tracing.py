"""OpenTelemetry tracing for inference-sentinel."""

from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


# Global tracer instance
_tracer: trace.Tracer | None = None


def setup_tracing(
    service_name: str = "inference-sentinel",
    service_version: str = "0.1.0",
    otlp_endpoint: str | None = None,
    console_export: bool = False
) -> None:
    """
    Initialize OpenTelemetry tracing.
    
    Args:
        service_name: Name of this service for traces
        service_version: Version of this service
        otlp_endpoint: OTLP exporter endpoint (e.g., "http://tempo:4317")
        console_export: If True, also export spans to console (for debugging)
    """
    global _tracer
    
    # Create resource with service info
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "service.namespace": "inference-sentinel",
        "deployment.environment": "development"
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add OTLP exporter if endpoint configured
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Add console exporter for debugging
    if console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    
    # Set as global tracer provider
    trace.set_tracer_provider(provider)
    
    # Get tracer instance
    _tracer = trace.get_tracer(service_name, service_version)


def get_tracer() -> trace.Tracer:
    """Get the configured tracer instance."""
    global _tracer
    if _tracer is None:
        # Return a no-op tracer if not configured
        return trace.get_tracer("sentinel-noop")
    return _tracer


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None
) -> Generator[Span, None, None]:
    """
    Create a traced span context.
    
    Usage:
        with trace_span("process_request", {"request_id": "123"}) as span:
            # do work
            span.set_attribute("result", "success")
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
        yield span


def trace_inference_request(
    request_id: str,
    route: str,
    backend: str,
    tier: int
) -> Span:
    """Start a span for an inference request."""
    tracer = get_tracer()
    span = tracer.start_span(
        "inference_request",
        attributes={
            "request.id": request_id,
            "request.route": route,
            "request.backend": backend,
            "privacy.tier": tier
        }
    )
    return span


def trace_classification(parent_span: Span | None = None) -> Span:
    """Start a span for classification."""
    tracer = get_tracer()
    context = trace.set_span_in_context(parent_span) if parent_span else None
    span = tracer.start_span("classify_content", context=context)
    return span


def trace_routing(parent_span: Span | None = None) -> Span:
    """Start a span for routing decision."""
    tracer = get_tracer()
    context = trace.set_span_in_context(parent_span) if parent_span else None
    span = tracer.start_span("route_request", context=context)
    return span


def trace_backend_call(
    backend: str,
    endpoint: str,
    model: str,
    parent_span: Span | None = None
) -> Span:
    """Start a span for a backend inference call."""
    tracer = get_tracer()
    context = trace.set_span_in_context(parent_span) if parent_span else None
    span = tracer.start_span(
        f"backend_{backend}",
        context=context,
        attributes={
            "backend.type": backend,
            "backend.endpoint": endpoint,
            "backend.model": model
        }
    )
    return span


def end_span_success(span: Span, attributes: dict[str, Any] | None = None) -> None:
    """End a span with success status."""
    if attributes:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    span.set_status(Status(StatusCode.OK))
    span.end()


def end_span_error(span: Span, error: Exception | str) -> None:
    """End a span with error status."""
    error_message = str(error)
    span.set_attribute("error.message", error_message)
    span.set_status(Status(StatusCode.ERROR, error_message))
    if isinstance(error, Exception):
        span.record_exception(error)
    span.end()


# =============================================================================
# Decorators for common tracing patterns
# =============================================================================

def traced(name: str | None = None):
    """
    Decorator to trace a function.
    
    Usage:
        @traced("my_operation")
        def my_function():
            pass
    """
    def decorator(func):
        span_name = name or func.__name__
        
        async def async_wrapper(*args, **kwargs):
            with trace_span(span_name) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        def sync_wrapper(*args, **kwargs):
            with trace_span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# Trace Context Propagation
# =============================================================================

propagator = TraceContextTextMapPropagator()


def extract_trace_context(headers: dict[str, str]) -> trace.Context:
    """Extract trace context from HTTP headers."""
    return propagator.extract(carrier=headers)


def inject_trace_context(headers: dict[str, str]) -> dict[str, str]:
    """Inject current trace context into HTTP headers."""
    propagator.inject(carrier=headers)
    return headers
