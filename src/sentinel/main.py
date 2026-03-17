"""Main FastAPI application for inference-sentinel."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from sentinel.api import router, set_backend_manager
from sentinel.backends import BackendManager, AnthropicBackend, GoogleBackend
from sentinel.config import LocalBackendsConfig, LocalEndpoint, get_settings
from sentinel.telemetry import (
    setup_logging,
    setup_tracing,
    get_logger,
    get_metrics,
    get_content_type,
    init_app_info,
    set_backend_health,
)

# Initialize logging first
settings = get_settings()
setup_logging(
    log_level=settings.telemetry.log_level,
    json_logs=True,  # JSON for Loki
)

logger = get_logger("sentinel.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    settings = get_settings()

    # Initialize tracing
    if settings.telemetry.otlp_endpoint:
        setup_tracing(
            service_name=settings.telemetry.service_name,
            service_version="0.1.0",
            otlp_endpoint=settings.telemetry.otlp_endpoint,
        )
        logger.info(
            "OpenTelemetry tracing initialized",
            otlp_endpoint=settings.telemetry.otlp_endpoint
        )
    
    # Initialize app info metric
    init_app_info(version="0.1.0", env=settings.env)

    logger.info(
        "Starting inference-sentinel",
        env=settings.env,
        debug=settings.debug,
    )

    # Build local backends config
    local_config = settings.local

    # If no endpoints configured, add a default local one
    if not local_config.endpoints:
        local_config = LocalBackendsConfig(
            endpoints=[
                LocalEndpoint(
                    name="local",
                    host="localhost",
                    port=11434,
                    model="gemma3:4b",
                    priority=1,
                ),
            ],
            selection_strategy="priority",
            failover_enabled=True,
            timeout_seconds=120.0,
        )

    # Initialize backend manager
    backend_manager = BackendManager(local_config)
    await backend_manager.initialize()
    
    # Update health metrics for local backends (initial status already checked in initialize)
    for endpoint in local_config.endpoints:
        if endpoint.enabled:
            healthy = backend_manager._health_status.get(endpoint.name, False)
            set_backend_health(endpoint.name, healthy, is_cloud=False)

    # Initialize cloud backends if API keys are provided
    cloud_config = settings.cloud
    
    if cloud_config.anthropic_api_key:
        logger.info("Initializing Anthropic backend")
        anthropic_backend = AnthropicBackend(
            api_key=cloud_config.anthropic_api_key,
            model=cloud_config.anthropic_model,
            timeout=cloud_config.timeout_seconds,
        )
        backend_manager.add_cloud_backend("anthropic", anthropic_backend)
        set_backend_health("anthropic", True, is_cloud=True)
    
    if cloud_config.google_api_key:
        logger.info("Initializing Google backend")
        google_backend = GoogleBackend(
            api_key=cloud_config.google_api_key,
            model=cloud_config.google_model,
            timeout=cloud_config.timeout_seconds,
        )
        backend_manager.add_cloud_backend("google", google_backend)
        set_backend_health("google", True, is_cloud=True)

    # Initialize cloud backends
    if backend_manager.has_cloud_backends:
        await backend_manager.initialize_cloud_backends()
        logger.info(
            "Cloud backends initialized",
            backends=list(backend_manager._cloud_backends.keys()),
        )
    else:
        logger.warning(
            "No cloud backends configured - all requests will route locally. "
            "Set ANTHROPIC_API_KEY or GOOGLE_API_KEY to enable cloud routing."
        )

    set_backend_manager(backend_manager)

    # Start background health check task
    async def health_check_loop() -> None:
        while True:
            await asyncio.sleep(local_config.health_check_interval_seconds)
            health_status = await backend_manager.refresh_health()
            # Update metrics
            for endpoint_name, healthy in health_status.items():
                set_backend_health(endpoint_name, healthy, is_cloud=False)

    health_task = asyncio.create_task(health_check_loop())

    logger.info("inference-sentinel started successfully")

    yield

    # Cleanup
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass

    await backend_manager.close()
    logger.info("inference-sentinel shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="inference-sentinel",
        description="Privacy-aware LLM routing gateway with production-grade observability",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)
    
    # Metrics endpoint (for Prometheus scraping)
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(
            content=get_metrics(),
            media_type=get_content_type()
        )

    return app


# Create app instance
app = create_app()


def run() -> None:
    """Run the application using uvicorn."""
    settings = get_settings()
    uvicorn.run(
        "sentinel.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.telemetry.log_level.lower(),
    )


if __name__ == "__main__":
    run()
