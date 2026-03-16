"""Main FastAPI application for inference-sentinel."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sentinel.api import router, set_backend_manager
from sentinel.backends import BackendManager
from sentinel.config import LocalBackendsConfig, LocalEndpoint, get_settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    settings = get_settings()

    logger.info(
        "Starting inference-sentinel",
        env=settings.env,
        debug=settings.debug,
    )

    # Build local backends config
    # For Phase 0, we configure endpoints from environment or defaults
    local_config = settings.local

    # If no endpoints configured, add a default local one
    if not local_config.endpoints:
        local_config = LocalBackendsConfig(
            endpoints=[
                LocalEndpoint(
                    name="local",
                    host="localhost",
                    port=11434,
                    model="gemma3:4b",  # Using the model you have
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
    set_backend_manager(backend_manager)

    # Start background health check task
    async def health_check_loop() -> None:
        while True:
            await asyncio.sleep(local_config.health_check_interval_seconds)
            await backend_manager.refresh_health()

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
