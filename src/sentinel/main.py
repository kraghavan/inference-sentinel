"""Main FastAPI application for inference-sentinel."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from sentinel.api.routes import router, set_backend_manager, set_shadow_runner
from sentinel.backends import BackendManager, AnthropicBackend, GoogleBackend
from sentinel.config import LocalBackendsConfig, LocalEndpoint, get_settings
from sentinel.classification import configure_hybrid_classifier
from sentinel.controller import (
    ClosedLoopController,
    ControllerConfig,
    set_controller,
)
from sentinel.shadow import configure_shadow, get_shadow_runner
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

    # =========================================================================
    # Initialize Classification (Hybrid = Regex + optional NER)
    # =========================================================================
    ner_config = settings.ner
    hybrid_classifier = configure_hybrid_classifier(
        ner_enabled=ner_config.enabled,
        ner_model=ner_config.model,
        ner_device=ner_config.device,
        ner_confidence_threshold=ner_config.confidence_threshold,
        ner_threshold_tier=ner_config.skip_if_regex_tier_gte,
    )
    
    if ner_config.enabled:
        await hybrid_classifier.initialize()
        logger.info(
            "Hybrid classifier initialized with NER",
            ner_model=ner_config.model,
            ner_device=ner_config.device,
        )
    else:
        logger.info("Hybrid classifier initialized (regex only, NER disabled)")

    # =========================================================================
    # Initialize Shadow Mode
    # =========================================================================
    shadow_config = settings.shadow
    shadow_runner = configure_shadow(
        enabled=shadow_config.enabled,
        shadow_tiers=shadow_config.shadow_tiers,
        sample_rate=shadow_config.sample_rate,
        similarity_enabled=shadow_config.similarity_enabled,
        similarity_model=shadow_config.similarity_model,
        similarity_device=shadow_config.similarity_device,
        local_timeout_seconds=shadow_config.local_timeout_seconds,
        store_responses=shadow_config.store_responses,
    )
    
    if shadow_config.enabled:
        await shadow_runner.initialize()
        logger.info(
            "Shadow mode initialized",
            shadow_tiers=shadow_config.shadow_tiers,
            sample_rate=shadow_config.sample_rate,
            similarity_enabled=shadow_config.similarity_enabled,
        )
    else:
        logger.info("Shadow mode disabled")
    
    set_shadow_runner(shadow_runner)

    # =========================================================================
    # Initialize Closed-Loop Controller
    # =========================================================================
    controller_config = ControllerConfig(
        enabled=settings.controller.enabled if hasattr(settings, 'controller') else False,
        mode=getattr(settings.controller, 'mode', 'observe') if hasattr(settings, 'controller') else 'observe',
        evaluation_interval_seconds=getattr(settings.controller, 'evaluation_interval_seconds', 60) if hasattr(settings, 'controller') else 60,
        window_seconds=getattr(settings.controller, 'window_seconds', 300) if hasattr(settings, 'controller') else 300,
    )
    
    controller = ClosedLoopController(controller_config)
    controller.set_shadow_runner(shadow_runner)
    set_controller(controller)
    
    if controller_config.enabled:
        await controller.start()
        logger.info(
            "Closed-loop controller started",
            mode=controller_config.mode,
            evaluation_interval=controller_config.evaluation_interval_seconds,
            window_seconds=controller_config.window_seconds,
        )
    else:
        logger.info("Closed-loop controller disabled")

    # =========================================================================
    # Initialize Session Manager (One-Way Trapdoor)
    # =========================================================================
    from sentinel.session import configure_session_manager
    
    session_config = settings.session
    session_manager = configure_session_manager(
        enabled=session_config.enabled,
        ttl_seconds=session_config.ttl_seconds,
        max_sessions=session_config.max_sessions,
        lock_threshold_tier=session_config.lock_threshold_tier,
        buffer_max_turns=session_config.buffer_size,
        buffer_max_chars=session_config.buffer_size * 800,  # ~200 tokens per turn
    )
    
    if session_config.enabled:
        logger.info(
            "Session manager initialized",
            ttl_seconds=session_config.ttl_seconds,
            lock_threshold_tier=session_config.lock_threshold_tier,
            buffer_size=session_config.buffer_size,
            capability_guardrail=session_config.capability_guardrail,
        )
    else:
        logger.info("Session management disabled")

    # =========================================================================
    # Initialize Backend Manager
    # =========================================================================
    local_config = settings.local
    cloud_selection = settings.cloud_selection

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

    # Initialize backend manager with cloud selection strategy
    backend_manager = BackendManager(
        config=local_config,
        cloud_selection_strategy=cloud_selection.strategy,
        cloud_primary=cloud_selection.primary,
        cloud_fallback=cloud_selection.fallback,
    )
    await backend_manager.initialize()
    
    logger.info(
        "Backend manager initialized",
        cloud_selection_strategy=cloud_selection.strategy,
        cloud_primary=cloud_selection.primary,
        cloud_fallback=cloud_selection.fallback,
    )
    
    # Update health metrics for local backends
    for endpoint in local_config.endpoints:
        if endpoint.enabled:
            healthy = backend_manager._health_status.get(endpoint.name, False)
            set_backend_health(endpoint.name, healthy, is_cloud=False, model=endpoint.model)

    # =========================================================================
    # Initialize Cloud Backends
    # =========================================================================
    cloud_config = settings.cloud
    cloud_selection = settings.cloud_selection
    
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

    if backend_manager.has_cloud_backends:
        await backend_manager.initialize_cloud_backends()
        logger.info(
            "Cloud backends initialized",
            backends=list(backend_manager._cloud_backends.keys()),
            primary=cloud_selection.primary,
            fallback=cloud_selection.fallback,
        )
    else:
        logger.warning(
            "No cloud backends configured - all requests will route locally. "
            "Set ANTHROPIC_API_KEY or GOOGLE_API_KEY to enable cloud routing."
        )

    set_backend_manager(backend_manager)

    # =========================================================================
    # Start Background Tasks
    # =========================================================================
    # Create endpoint name -> model mapping for health metrics
    endpoint_models = {ep.name: ep.model for ep in local_config.endpoints if ep.enabled}
    
    async def health_check_loop() -> None:
        while True:
            await asyncio.sleep(local_config.health_check_interval_seconds)
            health_status = await backend_manager.refresh_health()
            for endpoint_name, healthy in health_status.items():
                is_cloud = endpoint_name in backend_manager._cloud_backends
                model = endpoint_models.get(endpoint_name, "") if not is_cloud else ""
                set_backend_health(endpoint_name, healthy, is_cloud=is_cloud, model=model)

    health_task = asyncio.create_task(health_check_loop())

    logger.info("inference-sentinel started successfully")

    yield

    # =========================================================================
    # Cleanup
    # =========================================================================
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass

    # Stop controller
    await controller.stop()

    # Close shadow runner
    await shadow_runner.close()
    
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
