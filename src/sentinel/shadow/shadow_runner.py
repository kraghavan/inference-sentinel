"""Shadow mode runner for A/B comparison between local and cloud models.

Shadow mode runs local inference in the background for cloud-routed requests,
enabling quality comparison without affecting user experience.

Flow:
1. Request routes to cloud (tier 0-1)
2. Cloud response returned to user immediately
3. Background task runs same request on local
4. Compare outputs using similarity scoring
5. Log results for analysis

This provides data for:
- Proving local quality matches cloud
- Measuring latency differences  
- Calculating cost savings
- Future closed-loop controller
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sentinel.shadow.similarity import (
    SimilarityScorer,
    SimilarityResult,
    get_similarity_scorer,
    configure_similarity,
)
from sentinel.telemetry import get_logger, record_shadow_result

if TYPE_CHECKING:
    from sentinel.backends import BackendManager
    from sentinel.backends.base import InferenceResult

logger = get_logger("sentinel.shadow")


@dataclass
class ShadowResult:
    """Result of a shadow mode comparison."""
    
    # Identifiers
    shadow_id: str
    request_id: str
    timestamp: str
    
    # Responses
    cloud_response: str
    local_response: str
    
    # Models used
    cloud_model: str
    local_model: str
    cloud_backend: str
    local_backend: str
    
    # Latency comparison
    cloud_latency_ms: float
    local_latency_ms: float
    latency_diff_ms: float  # local - cloud (negative = local faster)
    
    # Token comparison
    cloud_tokens: int
    local_tokens: int
    
    # Cost comparison
    cloud_cost_usd: float
    local_cost_usd: float  # Usually 0 for local
    cost_savings_usd: float
    
    # Quality comparison
    similarity: SimilarityResult | None = None
    
    # Metadata
    privacy_tier: int = 0
    messages_hash: str = ""
    error: str | None = None
    
    @property
    def is_quality_match(self) -> bool:
        """Check if local quality matches cloud."""
        if self.similarity is None:
            return False
        return self.similarity.is_quality_match
    
    @property
    def local_is_faster(self) -> bool:
        """Check if local was faster than cloud."""
        return self.latency_diff_ms < 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "shadow_id": self.shadow_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "cloud_model": self.cloud_model,
            "local_model": self.local_model,
            "cloud_latency_ms": round(self.cloud_latency_ms, 2),
            "local_latency_ms": round(self.local_latency_ms, 2),
            "latency_diff_ms": round(self.latency_diff_ms, 2),
            "local_is_faster": self.local_is_faster,
            "cloud_tokens": self.cloud_tokens,
            "local_tokens": self.local_tokens,
            "cloud_cost_usd": round(self.cloud_cost_usd, 6),
            "cost_savings_usd": round(self.cost_savings_usd, 6),
            "similarity_score": self.similarity.similarity_score if self.similarity else None,
            "is_quality_match": self.is_quality_match,
            "privacy_tier": self.privacy_tier,
            "error": self.error,
        }


@dataclass
class ShadowConfig:
    """Configuration for shadow mode."""
    
    enabled: bool = False
    
    # Which tiers to shadow (only safe tiers)
    shadow_tiers: list[int] = field(default_factory=lambda: [0, 1])
    
    # Sampling rate (0.0 to 1.0)
    sample_rate: float = 1.0  # 100% of eligible requests
    
    # Similarity scoring
    similarity_enabled: bool = True
    similarity_model: str = "fast"
    similarity_device: str = "cpu"
    
    # Timeouts
    local_timeout_seconds: float = 60.0
    
    # Storage
    store_responses: bool = False  # Store full responses (memory heavy)
    max_stored_results: int = 1000


class ShadowRunner:
    """Runs shadow mode comparisons between local and cloud models.
    
    Shadow mode is non-blocking - cloud response is returned immediately
    while local inference runs in the background.
    """
    
    def __init__(
        self,
        config: ShadowConfig | None = None,
        similarity_scorer: SimilarityScorer | None = None,
    ):
        self.config = config or ShadowConfig()
        self._similarity = similarity_scorer or get_similarity_scorer()
        
        # Results storage (circular buffer)
        self._results: list[ShadowResult] = []
        self._results_lock = asyncio.Lock()
        
        # Background tasks
        self._pending_tasks: set[asyncio.Task] = set()
        
        # Metrics
        self._total_shadows = 0
        self._successful_shadows = 0
        self._quality_matches = 0
        self._total_cost_savings = 0.0
    
    async def initialize(self) -> None:
        """Initialize shadow runner and dependencies."""
        if self.config.enabled and self.config.similarity_enabled:
            await self._similarity.initialize()
            logger.info(
                "Shadow runner initialized",
                sample_rate=self.config.sample_rate,
                shadow_tiers=self.config.shadow_tiers,
                similarity_enabled=self.config.similarity_enabled,
            )
    
    def should_shadow(self, privacy_tier: int) -> bool:
        """Determine if this request should be shadowed.
        
        Args:
            privacy_tier: Privacy tier of the request
            
        Returns:
            True if request should run shadow mode
        """
        if not self.config.enabled:
            return False
        
        # Only shadow safe tiers
        if privacy_tier not in self.config.shadow_tiers:
            return False
        
        # Apply sampling rate
        if self.config.sample_rate < 1.0:
            import random
            if random.random() > self.config.sample_rate:
                return False
        
        return True
    
    async def run_shadow(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        cloud_result: "InferenceResult",
        cloud_backend_name: str,
        cloud_latency_ms: float,
        privacy_tier: int,
        backend_manager: "BackendManager",
    ) -> None:
        """Run shadow inference on local backend.
        
        This method is fire-and-forget - it schedules the shadow
        task and returns immediately.
        
        Args:
            request_id: Original request ID
            messages: Original messages
            cloud_result: Result from cloud backend
            cloud_backend_name: Name of cloud backend used
            cloud_latency_ms: Cloud inference latency
            privacy_tier: Privacy tier of request
            backend_manager: Backend manager for local inference
        """
        task = asyncio.create_task(
            self._run_shadow_async(
                request_id=request_id,
                messages=messages,
                cloud_result=cloud_result,
                cloud_backend_name=cloud_backend_name,
                cloud_latency_ms=cloud_latency_ms,
                privacy_tier=privacy_tier,
                backend_manager=backend_manager,
            )
        )
        
        # Track task for cleanup
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
    
    async def _run_shadow_async(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        cloud_result: "InferenceResult",
        cloud_backend_name: str,
        cloud_latency_ms: float,
        privacy_tier: int,
        backend_manager: "BackendManager",
    ) -> ShadowResult | None:
        """Internal async shadow execution."""
        shadow_id = f"shadow_{uuid.uuid4().hex[:12]}"
        start_time = time.perf_counter()
        
        self._total_shadows += 1
        
        try:
            # Run local inference
            local_result, local_backend = await asyncio.wait_for(
                backend_manager.generate(
                    messages=messages,
                    max_tokens=1024,  # Match typical cloud limits
                ),
                timeout=self.config.local_timeout_seconds,
            )
            
            local_latency_ms = (time.perf_counter() - start_time) * 1000
            
            if local_result.error:
                logger.warning(
                    "Shadow local inference failed",
                    shadow_id=shadow_id,
                    request_id=request_id,
                    error=local_result.error,
                )
                return None
            
            # Compute similarity if enabled
            similarity = None
            if self.config.similarity_enabled:
                similarity = await self._similarity.compute_similarity(
                    cloud_response=cloud_result.content,
                    local_response=local_result.content,
                )
            
            # Build result
            result = ShadowResult(
                shadow_id=shadow_id,
                request_id=request_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                cloud_response=cloud_result.content if self.config.store_responses else "",
                local_response=local_result.content if self.config.store_responses else "",
                cloud_model=cloud_result.model,
                local_model=local_result.model,
                cloud_backend=cloud_backend_name,
                local_backend=local_backend.endpoint_name if local_backend else "unknown",
                cloud_latency_ms=cloud_latency_ms,
                local_latency_ms=local_latency_ms,
                latency_diff_ms=local_latency_ms - cloud_latency_ms,
                cloud_tokens=cloud_result.prompt_tokens + cloud_result.completion_tokens,
                local_tokens=local_result.prompt_tokens + local_result.completion_tokens,
                cloud_cost_usd=cloud_result.cost_usd or 0.0,
                local_cost_usd=0.0,  # Local is free
                cost_savings_usd=cloud_result.cost_usd or 0.0,
                similarity=similarity,
                privacy_tier=privacy_tier,
            )
            
            # Update metrics
            self._successful_shadows += 1
            self._total_cost_savings += result.cost_savings_usd
            if result.is_quality_match:
                self._quality_matches += 1
            
            # Record Prometheus metrics
            record_shadow_result(
                status="success",
                tier=privacy_tier,
                similarity_score=result.similarity.similarity_score if result.similarity else None,
                latency_diff_ms=result.latency_diff_ms,
                cost_savings_usd=result.cost_savings_usd,
                is_quality_match=result.is_quality_match,
            )
            
            # Store result
            await self._store_result(result)
            
            # Log comparison
            logger.info(
                "Shadow comparison complete",
                **result.to_dict()
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(
                "Shadow local inference timed out",
                shadow_id=shadow_id,
                request_id=request_id,
                timeout=self.config.local_timeout_seconds,
            )
            record_shadow_result(status="timeout", tier=privacy_tier)
            return None
        except Exception as e:
            logger.error(
                "Shadow execution failed",
                shadow_id=shadow_id,
                request_id=request_id,
                error=str(e),
            )
            record_shadow_result(status="error", tier=privacy_tier)
            return None
    
    async def _store_result(self, result: ShadowResult) -> None:
        """Store shadow result (circular buffer)."""
        async with self._results_lock:
            self._results.append(result)
            
            # Trim to max size
            if len(self._results) > self.config.max_stored_results:
                self._results = self._results[-self.config.max_stored_results:]
    
    def get_metrics(self) -> dict:
        """Get shadow mode metrics."""
        quality_rate = (
            self._quality_matches / self._successful_shadows
            if self._successful_shadows > 0 else 0.0
        )
        
        return {
            "total_shadows": self._total_shadows,
            "successful_shadows": self._successful_shadows,
            "quality_matches": self._quality_matches,
            "quality_match_rate": round(quality_rate, 4),
            "total_cost_savings_usd": round(self._total_cost_savings, 4),
            "pending_tasks": len(self._pending_tasks),
            "stored_results": len(self._results),
        }
    
    def get_recent_results(self, limit: int = 10) -> list[dict]:
        """Get recent shadow results."""
        results = self._results[-limit:]
        return [r.to_dict() for r in reversed(results)]
    
    async def close(self) -> None:
        """Clean up pending tasks."""
        if self._pending_tasks:
            logger.info(
                "Cancelling pending shadow tasks",
                count=len(self._pending_tasks)
            )
            for task in self._pending_tasks:
                task.cancel()
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()


# Global singleton
_shadow_runner: ShadowRunner | None = None


def get_shadow_runner() -> ShadowRunner:
    """Get the global shadow runner instance."""
    global _shadow_runner
    if _shadow_runner is None:
        _shadow_runner = ShadowRunner()
    return _shadow_runner


def configure_shadow(
    enabled: bool = False,
    shadow_tiers: list[int] | None = None,
    sample_rate: float = 1.0,
    similarity_enabled: bool = True,
    similarity_model: str = "fast",
    similarity_device: str = "cpu",
    local_timeout_seconds: float = 60.0,
    store_responses: bool = False,
) -> ShadowRunner:
    """Configure the global shadow runner.
    
    Args:
        enabled: Enable shadow mode
        shadow_tiers: Which privacy tiers to shadow (default: [0, 1])
        sample_rate: Fraction of eligible requests to shadow (0.0 to 1.0)
        similarity_enabled: Enable similarity scoring
        similarity_model: Similarity model ("fast", "balanced", "accurate")
        similarity_device: Device for similarity model
        local_timeout_seconds: Timeout for local shadow inference
        store_responses: Store full responses (memory heavy)
        
    Returns:
        Configured ShadowRunner instance
    """
    global _shadow_runner
    
    # Configure similarity scorer
    if similarity_enabled:
        configure_similarity(
            model_name=similarity_model,
            device=similarity_device,
            enabled=True,
        )
    
    config = ShadowConfig(
        enabled=enabled,
        shadow_tiers=shadow_tiers or [0, 1],
        sample_rate=sample_rate,
        similarity_enabled=similarity_enabled,
        similarity_model=similarity_model,
        similarity_device=similarity_device,
        local_timeout_seconds=local_timeout_seconds,
        store_responses=store_responses,
    )
    
    _shadow_runner = ShadowRunner(config=config)
    
    return _shadow_runner
