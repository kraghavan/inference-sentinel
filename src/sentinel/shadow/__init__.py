"""Shadow mode module for A/B comparison between local and cloud models.

This module provides:
- ShadowRunner: Runs background local inference for cloud-routed requests
- SimilarityScorer: Compares outputs using semantic embeddings
- ShadowResult: Comparison results with quality/latency/cost metrics

Usage:
    from sentinel.shadow import (
        configure_shadow,
        get_shadow_runner,
        ShadowConfig,
    )
    
    # Enable shadow mode
    runner = configure_shadow(
        enabled=True,
        shadow_tiers=[0, 1],  # Only shadow safe tiers
        sample_rate=0.5,       # Shadow 50% of eligible requests
        similarity_enabled=True,
    )
    
    # In request handler
    if runner.should_shadow(privacy_tier):
        runner.run_shadow(
            request_id=request_id,
            messages=messages,
            cloud_result=result,
            ...
        )
"""

from sentinel.shadow.similarity import (
    SimilarityScorer,
    SimilarityResult,
    get_similarity_scorer,
    configure_similarity,
    compute_similarity,
)

from sentinel.shadow.shadow_runner import (
    ShadowRunner,
    ShadowResult,
    ShadowConfig,
    get_shadow_runner,
    configure_shadow,
)

__all__ = [
    # Similarity
    "SimilarityScorer",
    "SimilarityResult",
    "get_similarity_scorer",
    "configure_similarity",
    "compute_similarity",
    # Shadow
    "ShadowRunner",
    "ShadowResult",
    "ShadowConfig",
    "get_shadow_runner",
    "configure_shadow",
]
