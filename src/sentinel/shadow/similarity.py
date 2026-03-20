"""Output similarity computation using sentence embeddings.

Compares LLM outputs using semantic similarity to measure quality parity
between local and cloud models.

Uses sentence-transformers for efficient embedding generation.
"""

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from sentinel.telemetry import get_logger

logger = get_logger("sentinel.shadow.similarity")

# Lazy import for numpy (only needed when similarity is actually used)
np = None


def _get_numpy():
    """Lazy load numpy."""
    global np
    if np is None:
        try:
            import numpy
            np = numpy
        except ImportError:
            raise ImportError(
                "numpy is required for similarity scoring. "
                "Install with: pip install inference-sentinel[shadow]"
            )
    return np


@dataclass
class SimilarityResult:
    """Result of similarity comparison."""
    
    similarity_score: float  # 0.0 to 1.0
    interpretation: Literal["high", "medium", "low"]
    cloud_response_length: int
    local_response_length: int
    length_ratio: float
    latency_ms: float
    model_name: str
    error: str | None = None
    
    @property
    def is_quality_match(self) -> bool:
        """Check if responses are semantically similar enough."""
        return self.similarity_score >= 0.75
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/metrics."""
        return {
            "similarity_score": round(self.similarity_score, 4),
            "interpretation": self.interpretation,
            "cloud_length": self.cloud_response_length,
            "local_length": self.local_response_length,
            "length_ratio": round(self.length_ratio, 2),
            "latency_ms": round(self.latency_ms, 2),
            "is_quality_match": self.is_quality_match,
        }


class SimilarityScorer:
    """Computes semantic similarity between LLM outputs.
    
    Uses sentence-transformers models for embedding generation.
    Supports various similarity metrics.
    """
    
    # Available models (speed vs accuracy tradeoff)
    MODELS = {
        "fast": "all-MiniLM-L6-v2",           # 80MB, 384 dims, fast
        "balanced": "all-mpnet-base-v2",       # 420MB, 768 dims, good balance
        "accurate": "all-roberta-large-v1",    # 1.3GB, 1024 dims, best quality
    }
    
    # Similarity thresholds
    THRESHOLDS = {
        "high": 0.85,    # Responses are very similar
        "medium": 0.70,  # Responses convey similar meaning
        "low": 0.0,      # Responses differ significantly
    }
    
    def __init__(
        self,
        model_name: str = "fast",
        device: str = "cpu",
        enabled: bool = True,
    ):
        self.model_key = model_name
        self.model_name = self.MODELS.get(model_name, model_name)
        self.device = device
        self.enabled = enabled
        
        self._model = None
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the embedding model (lazy load)."""
        if not self.enabled:
            logger.info("Similarity scorer disabled")
            return
        
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info(
                    "Loading similarity model",
                    model=self.model_name,
                    device=self.device
                )
                
                # Load in thread pool
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(self.model_name, device=self.device)
                )
                
                self._initialized = True
                logger.info("Similarity model loaded successfully")
                
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                self.enabled = False
            except Exception as e:
                logger.error("Failed to load similarity model", error=str(e))
                self.enabled = False
    
    async def compute_similarity(
        self,
        cloud_response: str,
        local_response: str,
    ) -> SimilarityResult:
        """Compute semantic similarity between two responses.
        
        Args:
            cloud_response: Response from cloud model
            local_response: Response from local model
            
        Returns:
            SimilarityResult with score and metadata
        """
        import time
        start = time.perf_counter()
        
        if not self.enabled:
            return SimilarityResult(
                similarity_score=0.0,
                interpretation="low",
                cloud_response_length=len(cloud_response),
                local_response_length=len(local_response),
                length_ratio=0.0,
                latency_ms=0.0,
                model_name=self.model_name,
                error="Similarity scorer disabled",
            )
        
        if not self._initialized:
            await self.initialize()
        
        if not self._model:
            return SimilarityResult(
                similarity_score=0.0,
                interpretation="low",
                cloud_response_length=len(cloud_response),
                local_response_length=len(local_response),
                length_ratio=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
                model_name=self.model_name,
                error="Similarity model not available",
            )
        
        try:
            # Compute embeddings in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    [cloud_response, local_response],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            )
            
            # Cosine similarity (embeddings are normalized)
            numpy = _get_numpy()
            similarity = float(numpy.dot(embeddings[0], embeddings[1]))
            
            # Clamp to [0, 1]
            similarity = max(0.0, min(1.0, similarity))
            
            # Interpret score
            if similarity >= self.THRESHOLDS["high"]:
                interpretation = "high"
            elif similarity >= self.THRESHOLDS["medium"]:
                interpretation = "medium"
            else:
                interpretation = "low"
            
            # Length analysis
            cloud_len = len(cloud_response)
            local_len = len(local_response)
            length_ratio = local_len / cloud_len if cloud_len > 0 else 0.0
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            return SimilarityResult(
                similarity_score=similarity,
                interpretation=interpretation,
                cloud_response_length=cloud_len,
                local_response_length=local_len,
                length_ratio=length_ratio,
                latency_ms=latency_ms,
                model_name=self.model_name,
            )
            
        except Exception as e:
            logger.error("Similarity computation failed", error=str(e))
            return SimilarityResult(
                similarity_score=0.0,
                interpretation="low",
                cloud_response_length=len(cloud_response),
                local_response_length=len(local_response),
                length_ratio=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
                model_name=self.model_name,
                error=str(e),
            )
    
    def compute_similarity_sync(
        self,
        cloud_response: str,
        local_response: str,
    ) -> SimilarityResult:
        """Synchronous similarity computation (for testing)."""
        return asyncio.run(self.compute_similarity(cloud_response, local_response))


# Global singleton
_similarity_scorer: SimilarityScorer | None = None


def get_similarity_scorer() -> SimilarityScorer:
    """Get the global similarity scorer instance."""
    global _similarity_scorer
    if _similarity_scorer is None:
        _similarity_scorer = SimilarityScorer(enabled=False)
    return _similarity_scorer


def configure_similarity(
    model_name: str = "fast",
    device: str = "cpu",
    enabled: bool = True,
) -> SimilarityScorer:
    """Configure the global similarity scorer.
    
    Args:
        model_name: Model to use ("fast", "balanced", "accurate")
        device: Device for computation ("cpu", "cuda", "mps")
        enabled: Whether similarity scoring is enabled
        
    Returns:
        Configured SimilarityScorer instance
    """
    global _similarity_scorer
    _similarity_scorer = SimilarityScorer(
        model_name=model_name,
        device=device,
        enabled=enabled,
    )
    return _similarity_scorer


async def compute_similarity(
    cloud_response: str,
    local_response: str,
) -> SimilarityResult:
    """Compute similarity using global scorer."""
    scorer = get_similarity_scorer()
    return await scorer.compute_similarity(cloud_response, local_response)
