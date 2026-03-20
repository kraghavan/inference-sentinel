"""Hybrid classification pipeline combining regex and NER.

Pipeline flow:
1. Regex classifier runs first (fast, ~0.2ms)
2. If tier 3 found → stop, route local immediately
3. If tier < 3 and NER enabled → run NER for additional entities
4. Merge results, take highest tier

This provides the best of both worlds:
- Fast path for obvious PII (SSN, credit cards, etc.)
- NER catches names, organizations, addresses that regex misses
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

from sentinel.classification.regex_classifier import RegexClassifier, get_classifier
from sentinel.classification.ner_classifier import (
    NERClassifier,
    NERResult,
    get_ner_classifier,
    configure_ner,
)
from sentinel.classification.schemas import ClassificationResult, DetectedEntity
from sentinel.telemetry import get_logger, record_classification

logger = get_logger("sentinel.classification.hybrid")


@dataclass
class HybridResult:
    """Result from hybrid classification pipeline."""
    
    # Final merged result
    tier: int = 0
    tier_label: str = "PUBLIC"
    entities_detected: list[DetectedEntity] = field(default_factory=list)
    entity_types: list[str] = field(default_factory=list)
    entity_count: int = 0
    
    # Pipeline metadata
    regex_tier: int = 0
    ner_tier: int = 0
    regex_latency_ms: float = 0.0
    ner_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Flags
    ner_skipped: bool = False
    ner_skipped_reason: str = ""
    detection_method: str = "hybrid"
    
    @property
    def is_sensitive(self) -> bool:
        return self.tier > 0
    
    @property
    def requires_local(self) -> bool:
        return self.tier >= 2
    
    def to_classification_result(self) -> ClassificationResult:
        """Convert to standard ClassificationResult."""
        return ClassificationResult(
            tier=self.tier,
            tier_label=self.tier_label,
            entities_detected=self.entities_detected,
            entity_types=self.entity_types,
            confidence=1.0,
            detection_method=self.detection_method,
            detection_latency_ms=self.total_latency_ms,
        )


class HybridClassifier:
    """Hybrid classification pipeline combining regex and NER.
    
    Configuration options:
    - ner_enabled: Whether to use NER at all
    - ner_threshold_tier: Only run NER if regex tier is below this
    - ner_async: Run NER asynchronously (non-blocking)
    - skip_ner_on_tier3: Skip NER if regex finds tier 3 (default: True)
    """
    
    TIER_LABELS = {
        0: "PUBLIC",
        1: "INTERNAL", 
        2: "CONFIDENTIAL",
        3: "RESTRICTED",
    }
    
    def __init__(
        self,
        regex_classifier: RegexClassifier | None = None,
        ner_classifier: NERClassifier | None = None,
        ner_enabled: bool = False,
        ner_threshold_tier: int = 3,  # Run NER if regex tier < this
        skip_ner_on_tier3: bool = True,
    ):
        self._regex = regex_classifier or get_classifier()
        self._ner = ner_classifier or get_ner_classifier()
        self.ner_enabled = ner_enabled
        self.ner_threshold_tier = ner_threshold_tier
        self.skip_ner_on_tier3 = skip_ner_on_tier3
    
    async def initialize(self) -> None:
        """Initialize classifiers."""
        if self.ner_enabled:
            await self._ner.initialize()
    
    def classify_sync(self, text: str) -> HybridResult:
        """Synchronous classification (regex only, fast path)."""
        return asyncio.run(self.classify(text))
    
    async def classify(self, text: str) -> HybridResult:
        """Run hybrid classification pipeline.
        
        Args:
            text: Text to classify
            
        Returns:
            HybridResult with merged entities from both classifiers
        """
        start = time.perf_counter()
        
        # Step 1: Run regex classifier (always, fast)
        regex_start = time.perf_counter()
        regex_result = self._regex.classify(text)
        regex_latency_ms = (time.perf_counter() - regex_start) * 1000
        
        result = HybridResult(
            tier=regex_result.tier,
            tier_label=regex_result.tier_label,
            entities_detected=list(regex_result.entities_detected),
            entity_types=list(regex_result.entity_types),
            entity_count=regex_result.entity_count,
            regex_tier=regex_result.tier,
            regex_latency_ms=regex_latency_ms,
        )
        
        # Step 2: Decide if we should run NER
        should_run_ner = self._should_run_ner(regex_result.tier)
        
        if not should_run_ner:
            result.ner_skipped = True
            result.ner_skipped_reason = self._get_skip_reason(regex_result.tier)
            result.detection_method = "regex"
            result.total_latency_ms = (time.perf_counter() - start) * 1000
            return result
        
        # Step 3: Run NER classifier
        ner_result = await self._ner.classify(text)
        result.ner_latency_ms = ner_result.latency_ms
        result.ner_tier = ner_result.highest_tier
        
        if ner_result.error:
            logger.warning("NER failed, using regex only", error=ner_result.error)
            result.ner_skipped = True
            result.ner_skipped_reason = f"NER error: {ner_result.error}"
            result.detection_method = "regex"
            result.total_latency_ms = (time.perf_counter() - start) * 1000
            return result
        
        # Step 4: Merge results
        result = self._merge_results(result, ner_result)
        result.total_latency_ms = (time.perf_counter() - start) * 1000
        result.detection_method = "hybrid"
        
        return result
    
    async def classify_messages(
        self,
        messages: list[dict[str, str]]
    ) -> HybridResult:
        """Classify a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            HybridResult for combined content
        """
        # Combine all message content
        combined = " ".join(
            msg.get("content", "") 
            for msg in messages 
            if msg.get("content")
        )
        return await self.classify(combined)
    
    def _should_run_ner(self, regex_tier: int) -> bool:
        """Determine if NER should run based on regex result."""
        if not self.ner_enabled:
            return False
        
        if not self._ner.enabled:
            return False
        
        if self.skip_ner_on_tier3 and regex_tier >= 3:
            return False
        
        if regex_tier >= self.ner_threshold_tier:
            return False
        
        return True
    
    def _get_skip_reason(self, regex_tier: int) -> str:
        """Get reason for skipping NER."""
        if not self.ner_enabled:
            return "NER disabled in config"
        
        if not self._ner.enabled:
            return "NER classifier not available"
        
        if self.skip_ner_on_tier3 and regex_tier >= 3:
            return "Tier 3 detected by regex, NER not needed"
        
        if regex_tier >= self.ner_threshold_tier:
            return f"Regex tier {regex_tier} >= threshold {self.ner_threshold_tier}"
        
        return "Unknown"
    
    def _merge_results(
        self,
        hybrid: HybridResult,
        ner: NERResult
    ) -> HybridResult:
        """Merge NER results into hybrid result."""
        # Add NER entities
        for ner_entity in ner.entities:
            hybrid.entities_detected.append(DetectedEntity(
                entity_type=f"ner_{ner_entity.entity_type.lower()}",
                tier=ner_entity.tier,
                start_pos=ner_entity.start_pos,
                end_pos=ner_entity.end_pos,
                confidence=ner_entity.confidence,
                pattern_matched=f"NER:{ner_entity.entity_type}",
            ))
            
            if ner_entity.entity_type not in hybrid.entity_types:
                hybrid.entity_types.append(f"ner_{ner_entity.entity_type.lower()}")
        
        # Update counts and tier
        hybrid.entity_count = len(hybrid.entities_detected)
        hybrid.tier = max(hybrid.regex_tier, ner.highest_tier)
        hybrid.tier_label = self.TIER_LABELS.get(hybrid.tier, "UNKNOWN")
        
        return hybrid


# Global singleton
_hybrid_classifier: HybridClassifier | None = None


def get_hybrid_classifier() -> HybridClassifier:
    """Get the global hybrid classifier instance."""
    global _hybrid_classifier
    if _hybrid_classifier is None:
        _hybrid_classifier = HybridClassifier(ner_enabled=False)
    return _hybrid_classifier


def configure_hybrid_classifier(
    ner_enabled: bool = False,
    ner_model: str = "fast",
    ner_device: str = "cpu",
    ner_confidence_threshold: float = 0.7,
    ner_threshold_tier: int = 3,
    skip_ner_on_tier3: bool = True,
) -> HybridClassifier:
    """Configure the global hybrid classifier.
    
    Args:
        ner_enabled: Enable NER classification
        ner_model: NER model to use ("fast", "accurate", "multilingual")
        ner_device: Device for NER ("cpu", "cuda", "mps")
        ner_confidence_threshold: Minimum confidence for NER entities
        ner_threshold_tier: Only run NER if regex tier < this
        skip_ner_on_tier3: Skip NER if regex finds tier 3
        
    Returns:
        Configured HybridClassifier instance
    """
    global _hybrid_classifier
    
    # Configure NER
    ner = configure_ner(
        model_name=ner_model,
        device=ner_device,
        confidence_threshold=ner_confidence_threshold,
        enabled=ner_enabled,
    )
    
    # Create hybrid classifier
    _hybrid_classifier = HybridClassifier(
        ner_classifier=ner,
        ner_enabled=ner_enabled,
        ner_threshold_tier=ner_threshold_tier,
        skip_ner_on_tier3=skip_ner_on_tier3,
    )
    
    return _hybrid_classifier


async def classify_hybrid(text: str) -> HybridResult:
    """Classify text using hybrid pipeline."""
    classifier = get_hybrid_classifier()
    return await classifier.classify(text)


async def classify_messages_hybrid(
    messages: list[dict[str, str]]
) -> HybridResult:
    """Classify messages using hybrid pipeline."""
    classifier = get_hybrid_classifier()
    return await classifier.classify_messages(messages)
