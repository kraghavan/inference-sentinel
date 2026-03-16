"""Regex-based privacy classifier."""

import hashlib
import time
from typing import Literal

import structlog

from sentinel.classification.schemas import (
    ClassificationResult,
    DetectedEntity,
    get_tier_label,
)
from sentinel.classification.taxonomy import PrivacyTaxonomy, get_taxonomy

logger = structlog.get_logger()


class RegexClassifier:
    """Fast regex-based classifier for detecting sensitive content.

    This classifier uses pre-compiled regex patterns to detect PII
    and other sensitive data in text. It's designed for high-precision
    detection of Tier 2-3 entities with minimal false positives.
    """

    def __init__(self, taxonomy: PrivacyTaxonomy | None = None):
        """Initialize the classifier.

        Args:
            taxonomy: Privacy taxonomy to use. If None, loads default.
        """
        self._taxonomy = taxonomy or get_taxonomy()

    @property
    def taxonomy(self) -> PrivacyTaxonomy:
        """Get the taxonomy."""
        return self._taxonomy

    def classify(self, text: str) -> ClassificationResult:
        """Classify text for privacy-sensitive content.

        Args:
            text: The text to classify.

        Returns:
            ClassificationResult with detected entities and tier.
        """
        start_time = time.perf_counter()

        entities: list[DetectedEntity] = []
        seen_spans: set[tuple[int, int]] = set()  # Avoid duplicate detections

        # Check each entity type
        for entity_config in self._taxonomy.entities.values():
            for pattern in entity_config.compiled_patterns:
                for match in pattern.finditer(text):
                    start, end = match.span()

                    # Skip if we've already detected something at this span
                    if (start, end) in seen_spans:
                        continue
                    seen_spans.add((start, end))

                    # Hash the matched value for safe logging
                    matched_value = match.group()
                    value_hash = hashlib.sha256(matched_value.encode()).hexdigest()[:16]

                    entities.append(
                        DetectedEntity(
                            entity_type=entity_config.name,
                            tier=entity_config.tier,
                            start_pos=start,
                            end_pos=end,
                            confidence=1.0,  # Regex matches are high confidence
                            pattern_matched=pattern.pattern,
                            value_hash=value_hash,
                        )
                    )

        # Determine overall tier (highest detected)
        if entities:
            max_tier = max(e.tier for e in entities)
        else:
            max_tier = 0

        # Get unique entity types
        entity_types = sorted(set(e.entity_type for e in entities))

        latency_ms = (time.perf_counter() - start_time) * 1000

        result = ClassificationResult(
            tier=max_tier,
            tier_label=get_tier_label(max_tier),
            entities_detected=entities,
            entity_types=entity_types,
            confidence=1.0 if entities else 1.0,
            detection_method="regex",
            detection_latency_ms=latency_ms,
        )

        if entities:
            logger.debug(
                "Classification complete",
                tier=max_tier,
                tier_label=result.tier_label,
                entity_count=len(entities),
                entity_types=entity_types,
                latency_ms=round(latency_ms, 2),
            )

        return result

    def classify_messages(
        self, messages: list[dict[str, str]]
    ) -> ClassificationResult:
        """Classify a list of conversation messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            ClassificationResult for the combined messages.
        """
        # Combine all message content
        combined_text = "\n".join(
            msg.get("content", "") for msg in messages if msg.get("content")
        )
        return self.classify(combined_text)

    def quick_check(self, text: str, min_tier: int = 2) -> bool:
        """Quick check if text contains sensitive content at or above a tier.

        This is optimized for speed - it returns True on first match
        at or above the specified tier.

        Args:
            text: Text to check.
            min_tier: Minimum tier to check for (default: 2 = CONFIDENTIAL).

        Returns:
            True if sensitive content at or above min_tier is detected.
        """
        for entity_config in self._taxonomy.entities.values():
            if entity_config.tier < min_tier:
                continue

            for pattern in entity_config.compiled_patterns:
                if pattern.search(text):
                    return True

        return False

    def get_tier_for_entity(self, entity_type: str) -> int:
        """Get the tier for an entity type.

        Args:
            entity_type: The entity type name.

        Returns:
            The tier (0-3) or 0 if unknown.
        """
        if entity_type in self._taxonomy.entities:
            return self._taxonomy.entities[entity_type].tier
        return 0


# Global classifier instance
_classifier: RegexClassifier | None = None


def get_classifier() -> RegexClassifier:
    """Get the global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = RegexClassifier()
    return _classifier


def classify(text: str) -> ClassificationResult:
    """Convenience function to classify text using the global classifier."""
    return get_classifier().classify(text)


def classify_messages(messages: list[dict[str, str]]) -> ClassificationResult:
    """Convenience function to classify messages using the global classifier."""
    return get_classifier().classify_messages(messages)
