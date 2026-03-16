"""Classification result schemas."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DetectedEntity:
    """A single detected entity in the text."""

    entity_type: str  # e.g., "ssn", "email"
    tier: int  # 0-3
    start_pos: int  # Character position
    end_pos: int
    confidence: float  # 0.0-1.0
    pattern_matched: str  # Which pattern triggered
    value_hash: str = ""  # SHA-256 of matched value (never log raw)

    def __post_init__(self) -> None:
        """Hash the matched value for logging without exposing PII."""
        # Hash is computed by the classifier, not here


@dataclass
class ClassificationResult:
    """Result of classifying a prompt for privacy-sensitive content."""

    tier: int  # Highest tier detected (0-3)
    tier_label: Literal["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"]
    entities_detected: list[DetectedEntity] = field(default_factory=list)
    entity_types: list[str] = field(default_factory=list)  # Unique types found
    confidence: float = 1.0  # Overall confidence
    detection_method: str = "regex"  # "regex", "ner", "hybrid"
    detection_latency_ms: float = 0.0

    @property
    def is_sensitive(self) -> bool:
        """Returns True if any sensitive content was detected (tier > 0)."""
        return self.tier > 0

    @property
    def requires_local(self) -> bool:
        """Returns True if content should be routed locally (tier >= 2)."""
        return self.tier >= 2

    @property
    def entity_count(self) -> int:
        """Number of entities detected."""
        return len(self.entities_detected)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tier": self.tier,
            "tier_label": self.tier_label,
            "entities_detected": [
                {
                    "entity_type": e.entity_type,
                    "tier": e.tier,
                    "start_pos": e.start_pos,
                    "end_pos": e.end_pos,
                    "confidence": e.confidence,
                }
                for e in self.entities_detected
            ],
            "entity_types": self.entity_types,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "detection_latency_ms": self.detection_latency_ms,
        }


# Tier label mapping
TIER_LABELS: dict[int, Literal["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"]] = {
    0: "PUBLIC",
    1: "INTERNAL",
    2: "CONFIDENTIAL",
    3: "RESTRICTED",
}


def get_tier_label(tier: int) -> Literal["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"]:
    """Get the label for a tier number."""
    return TIER_LABELS.get(tier, "PUBLIC")
