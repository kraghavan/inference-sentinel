"""Privacy classification module."""

from sentinel.classification.regex_classifier import (
    RegexClassifier,
    classify,
    classify_messages,
    get_classifier,
)
from sentinel.classification.schemas import (
    ClassificationResult,
    DetectedEntity,
    TIER_LABELS,
    get_tier_label,
)
from sentinel.classification.taxonomy import (
    EntityConfig,
    PrivacyTaxonomy,
    TierConfig,
    get_taxonomy,
    load_taxonomy,
    reload_taxonomy,
)

__all__ = [
    # Classifier
    "RegexClassifier",
    "classify",
    "classify_messages",
    "get_classifier",
    # Schemas
    "ClassificationResult",
    "DetectedEntity",
    "TIER_LABELS",
    "get_tier_label",
    # Taxonomy
    "PrivacyTaxonomy",
    "TierConfig",
    "EntityConfig",
    "get_taxonomy",
    "load_taxonomy",
    "reload_taxonomy",
]
