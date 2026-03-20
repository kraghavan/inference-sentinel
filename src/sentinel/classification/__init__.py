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
from sentinel.classification.ner_classifier import (
    NERClassifier,
    NEREntity,
    NERResult,
    get_ner_classifier,
    configure_ner,
)
from sentinel.classification.hybrid_classifier import (
    HybridClassifier,
    HybridResult,
    get_hybrid_classifier,
    configure_hybrid_classifier,
    classify_hybrid,
    classify_messages_hybrid,
)

__all__ = [
    # Regex Classifier
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
    # NER Classifier
    "NERClassifier",
    "NEREntity",
    "NERResult",
    "get_ner_classifier",
    "configure_ner",
    # Hybrid Classifier
    "HybridClassifier",
    "HybridResult",
    "get_hybrid_classifier",
    "configure_hybrid_classifier",
    "classify_hybrid",
    "classify_messages_hybrid",
]
