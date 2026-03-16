"""Privacy taxonomy loader and data structures."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class TierConfig:
    """Configuration for a privacy tier."""

    level: int
    label: str
    description: str
    default_route: Literal["local", "cloud"]
    override_allowed: bool


@dataclass
class EntityConfig:
    """Configuration for an entity type."""

    name: str
    tier: int
    description: str
    patterns: list[str]
    compiled_patterns: list[re.Pattern] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compile regex patterns."""
        self.compiled_patterns = []
        for pattern in self.patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern for {self.name}: {pattern} - {e}")


@dataclass
class PrivacyTaxonomy:
    """Complete privacy taxonomy configuration."""

    tiers: dict[int, TierConfig]
    entities: dict[str, EntityConfig]

    def get_entities_by_tier(self, tier: int) -> list[EntityConfig]:
        """Get all entities for a specific tier."""
        return [e for e in self.entities.values() if e.tier == tier]

    def get_tier_label(self, tier: int) -> str:
        """Get the label for a tier."""
        if tier in self.tiers:
            return self.tiers[tier].label
        return "PUBLIC"

    def get_default_route(self, tier: int) -> Literal["local", "cloud"]:
        """Get the default route for a tier."""
        if tier in self.tiers:
            return self.tiers[tier].default_route
        return "cloud"

    def is_override_allowed(self, tier: int) -> bool:
        """Check if override is allowed for a tier."""
        if tier in self.tiers:
            return self.tiers[tier].override_allowed
        return True


def load_taxonomy(config_path: str | Path | None = None) -> PrivacyTaxonomy:
    """Load privacy taxonomy from YAML file.

    Args:
        config_path: Path to the taxonomy YAML file.
                     If None, uses default location.

    Returns:
        Loaded PrivacyTaxonomy instance.
    """
    if config_path is None:
        # Default paths to check
        default_paths = [
            Path("config/privacy_taxonomy.yaml"),
            Path("/app/config/privacy_taxonomy.yaml"),
            Path(__file__).parent.parent.parent.parent / "config" / "privacy_taxonomy.yaml",
        ]
        for path in default_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(
                f"Privacy taxonomy not found. Searched: {[str(p) for p in default_paths]}"
            )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Privacy taxonomy not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Parse tiers
    tiers: dict[int, TierConfig] = {}
    for tier_num, tier_data in data.get("tiers", {}).items():
        tiers[int(tier_num)] = TierConfig(
            level=int(tier_num),
            label=tier_data["label"],
            description=tier_data["description"],
            default_route=tier_data["default_route"],
            override_allowed=tier_data["override_allowed"],
        )

    # Parse entities
    entities: dict[str, EntityConfig] = {}
    for entity_name, entity_data in data.get("entities", {}).items():
        entities[entity_name] = EntityConfig(
            name=entity_name,
            tier=entity_data["tier"],
            description=entity_data["description"],
            patterns=entity_data.get("patterns", []),
        )

    return PrivacyTaxonomy(tiers=tiers, entities=entities)


# Global cached taxonomy instance
_cached_taxonomy: PrivacyTaxonomy | None = None


def get_taxonomy() -> PrivacyTaxonomy:
    """Get the cached taxonomy instance, loading if necessary."""
    global _cached_taxonomy
    if _cached_taxonomy is None:
        _cached_taxonomy = load_taxonomy()
    return _cached_taxonomy


def reload_taxonomy(config_path: str | Path | None = None) -> PrivacyTaxonomy:
    """Reload the taxonomy from file."""
    global _cached_taxonomy
    _cached_taxonomy = load_taxonomy(config_path)
    return _cached_taxonomy
