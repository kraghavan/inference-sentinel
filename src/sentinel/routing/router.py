"""Routing engine for inference-sentinel.

Routes requests to local or cloud backends based on privacy classification.
"""

from dataclasses import dataclass
from typing import Literal

import structlog

from sentinel.classification import ClassificationResult, get_taxonomy

logger = structlog.get_logger()


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    route: Literal["local", "cloud"]
    reason: str
    tier: int
    tier_label: str
    override_applied: bool = False
    original_route: Literal["local", "cloud"] | None = None


class Router:
    """Routes requests based on privacy classification.

    Routing Rules:
    - Tier 0 (PUBLIC): Cloud (fastest/cheapest)
    - Tier 1 (INTERNAL): Cloud (configurable)
    - Tier 2 (CONFIDENTIAL): Local by default, override allowed
    - Tier 3 (RESTRICTED): Local ALWAYS, no override permitted

    The router respects user overrides except for Tier 3 content,
    which must always be processed locally for compliance.
    """

    def __init__(
        self,
        default_route: Literal["local", "cloud"] = "cloud",
        tier1_route: Literal["local", "cloud"] = "cloud",
        tier2_route: Literal["local", "cloud"] = "local",
    ):
        """Initialize the router.

        Args:
            default_route: Default route for Tier 0 content.
            tier1_route: Route for Tier 1 (INTERNAL) content.
            tier2_route: Route for Tier 2 (CONFIDENTIAL) content.
        """
        self._default_route = default_route
        self._tier1_route = tier1_route
        self._tier2_route = tier2_route
        self._taxonomy = get_taxonomy()

    def route(
        self,
        classification: ClassificationResult,
        override: Literal["local", "cloud"] | None = None,
    ) -> RoutingDecision:
        """Determine the route for a request based on classification.

        Args:
            classification: The privacy classification result.
            override: Optional user-requested route override.

        Returns:
            RoutingDecision with the chosen route and reason.
        """
        tier = classification.tier
        tier_label = classification.tier_label

        # Determine base route based on tier
        if tier == 0:
            base_route = self._default_route
            reason = "No sensitive content detected"
        elif tier == 1:
            base_route = self._tier1_route
            reason = f"Internal content detected: {', '.join(classification.entity_types)}"
        elif tier == 2:
            base_route = self._tier2_route
            reason = f"Confidential content detected: {', '.join(classification.entity_types)}"
        else:  # tier == 3
            base_route: Literal["local", "cloud"] = "local"
            reason = f"Restricted content detected: {', '.join(classification.entity_types)}"

        # Apply override if allowed
        final_route = base_route
        override_applied = False
        original_route = None

        if override and override != base_route:
            if tier == 3:
                # Tier 3: Override NOT allowed - log warning
                logger.warning(
                    "Override rejected for restricted content",
                    tier=tier,
                    requested_override=override,
                    entity_types=classification.entity_types,
                )
                reason = f"RESTRICTED content - override rejected, must route locally: {', '.join(classification.entity_types)}"
            elif tier == 2 and override == "cloud":
                # Tier 2: Override allowed but logged
                logger.info(
                    "Override applied for confidential content",
                    tier=tier,
                    override=override,
                    entity_types=classification.entity_types,
                )
                final_route = override
                override_applied = True
                original_route = base_route
                reason = f"Override applied: routing CONFIDENTIAL content to cloud (user requested)"
            else:
                # Tier 0-1: Override freely allowed
                final_route = override
                override_applied = True
                original_route = base_route
                reason = f"Override applied: {override}"

        logger.debug(
            "Routing decision",
            tier=tier,
            tier_label=tier_label,
            base_route=base_route,
            final_route=final_route,
            override_applied=override_applied,
        )

        return RoutingDecision(
            route=final_route,
            reason=reason,
            tier=tier,
            tier_label=tier_label,
            override_applied=override_applied,
            original_route=original_route,
        )

    def quick_route(self, text: str) -> Literal["local", "cloud"]:
        """Quick routing decision for simple cases.

        This performs classification and routing in one step.
        Use the full route() method when you need detailed results.

        Args:
            text: The text to classify and route.

        Returns:
            "local" or "cloud"
        """
        from sentinel.classification import classify

        classification = classify(text)
        decision = self.route(classification)
        return decision.route


# Global router instance
_router: Router | None = None


def get_router() -> Router:
    """Get the global router instance."""
    global _router
    if _router is None:
        _router = Router()
    return _router


def route(
    classification: ClassificationResult,
    override: Literal["local", "cloud"] | None = None,
) -> RoutingDecision:
    """Convenience function to route using the global router."""
    return get_router().route(classification, override)
