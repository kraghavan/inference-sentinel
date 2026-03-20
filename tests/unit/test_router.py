"""Tests for the routing module."""

import pytest

from sentinel.classification import ClassificationResult
from sentinel.routing import Router, RoutingDecision, route


@pytest.fixture
def router() -> Router:
    """Create a router with default configuration."""
    return Router()


class TestRouterBasicRouting:
    """Tests for basic routing logic."""

    def test_tier0_routes_to_cloud(self, router: Router) -> None:
        """Tier 0 (PUBLIC) should route to cloud by default."""
        classification = ClassificationResult(
            tier=0,
            tier_label="PUBLIC",
            entities_detected=[],
            entity_types=[],
        )
        decision = router.route(classification)
        assert decision.route == "cloud"
        assert decision.tier == 0
        assert not decision.override_applied

    def test_tier1_routes_to_cloud(self, router: Router) -> None:
        """Tier 1 (INTERNAL) should route to cloud by default."""
        classification = ClassificationResult(
            tier=1,
            tier_label="INTERNAL",
            entities_detected=[],
            entity_types=["employee_id"],
        )
        decision = router.route(classification)
        assert decision.route == "cloud"
        assert decision.tier == 1

    def test_tier2_routes_to_local(self, router: Router) -> None:
        """Tier 2 (CONFIDENTIAL) should route to local by default."""
        classification = ClassificationResult(
            tier=2,
            tier_label="CONFIDENTIAL",
            entities_detected=[],
            entity_types=["email"],
        )
        decision = router.route(classification)
        assert decision.route == "local"
        assert decision.tier == 2

    def test_tier3_routes_to_local(self, router: Router) -> None:
        """Tier 3 (RESTRICTED) should route to local."""
        classification = ClassificationResult(
            tier=3,
            tier_label="RESTRICTED",
            entities_detected=[],
            entity_types=["ssn"],
        )
        decision = router.route(classification)
        assert decision.route == "local"
        assert decision.tier == 3


class TestRouterOverrides:
    """Tests for routing override behavior."""

    def test_tier0_override_to_local_allowed(self, router: Router) -> None:
        """Tier 0 can be overridden to local."""
        classification = ClassificationResult(
            tier=0,
            tier_label="PUBLIC",
        )
        decision = router.route(classification, override="local")
        assert decision.route == "local"
        assert decision.override_applied is True
        assert decision.original_route == "cloud"

    def test_tier1_override_to_local_allowed(self, router: Router) -> None:
        """Tier 1 can be overridden to local."""
        classification = ClassificationResult(
            tier=1,
            tier_label="INTERNAL",
            entity_types=["project_code"],
        )
        decision = router.route(classification, override="local")
        assert decision.route == "local"
        assert decision.override_applied is True

    def test_tier2_override_to_cloud_allowed(self, router: Router) -> None:
        """Tier 2 can be overridden to cloud (with logging)."""
        classification = ClassificationResult(
            tier=2,
            tier_label="CONFIDENTIAL",
            entity_types=["email"],
        )
        decision = router.route(classification, override="cloud")
        assert decision.route == "cloud"
        assert decision.override_applied is True
        assert decision.original_route == "local"

    def test_tier3_override_to_cloud_rejected(self, router: Router) -> None:
        """Tier 3 CANNOT be overridden to cloud - must stay local."""
        classification = ClassificationResult(
            tier=3,
            tier_label="RESTRICTED",
            entity_types=["ssn"],
        )
        decision = router.route(classification, override="cloud")
        # Override should be rejected
        assert decision.route == "local"
        assert decision.override_applied is False
        assert "rejected" in decision.reason.lower()

    def test_same_route_override_no_change(self, router: Router) -> None:
        """Override to same route doesn't count as override."""
        classification = ClassificationResult(
            tier=0,
            tier_label="PUBLIC",
        )
        decision = router.route(classification, override="cloud")
        # Already going to cloud, so no override applied
        assert decision.route == "cloud"
        assert decision.override_applied is False


class TestRouterCustomConfig:
    """Tests for router with custom configuration."""

    def test_custom_tier1_route(self) -> None:
        """Router can be configured to route tier1 to local."""
        router = Router(tier1_route="local")
        classification = ClassificationResult(
            tier=1,
            tier_label="INTERNAL",
            entity_types=["employee_id"],
        )
        decision = router.route(classification)
        assert decision.route == "local"

    def test_custom_default_route(self) -> None:
        """Router can be configured with different default route."""
        router = Router(default_route="local")
        classification = ClassificationResult(
            tier=0,
            tier_label="PUBLIC",
        )
        decision = router.route(classification)
        assert decision.route == "local"


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_decision_fields(self) -> None:
        """Test RoutingDecision has all expected fields."""
        decision = RoutingDecision(
            route="local",
            reason="Test reason",
            tier=3,
            tier_label="RESTRICTED",
            override_applied=True,
            original_route="cloud",
        )
        assert decision.route == "local"
        assert decision.reason == "Test reason"
        assert decision.tier == 3
        assert decision.tier_label == "RESTRICTED"
        assert decision.override_applied is True
        assert decision.original_route == "cloud"


class TestQuickRoute:
    """Tests for quick_route method."""

    def test_quick_route_safe_text(self, router: Router) -> None:
        """Quick route for safe text returns cloud."""
        result = router.quick_route("What is the capital of France?")
        assert result == "cloud"

    def test_quick_route_sensitive_text(self, router: Router) -> None:
        """Quick route for sensitive text returns local."""
        result = router.quick_route("My SSN is 123-45-6789")
        assert result == "local"


class TestConvenienceFunction:
    """Tests for module-level convenience function."""

    def test_route_function(self) -> None:
        """Test the module-level route function."""
        classification = ClassificationResult(
            tier=3,
            tier_label="RESTRICTED",
            entity_types=["credit_card"],
        )
        decision = route(classification)
        assert decision.route == "local"
        assert decision.tier == 3
