"""Routing module for inference-sentinel."""

from sentinel.routing.router import (
    Router,
    RoutingDecision,
    get_router,
    route,
)

__all__ = [
    "Router",
    "RoutingDecision",
    "get_router",
    "route",
]
