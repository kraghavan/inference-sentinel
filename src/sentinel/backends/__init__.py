"""Inference backends module."""

from sentinel.backends.base import (
    BaseBackend,
    LocalBackend,
    CloudBackend,
    InferenceResult,
    StreamChunk,
)
from sentinel.backends.manager import BackendManager
from sentinel.backends.ollama import OllamaBackend
from sentinel.backends.anthropic import AnthropicBackend
from sentinel.backends.google import GoogleBackend

__all__ = [
    # Base classes
    "BaseBackend",
    "LocalBackend",
    "CloudBackend",
    "InferenceResult",
    "StreamChunk",
    # Local backends
    "OllamaBackend",
    # Cloud backends
    "AnthropicBackend",
    "GoogleBackend",
    # Manager
    "BackendManager",
]
