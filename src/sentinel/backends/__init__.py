"""Inference backends module."""

from sentinel.backends.base import BaseBackend, InferenceResult, StreamChunk
from sentinel.backends.manager import BackendManager
from sentinel.backends.ollama import OllamaBackend

__all__ = [
    "BaseBackend",
    "InferenceResult",
    "StreamChunk",
    "OllamaBackend",
    "BackendManager",
]
