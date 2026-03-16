"""Ollama backend adapter for local inference."""

import time
from typing import AsyncIterator

import httpx
import structlog

from sentinel.backends.base import BaseBackend, InferenceResult, StreamChunk
from sentinel.config import LocalEndpoint

logger = structlog.get_logger()


class OllamaBackend(BaseBackend):
    """Backend adapter for Ollama local inference."""

    def __init__(self, endpoint: LocalEndpoint, timeout: float = 120.0):
        self._endpoint = endpoint
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self._endpoint.name}"

    @property
    def endpoint_name(self) -> str:
        return self._endpoint.name

    @property
    def base_url(self) -> str:
        return self._endpoint.base_url

    @property
    def model(self) -> str:
        return self._endpoint.model

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._endpoint.base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> InferenceResult:
        """Generate a completion using Ollama's chat API."""
        client = await self._get_client()
        model = model or self._endpoint.model

        start_time = time.perf_counter()
        ttft_ms: float | None = None
        itl_values: list[float] = []
        content_parts: list[str] = []
        last_token_time: float | None = None

        try:
            # Use streaming internally to capture TTFT and ITL
            async with client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    import json
                    chunk = json.loads(line)
                    current_time = time.perf_counter()

                    if "message" in chunk and "content" in chunk["message"]:
                        token_content = chunk["message"]["content"]
                        if token_content:
                            content_parts.append(token_content)

                            # Record TTFT on first token
                            if ttft_ms is None:
                                ttft_ms = (current_time - start_time) * 1000

                            # Record ITL for subsequent tokens
                            if last_token_time is not None:
                                itl_ms = (current_time - last_token_time) * 1000
                                itl_values.append(itl_ms)

                            last_token_time = current_time

                    # Check for completion
                    if chunk.get("done", False):
                        break

            total_latency_ms = (time.perf_counter() - start_time) * 1000
            content = "".join(content_parts)

            # Estimate tokens (Ollama doesn't always return accurate counts)
            prompt_tokens = sum(len(m.get("content", "").split()) for m in messages) * 4 // 3
            completion_tokens = len(content.split()) * 4 // 3

            return InferenceResult(
                content=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms,
                itl_values_ms=itl_values if itl_values else None,
                total_latency_ms=total_latency_ms,
                finish_reason="stop",
            )

        except httpx.HTTPStatusError as e:
            logger.error("Ollama HTTP error", status=e.response.status_code, error=str(e))
            return InferenceResult(
                content="",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_latency_ms=(time.perf_counter() - start_time) * 1000,
                finish_reason="error",
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except httpx.RequestError as e:
            logger.error("Ollama request error", error=str(e))
            return InferenceResult(
                content="",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_latency_ms=(time.perf_counter() - start_time) * 1000,
                finish_reason="error",
                error=f"Request failed: {str(e)}",
            )

    async def generate_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion."""
        client = await self._get_client()
        model = model or self._endpoint.model

        start_time = time.perf_counter()
        is_first = True

        try:
            async with client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    import json
                    chunk = json.loads(line)
                    current_time = (time.perf_counter() - start_time) * 1000

                    if "message" in chunk and "content" in chunk["message"]:
                        token_content = chunk["message"]["content"]
                        if token_content:
                            yield StreamChunk(
                                content=token_content,
                                is_first=is_first,
                                is_last=chunk.get("done", False),
                                timestamp_ms=current_time,
                            )
                            is_first = False

                    if chunk.get("done", False):
                        break

        except httpx.HTTPStatusError as e:
            logger.error("Ollama streaming error", error=str(e))
            yield StreamChunk(
                content="",
                is_first=is_first,
                is_last=True,
                timestamp_ms=(time.perf_counter() - start_time) * 1000,
            )

    async def health_check(self) -> bool:
        """Check if Ollama is healthy and the model is available."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [m["name"] for m in data.get("models", [])]

            # Check if our configured model is available
            model_available = any(
                self._endpoint.model in m or m in self._endpoint.model
                for m in models
            )

            if not model_available:
                logger.warning(
                    "Model not found on endpoint",
                    endpoint=self._endpoint.name,
                    model=self._endpoint.model,
                    available=models,
                )

            return model_available

        except Exception as e:
            logger.error(
                "Health check failed",
                endpoint=self._endpoint.name,
                error=str(e),
            )
            return False

    async def list_models(self) -> list[str]:
        """List available models on this Ollama instance."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            return [m["name"] for m in data.get("models", [])]

        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            return []
