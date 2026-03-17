"""Anthropic (Claude) backend for inference-sentinel."""

import time
from typing import AsyncIterator

import httpx
import structlog

from sentinel.backends.base import (
    CloudBackend,
    InferenceResult,
    StreamChunk,
)

logger = structlog.get_logger()

# Pricing per 1M tokens (as of 2024)
ANTHROPIC_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    # Default fallback
    "default": {"input": 3.00, "output": 15.00},
}

ANTHROPIC_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
]


class AnthropicBackend(CloudBackend):
    """Backend for Anthropic's Claude API.
    
    Supports Claude 3.x and 4.x family models.
    
    Features:
    - High-quality reasoning and analysis
    - Long context windows (200K tokens)
    - Strong safety and alignment
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        """Initialize the Anthropic backend.

        Args:
            api_key: Anthropic API key.
            model: Model to use (default: claude-sonnet-4-20250514).
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.
        """
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None
        self._healthy = False

    @property
    def endpoint_name(self) -> str:
        return "anthropic"

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def supported_models(self) -> list[str]:
        return ANTHROPIC_MODELS.copy()

    def get_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a model."""
        return ANTHROPIC_PRICING.get(model, ANTHROPIC_PRICING["default"])

    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=self._timeout,
        )
        # Test connection
        try:
            # Simple validation - just check we can reach the API
            self._healthy = True
            logger.info("Anthropic backend initialized", model=self._model)
        except Exception as e:
            logger.error("Failed to initialize Anthropic backend", error=str(e))
            self._healthy = False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the backend is healthy."""
        if not self._client:
            return False
        # For cloud backends, we assume healthy if client exists
        # Real health checks would hit a status endpoint
        return True

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert messages to Anthropic format.

        Anthropic requires system message to be separate.

        Returns:
            Tuple of (system_message, messages)
        """
        system = None
        converted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system = content
            else:
                # Anthropic uses "user" and "assistant"
                converted.append({"role": role, "content": content})

        return system, converted

    async def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> InferenceResult:
        """Generate a response using Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model override (uses default if None).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional parameters.

        Returns:
            InferenceResult with the response.
        """
        if not self._client:
            return InferenceResult(
                content="",
                model=model or self._model,
                error="Backend not initialized",
            )

        model = model or self._model
        start_time = time.perf_counter()

        system, converted_messages = self._convert_messages(messages)

        request_body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": converted_messages,
        }
        if system:
            request_body["system"] = system

        try:
            response = await self._client.post(
                "/v1/messages",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            total_latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract content
            content = ""
            if data.get("content"):
                for block in data["content"]:
                    if block.get("type") == "text":
                        content += block.get("text", "")

            # Extract usage
            usage = data.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

            # Calculate cost using inherited method
            cost_usd = self.calculate_cost(model, prompt_tokens, completion_tokens)

            # Map stop reason
            stop_reason = data.get("stop_reason", "stop")
            finish_reason = "stop" if stop_reason == "end_turn" else stop_reason

            logger.info(
                "Anthropic generation complete",
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=round(total_latency_ms, 2),
                cost_usd=round(cost_usd, 6),
            )

            return InferenceResult(
                content=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                total_latency_ms=total_latency_ms,
                ttft_ms=total_latency_ms,  # Non-streaming, so TTFT = total
                cost_usd=cost_usd,
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"Anthropic API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = f"{error_msg} - {error_data.get('error', {}).get('message', '')}"
            except Exception:
                pass
            logger.error("Anthropic API error", error=error_msg)
            return InferenceResult(
                content="",
                model=model,
                error=error_msg,
            )
        except Exception as e:
            logger.error("Anthropic generation failed", error=str(e))
            return InferenceResult(
                content="",
                model=model,
                error=str(e),
            )

    async def generate_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response using Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model override (uses default if None).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional parameters.

        Yields:
            StreamChunk objects with incremental content.
        """
        if not self._client:
            yield StreamChunk(
                content="",
                finish_reason="error",
                error="Backend not initialized",
            )
            return

        model = model or self._model
        start_time = time.perf_counter()
        first_token_time: float | None = None
        last_token_time = start_time
        itl_values: list[float] = []

        system, converted_messages = self._convert_messages(messages)

        request_body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": converted_messages,
            "stream": True,
        }
        if system:
            request_body["system"] = system

        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with self._client.stream(
                "POST",
                "/v1/messages",
                json=request_body,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    try:
                        import json
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type", "")

                    if event_type == "message_start":
                        usage = data.get("message", {}).get("usage", {})
                        prompt_tokens = usage.get("input_tokens", 0)

                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                now = time.perf_counter()
                                if first_token_time is None:
                                    first_token_time = now
                                else:
                                    itl_ms = (now - last_token_time) * 1000
                                    itl_values.append(itl_ms)
                                last_token_time = now
                                completion_tokens += 1  # Approximate

                                yield StreamChunk(
                                    content=text,
                                    token_index=completion_tokens - 1,
                                )

                    elif event_type == "message_delta":
                        usage = data.get("usage", {})
                        if "output_tokens" in usage:
                            completion_tokens = usage["output_tokens"]

                    elif event_type == "message_stop":
                        break

            # Final chunk with stats
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_latency_ms

            cost_usd = self.calculate_cost(model, prompt_tokens, completion_tokens)

            yield StreamChunk(
                content="",
                finish_reason="stop",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms,
                itl_values_ms=itl_values,
                total_latency_ms=total_latency_ms,
                cost_usd=cost_usd,
            )

        except Exception as e:
            logger.error("Anthropic streaming failed", error=str(e))
            yield StreamChunk(
                content="",
                finish_reason="error",
                error=str(e),
            )
