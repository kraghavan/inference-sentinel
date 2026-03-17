"""Google (Gemini) backend for inference-sentinel."""

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
GOOGLE_PRICING = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # Default fallback
    "default": {"input": 0.075, "output": 0.30},
}

GOOGLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
]


class GoogleBackend(CloudBackend):
    """Backend for Google's Gemini API.
    
    Supports Gemini 1.5 and 2.0 family models.
    
    Features:
    - Long context windows (1M+ tokens)
    - Multimodal (text, images, video, audio)
    - Competitive pricing
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        """Initialize the Google backend.

        Args:
            api_key: Google AI API key.
            model: Model to use (default: gemini-1.5-flash).
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
        return "google"

    @property
    def provider(self) -> str:
        return "google"

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def supported_models(self) -> list[str]:
        return GOOGLE_MODELS.copy()

    def get_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a model."""
        return GOOGLE_PRICING.get(model, GOOGLE_PRICING["default"])

    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com",
            headers={
                "content-type": "application/json",
            },
            timeout=self._timeout,
        )
        self._healthy = True
        logger.info("Google backend initialized", model=self._model)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the backend is healthy."""
        if not self._client:
            return False
        return True

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert messages to Gemini format.

        Gemini uses a different format:
        - "user" and "model" roles (not "assistant")
        - "parts" array instead of "content" string
        - System instruction is separate

        Returns:
            Tuple of (system_instruction, contents)
        """
        system = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system = content
            else:
                # Gemini uses "model" instead of "assistant"
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })

        return system, contents

    async def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> InferenceResult:
        """Generate a response using Gemini.

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

        system, contents = self._convert_messages(messages)

        request_body = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            request_body["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            response = await self._client.post(
                f"/v1beta/models/{model}:generateContent",
                params={"key": self._api_key},
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            total_latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract content
            content = ""
            candidates = data.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                parts = candidate.get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        content += part["text"]

            # Extract usage
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            completion_tokens = usage.get("candidatesTokenCount", 0)

            # Calculate cost using inherited method
            cost_usd = self.calculate_cost(model, prompt_tokens, completion_tokens)

            # Map finish reason
            finish_reason = "stop"
            if candidates:
                gemini_reason = candidates[0].get("finishReason", "STOP")
                if gemini_reason == "MAX_TOKENS":
                    finish_reason = "length"
                elif gemini_reason == "SAFETY":
                    finish_reason = "content_filter"

            logger.info(
                "Google generation complete",
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
                ttft_ms=total_latency_ms,  # Non-streaming
                cost_usd=cost_usd,
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"Google API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = f"{error_msg} - {error_data.get('error', {}).get('message', '')}"
            except Exception:
                pass
            logger.error("Google API error", error=error_msg)
            return InferenceResult(
                content="",
                model=model,
                error=error_msg,
            )
        except Exception as e:
            logger.error("Google generation failed", error=str(e))
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
        """Generate a streaming response using Gemini.

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
        token_count = 0

        system, contents = self._convert_messages(messages)

        request_body = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            request_body["systemInstruction"] = {"parts": [{"text": system}]}

        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with self._client.stream(
                "POST",
                f"/v1beta/models/{model}:streamGenerateContent",
                params={"key": self._api_key, "alt": "sse"},
                json=request_body,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if not data_str:
                        continue

                    try:
                        import json
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract text from candidates
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            if "text" in part:
                                text = part["text"]
                                if text:
                                    now = time.perf_counter()
                                    if first_token_time is None:
                                        first_token_time = now
                                    else:
                                        itl_ms = (now - last_token_time) * 1000
                                        itl_values.append(itl_ms)
                                    last_token_time = now
                                    token_count += 1

                                    yield StreamChunk(
                                        content=text,
                                        token_index=token_count - 1,
                                    )

                    # Check for usage metadata
                    usage = data.get("usageMetadata", {})
                    if usage:
                        prompt_tokens = usage.get("promptTokenCount", prompt_tokens)
                        completion_tokens = usage.get("candidatesTokenCount", completion_tokens)

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
            logger.error("Google streaming failed", error=str(e))
            yield StreamChunk(
                content="",
                finish_reason="error",
                error=str(e),
            )
