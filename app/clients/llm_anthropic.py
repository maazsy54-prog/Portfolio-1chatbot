"""LLM client for Anthropic's Claude (native Messages API).

Same interface as the other LLM clients (`chat_json`), so generation and the evaluation judge do
not care which provider answers. Claude has no JSON-mode flag; the prompts demand JSON only and
our parser tolerates stray text or code fences around it.
"""
from typing import Any

import httpx

from app.clients.cohere_client import LLMResponse
from app.clients.resilience import call_with_retries
from app.clients.throttle import CallThrottle
from app.clients.upstream import UpstreamHTTPError, error_body
from app.config import Settings

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicLLM:
    """Calls POST /v1/messages and returns the reply text and token usage."""

    def __init__(
        self, settings: Settings, http: httpx.AsyncClient | None = None, throttle: Any | None = None
    ) -> None:
        self._settings = settings
        self._url = (settings.llm_base_url or DEFAULT_BASE_URL).rstrip("/") + "/v1/messages"
        self._http = http or httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        self._headers = {
            "x-api-key": settings.llm_api_key.get_secret_value(),
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        self._throttle = throttle or CallThrottle(settings.llm_calls_per_minute)

    async def chat_json(
        self, *, system: str, user: str,
        schema: dict[str, Any] | None = None, model: str | None = None,
    ) -> LLMResponse:
        """Send one system + user message pair and return the JSON reply.

        `schema` is accepted for interface compatibility; the field contract is described in the
        prompt and enforced by our own validation.
        """
        payload: dict[str, Any] = {
            "model": model or self._settings.llm_model,
            "max_tokens": self._settings.llm_max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._settings.llm_send_temperature:
            payload["temperature"] = self._settings.llm_temperature

        async def call() -> LLMResponse:
            response = await self._http.post(self._url, json=payload, headers=self._headers)
            if response.status_code >= 400:
                raise UpstreamHTTPError(response.status_code, error_body(response))
            data = response.json()
            blocks = data.get("content") if isinstance(data, dict) else None
            if not isinstance(blocks, list):
                raise UpstreamHTTPError(502, {"message": "malformed response from the LLM provider"})
            text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
            usage = data.get("usage") or {}
            return LLMResponse(
                text=text,
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
            )

        await self._throttle.acquire()
        return await call_with_retries(call, service="Language model", settings=self._settings)
