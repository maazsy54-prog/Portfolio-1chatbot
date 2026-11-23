"""LLM client for any OpenAI-compatible chat API (Groq, Google Gemini, OpenRouter, OpenAI, ...).

Same interface as the Cohere LLMClient, so generation and the evaluation judge do not care which
provider answers. Timeouts, retries and pacing come from the shared wrappers.
"""
from typing import Any

import httpx

from app.clients.cohere_client import LLMResponse
from app.clients.resilience import call_with_retries
from app.clients.throttle import CallThrottle
from app.config import Settings


class UpstreamHTTPError(Exception):
    """An HTTP error from the LLM provider. Exposes `status_code` and `body` so the shared
    retry logic and the error log can read them (same shape as the Cohere SDK's errors)."""

    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        super().__init__(body.get("message", f"HTTP {status_code}"))
        self.status_code = status_code
        self.body = body


def _error_body(response: httpx.Response) -> dict[str, Any]:
    """Extract the provider's message from an error response (OpenAI-style or plain)."""
    try:
        data = response.json()
        error = data.get("error", data) if isinstance(data, dict) else {}
        message = error.get("message") if isinstance(error, dict) else str(error)
        return {"message": message or response.text[:300]}
    except ValueError:
        return {"message": response.text[:300]}


class OpenAICompatLLM:
    """Chat completions over HTTP, returning the model's JSON text and token usage."""

    def __init__(
        self, settings: Settings, http: httpx.AsyncClient | None = None, throttle: Any | None = None
    ) -> None:
        self._settings = settings
        self._url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        self._http = http or httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        self._headers = {"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"}
        self._throttle = throttle or CallThrottle(settings.llm_calls_per_minute)

    async def chat_json(
        self, *, system: str, user: str,
        schema: dict[str, Any] | None = None, model: str | None = None,
    ) -> LLMResponse:
        """Send one system + user message pair and return the JSON reply.

        `schema` is accepted for interface compatibility; OpenAI-style JSON mode only guarantees
        valid JSON, so the field contract is enforced by the prompt and by our own validation.
        """
        payload: dict[str, Any] = {
            "model": model or self._settings.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
        }
        if self._settings.llm_json_mode:
            payload["response_format"] = {"type": "json_object"}

        async def call() -> LLMResponse:
            response = await self._http.post(self._url, json=payload, headers=self._headers)
            if response.status_code >= 400:
                raise UpstreamHTTPError(response.status_code, _error_body(response))
            data = response.json()
            try:
                text = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                raise UpstreamHTTPError(502, {"message": "malformed response from the LLM provider"})
            usage = data.get("usage") or {}
            return LLMResponse(
                text=text,
                input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
            )

        await self._throttle.acquire()
        return await call_with_retries(call, service="Language model", settings=self._settings)
