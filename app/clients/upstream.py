"""Shared HTTP error type for providers we call with plain httpx."""
from typing import Any

import httpx


class UpstreamHTTPError(Exception):
    """An HTTP error from an upstream API. Exposes `status_code` and `body` so the shared retry
    logic and the error log can read them (same shape as the Cohere SDK's errors)."""

    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        super().__init__(body.get("message", f"HTTP {status_code}"))
        self.status_code = status_code
        self.body = body


def error_body(response: httpx.Response) -> dict[str, Any]:
    """Extract the provider's message from an error response (OpenAI- or Anthropic-style, or plain)."""
    try:
        data = response.json()
        error = data.get("error", data) if isinstance(data, dict) else {}
        message = error.get("message") if isinstance(error, dict) else str(error)
        return {"message": message or response.text[:300]}
    except ValueError:
        return {"message": response.text[:300]}
