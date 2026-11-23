"""OpenAI-compatible LLM client: request shape, retries, errors and provider selection."""
import json

import httpx
import pytest

from app.clients.cohere_client import LLMClient
from app.clients.llm import make_llm
from app.clients.llm_openai import OpenAICompatLLM
from app.config import Settings
from app.errors import ServiceUnavailableError

SETTINGS = Settings(
    _env_file=None, llm_provider="openai_compatible", llm_base_url="https://api.example.test/openai/v1/",
    llm_api_key="sk-test-key-123456", llm_model="llama-test", retry_backoff_seconds=0.0, max_retries=3,
)
GOOD = {"choices": [{"message": {"content": '{"answerable": true, "answer": "ok [1]", "citations": [1]}'}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 15}}


def client_for(handler, settings=SETTINGS, **kw) -> OpenAICompatLLM:
    return OpenAICompatLLM(settings, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw)


async def test_request_shape_and_response_parsing() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(url=str(request.url), auth=request.headers["authorization"], body=json.loads(request.content))
        return httpx.Response(200, json=GOOD)

    reply = await client_for(handler).chat_json(system="SYS", user="USR")
    assert seen["url"] == "https://api.example.test/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test-key-123456"
    body = seen["body"]
    assert body["model"] == "llama-test" and body["temperature"] == 0.0
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]
    assert body["response_format"] == {"type": "json_object"}
    assert reply.text.startswith('{"answerable"') and (reply.input_tokens, reply.output_tokens) == (120, 15)


async def test_model_override_for_the_judge_and_json_mode_off() -> None:
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=GOOD)

    await client_for(handler).chat_json(system="s", user="u", model="judge-model")
    plain = SETTINGS.model_copy(update={"llm_json_mode": False})
    await client_for(handler, plain).chat_json(system="s", user="u")
    assert bodies[0]["model"] == "judge-model"
    assert "response_format" not in bodies[1]


async def test_rate_limit_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=GOOD)

    reply = await client_for(handler).chat_json(system="s", user="u")
    assert calls["n"] == 3 and reply.text


async def test_auth_error_fails_fast_as_503_and_logs_the_providers_message(caplog) -> None:
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

    with caplog.at_level("ERROR"):
        with pytest.raises(ServiceUnavailableError):
            await client_for(handler).chat_json(system="s", user="u")
    assert calls["n"] == 1 and "Invalid API Key" in caplog.text and "status=401" in caplog.text


async def test_malformed_provider_reply_is_a_503() -> None:
    def handler(request):
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(ServiceUnavailableError):
        await client_for(handler).chat_json(system="s", user="u")


async def test_calls_are_paced_through_the_throttle() -> None:
    class Counting:
        n = 0

        async def acquire(self):
            Counting.n += 1

    await client_for(lambda r: httpx.Response(200, json=GOOD), throttle=Counting()).chat_json(system="s", user="u")
    assert Counting.n == 1


def test_factory_picks_the_provider() -> None:
    assert isinstance(make_llm(Settings(_env_file=None)), LLMClient)
    assert isinstance(make_llm(SETTINGS), OpenAICompatLLM)
    assert isinstance(make_llm(SETTINGS.model_copy(update={"llm_provider": "OpenAI"})), OpenAICompatLLM)


def test_factory_explains_missing_or_unknown_configuration() -> None:
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        make_llm(Settings(_env_file=None, llm_provider="openai_compatible"))
    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        make_llm(Settings(_env_file=None, llm_provider="nonsense"))


def test_llm_api_key_is_treated_as_a_secret() -> None:
    assert "sk-test-key-123456" in SETTINGS.secret_values()
    assert "sk-test-key-123456" not in repr(SETTINGS)
