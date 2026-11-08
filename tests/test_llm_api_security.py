import json
import os
import unittest
from unittest import mock

import httpx
from openai import RateLimitError

from src import llm_api
from src.llm_api import (
    BudgetExceeded,
    LLMAPIClient,
    LLMResponseError,
    api_key_from_env,
    import_config_module,
    validate_endpoint,
)

class EnvIsolated(unittest.TestCase):
    """Run each test without API configuration inherited from the environment.

    CI sets LLM_MAX_CALLS=0 as a safety net against real calls; tests that
    drive a stubbed client must not inherit it.
    """

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in [k for k in os.environ if k.startswith("LLM_") or k == "GROQ_API_KEY"]:
            del os.environ[key]

    def mock_http(self, handler):
        def client(**kwargs):
            return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)

        return mock.patch.object(llm_api, "HTTPClient", side_effect=client)


GOOD = {"m": "https://api.groq.com/openai/v1"}
KEYS = {"m": "k-config"}


def ok_response(_request):
    return httpx.Response(200, json={
        "id": "completion",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi"},
        }],
    })


class TestEndpointValidation(EnvIsolated):
    def test_rejects_plain_http_and_unknown_hosts(self):
        for url in ("http://api.groq.com/v1", "https://evil.example/v1",
                    "https://api.groq.com.evil.example/v1", "ftp://api.groq.com", "api.groq.com/v1"):
            with self.assertRaises(ValueError, msg=url):
                validate_endpoint(url, llm_api.DEFAULT_ALLOWED_HOSTS)

    def test_accepts_https_allowlist_and_loopback_http(self):
        for url in ("https://api.groq.com/openai/v1", "https://openrouter.ai/api/v1",
                    "http://localhost:8000/v1", "http://127.0.0.1:11434/v1"):
            validate_endpoint(url, llm_api.DEFAULT_ALLOWED_HOSTS)

    def test_client_refuses_bad_endpoint_at_construction(self):
        with self.assertRaises(ValueError):
            LLMAPIClient({"m": "http://api.groq.com/v1"}, KEYS)

    def test_extra_hosts_via_argument_and_environment(self):
        LLMAPIClient({"m": "https://llm.example.org/v1"}, KEYS, allowed_hosts=["llm.example.org"])
        with mock.patch.dict(os.environ, {"LLM_ALLOWED_HOSTS": "llm.example.org, other.org"}):
            LLMAPIClient({"m": "https://llm.example.org/v1"}, KEYS)


class TestKeysFromEnvironment(EnvIsolated):
    OPENROUTER = "https://openrouter.ai/api/v1"
    GROQ = "https://api.groq.com/openai/v1"

    def test_precedence(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(api_key_from_env("qwen2.5-72b", "cfg", self.GROQ), "cfg")
        with mock.patch.dict(os.environ, {"LLM_API_KEY_QWEN2_5_72B": "specific", "LLM_API_KEY": "or-key"}, clear=True):
            self.assertEqual(api_key_from_env("qwen2.5-72b", "cfg", self.GROQ), "specific")
            self.assertEqual(api_key_from_env("qwen2.5-72b", "cfg", self.OPENROUTER), "specific")

    def test_generic_key_is_only_for_openrouter(self):
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "or-key"}, clear=True):
            self.assertEqual(api_key_from_env("m", "cfg", self.OPENROUTER), "or-key")
            # another provider keeps its own key and never receives the OpenRouter one
            self.assertEqual(api_key_from_env("m", "groq-key", self.GROQ), "groq-key")
            self.assertEqual(api_key_from_env("m", "groq-key", "https://api.deepseek.com/v1"), "groq-key")
            self.assertIsNone(api_key_from_env("m", None, self.GROQ))
            self.assertIsNone(api_key_from_env("m", None))

    def test_generic_key_not_sent_to_other_provider(self):
        client = LLMAPIClient({"a": self.OPENROUTER, "b": self.GROQ}, {"a": "x", "b": "groq-key"})
        seen = {}

        def handle(request):
            seen[str(request.url)] = request.headers["Authorization"]
            return ok_response(request)

        with mock.patch.dict(os.environ, {"LLM_API_KEY": "or-key"}), self.mock_http(handle):
            client.call("a", "p", "t")
            client.call("b", "p", "t")
        self.assertEqual(seen[self.OPENROUTER + "/chat/completions"], "Bearer or-key")
        self.assertEqual(seen[self.GROQ + "/chat/completions"], "Bearer groq-key")

    def test_environment_key_and_model_alias_are_sent_to_chat_endpoint(self):
        client = LLMAPIClient(GOOD, KEYS, model_aliases={"m": "vendor/model"}, max_tokens_per_model={"m": 100})
        handler = mock.Mock(side_effect=ok_response)
        with mock.patch.dict(os.environ, {"LLM_API_KEY_M": "from-env"}), self.mock_http(handler):
            result = client.call("m", "p", "t")
        request = handler.call_args.args[0]
        self.assertEqual(str(request.url), self.GROQ + "/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer from-env")
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], "vendor/model")
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "p"})
        self.assertEqual(result["output"], "hi")
        self.assertTrue(result["success"])

    def test_groq_key_is_only_used_for_groq(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "groq-key"}):
            self.assertEqual(api_key_from_env("m", None, self.GROQ), "groq-key")
            self.assertIsNone(api_key_from_env("m", None, self.OPENROUTER))
            self.assertIsNone(api_key_from_env("m", None, "https://api.deepseek.com/v1"))


class TestConfigModuleAllowlist(EnvIsolated):
    def test_only_llm_config_modules(self):
        for name in ("os", "subprocess", "../llm_config", "llm_config.__init__", "my_llm_config", "llm_config; x"):
            with self.assertRaises(ValueError, msg=name):
                import_config_module(name, "src")

    def test_valid_name_reaches_import(self):
        with self.assertRaises(ImportError):
            import_config_module("llm_config_does_not_exist", "src")


class TestBudgetAndResponses(EnvIsolated):
    def test_budget_stops_before_the_request(self):
        client = LLMAPIClient(GOOD, KEYS, max_calls=2)
        handler = mock.Mock(side_effect=ok_response)
        with self.mock_http(handler):
            client.call("m", "p", "t")
            client.call("m", "p", "t")
            with self.assertRaises(BudgetExceeded):
                client.call("m", "p", "t")
        self.assertEqual(handler.call_count, 2)
        self.assertEqual(client.calls_made, 2)

    def test_budget_from_environment(self):
        with mock.patch.dict(os.environ, {"LLM_MAX_CALLS": "1"}):
            client = LLMAPIClient(GOOD, KEYS)
        self.assertEqual(client.max_calls, 1)

    def test_malformed_responses_raise_clear_error(self):
        client = LLMAPIClient(GOOD, KEYS)
        for body in (b'{}', b'{"choices":[]}', b'{"choices":[{"message":{"content":null}}]}',
                     b'{"choices":[{"message":{"content":5}}]}', b"not json", b'"text"'):
            with self.mock_http(lambda request: httpx.Response(200, content=body)):
                with self.assertRaises(LLMResponseError, msg=repr(body)):
                    client.call("m", "p", "t")

    def test_redirect_is_not_followed(self):
        client = LLMAPIClient(GOOD, KEYS)
        handler = mock.Mock(return_value=httpx.Response(302, headers={"Location": "https://evil.example/"}))
        with self.mock_http(handler):
            with self.assertRaises(LLMResponseError):
                client.call("m", "p", "t")
        handler.assert_called_once()

    def test_sdk_does_not_retry_outside_the_request_budget(self):
        client = LLMAPIClient(GOOD, KEYS, max_calls=1)
        handler = mock.Mock(return_value=httpx.Response(429, json={"error": {"message": "Rate limited"}}))
        with self.mock_http(handler):
            with self.assertRaises(RateLimitError):
                client.call("m", "p", "t")
            with self.assertRaises(BudgetExceeded):
                client.call("m", "p", "t")
        handler.assert_called_once()

    def test_budget_exceeded_is_not_swallowed_by_callers(self):
        from src.run_real_llm_comparison import PromptCachingLLMClient

        class Raw:
            def call(self, *a):
                raise BudgetExceeded("cap")

        with self.assertRaises(BudgetExceeded):
            PromptCachingLLMClient(Raw()).call("m", "p", "t")


if __name__ == "__main__":
    unittest.main()
