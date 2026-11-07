import os
import unittest
from unittest import mock

import requests

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
    """Run each test without LLM_* variables, so the result does not depend on the caller's environment.

    CI sets LLM_MAX_CALLS=0 as a safety net against real calls; tests that
    drive a stubbed client must not inherit it.
    """

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in [k for k in os.environ if k.startswith("LLM_")]:
            del os.environ[key]


GOOD = {"m": "https://api.groq.com/openai/v1"}
KEYS = {"m": "k-config"}


class FakeResponse:
    def __init__(self, body=None, status=200, redirect=False):
        self._body, self.status_code, self.is_redirect = body, status, redirect

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def ok_body(text="hi"):
    return {"choices": [{"message": {"content": text}}]}


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

        def fake_post(url, json=None, headers=None, **kwargs):
            seen[url] = headers["Authorization"]
            return FakeResponse(ok_body())

        with mock.patch.dict(os.environ, {"LLM_API_KEY": "or-key"}), \
                mock.patch.object(requests, "post", side_effect=fake_post):
            client.call("a", "p", "t")
            client.call("b", "p", "t")
        self.assertEqual(seen[self.OPENROUTER + "/chat/completions"], "Bearer or-key")
        self.assertEqual(seen[self.GROQ + "/chat/completions"], "Bearer groq-key")

    def test_environment_key_is_sent_in_header(self):
        client = LLMAPIClient(GOOD, KEYS)
        with mock.patch.dict(os.environ, {"LLM_API_KEY_M": "from-env"}), \
                mock.patch.object(requests, "post", return_value=FakeResponse(ok_body())) as post:
            client.call("m", "p", "t")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer from-env")
        self.assertFalse(post.call_args.kwargs["allow_redirects"])


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
        with mock.patch.object(requests, "post", return_value=FakeResponse(ok_body())) as post:
            client.call("m", "p", "t")
            client.call("m", "p", "t")
            with self.assertRaises(BudgetExceeded):
                client.call("m", "p", "t")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(client.calls_made, 2)

    def test_budget_from_environment(self):
        with mock.patch.dict(os.environ, {"LLM_MAX_CALLS": "1"}):
            client = LLMAPIClient(GOOD, KEYS)
        self.assertEqual(client.max_calls, 1)

    def test_malformed_responses_raise_clear_error(self):
        client = LLMAPIClient(GOOD, KEYS)
        for body in ({}, {"choices": []}, {"choices": [{"message": {"content": None}}]},
                     {"choices": [{"message": {"content": 5}}]}, ValueError("not json"), "text"):
            with mock.patch.object(requests, "post", return_value=FakeResponse(body)):
                with self.assertRaises(LLMResponseError, msg=repr(body)):
                    client.call("m", "p", "t")

    def test_redirect_is_not_followed(self):
        client = LLMAPIClient(GOOD, KEYS)
        with mock.patch.object(requests, "post", return_value=FakeResponse({}, status=302, redirect=True)):
            with self.assertRaises(LLMResponseError):
                client.call("m", "p", "t")

    def test_budget_exceeded_is_not_swallowed_by_callers(self):
        from src.run_real_llm_comparison import PromptCachingLLMClient

        class Raw:
            def call(self, *a):
                raise BudgetExceeded("cap")

        with self.assertRaises(BudgetExceeded):
            PromptCachingLLMClient(Raw()).call("m", "p", "t")


if __name__ == "__main__":
    unittest.main()
