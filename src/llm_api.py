import importlib
import logging
import math
import os
import re
import time
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse

from httpx import Client as HTTPClient, Response
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from openai import APIConnectionError

logger = logging.getLogger(__name__)

# Hosts that may receive an API key.  Extend with LLM_ALLOWED_HOSTS
# (comma-separated) or the allowed_hosts argument; plain http is accepted
# only for a local server (vLLM, Ollama) on a loopback address.
DEFAULT_ALLOWED_HOSTS = frozenset({
    "openrouter.ai",
    "api.groq.com",
    "api.together.xyz",
    "api.deepseek.com",
    "api.mistral.ai",
    "api.openai.com",
})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_CONFIG_NAME = re.compile(r"^llm_config[A-Za-z0-9_]*$")


class LLMResponseError(RuntimeError):
    """The provider answered 2xx with a body that is not a chat completion."""


class BudgetExceeded(RuntimeError):
    """The configured cap on API calls was reached; no further request is sent."""


def validate_endpoint(base_url: str, allowed_hosts) -> None:
    """Reject endpoints that would send the API key to an untrusted place."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and host in allowed_hosts:
        return
    if parsed.scheme == "http" and host in _LOOPBACK_HOSTS:
        return
    raise ValueError(
        f"Refusing to send an API key to {base_url!r}: need https and a host in the "
        "allow-list (add it via LLM_ALLOWED_HOSTS), or http on a loopback address."
    )


def api_key_from_env(model_name: str, configured: Optional[str], base_url: Optional[str] = None) -> Optional[str]:
    """Prefer the model-specific key, then the provider key, then the config value.

    The model name is upper-cased with non-alphanumerics replaced by '_',
    e.g. "qwen2.5-72b-instruct" -> LLM_API_KEY_QWEN2_5_72B_INSTRUCT.
    LLM_API_KEY is the OpenRouter key: it is never used for a model served by
    another provider, because a key must only reach the host it belongs to.
    """
    suffix = re.sub(r"[^A-Za-z0-9]", "_", model_name).upper()
    specific = os.environ.get(f"LLM_API_KEY_{suffix}")
    if specific:
        return specific
    host = (urlparse(base_url).hostname or "").lower() if base_url else ""
    if host == "api.groq.com" and os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    if host == "openrouter.ai" and os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    return configured


def import_config_module(config_name: str, package: Optional[str]):
    """Import a configuration module, but only one named llm_config*.

    The name comes from the command line, so an unrestricted import_module
    would execute any importable module.
    """
    if not _CONFIG_NAME.match(config_name):
        raise ValueError(f"Config module must be named llm_config*, got {config_name!r}")
    if package:
        try:
            return importlib.import_module(f".{config_name}", package=package)
        except ImportError:
            pass
    return importlib.import_module(config_name)


def _validate_chat_response(response: Response) -> None:
    """Check the wire response before LangChain normalizes missing content."""
    if response.has_redirect_location:
        raise LLMResponseError("Unexpected redirect from the LLM endpoint")
    if not response.is_success:
        return
    try:
        response.read()
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMResponseError("Provider response is not a chat completion") from None
    if not isinstance(content, str):
        raise LLMResponseError("Provider completion content is not text")


class LLMAPIClient:
    """LangChain client for endpoints implementing OpenAI Chat Completions.

    Parameters
    ----------
    endpoints : {model_name: base_url}
        Base URL before /chat/completions, e.g. "https://api.groq.com/openai/v1".
        Models served by the same provider can share a URL and key.
    api_keys : {model_name: key}
        Optional configured API keys per model, used after environment keys.
    model_aliases : {model_name: remote_model_id}, optional
        Maps internal model names to provider model IDs, e.g.
        {"llama3-8b-instruct": "meta-llama/llama-3-8b-instruct"}.
    max_tokens : int
        Max tokens per completion (default 512).
    timeout : int
        HTTP timeout in seconds (default 60).
    allowed_hosts : iterable of str, optional
        Hosts that may receive an API key, in addition to the defaults and
        the LLM_ALLOWED_HOSTS environment variable.
    max_calls : int, optional
        Hard cap on the number of requests sent; the next call raises
        BudgetExceeded.  Defaults to LLM_MAX_CALLS if set, else no cap.
    token_prices : {remote_model_id: (input_rate, output_rate)}, optional
        USD per million tokens. Missing prices or usage leave cost unknown.
    """

    def __init__(
        self,
        endpoints: Dict[str, str],
        api_keys: Dict[str, str],
        model_aliases: Optional[Dict[str, str]] = None,
        max_tokens: int = 512,
        max_tokens_per_model: Optional[Dict[str, int]] = None,
        timeout: int = 60,
        allowed_hosts=None,
        max_calls: Optional[int] = None,
        token_prices: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        hosts = set(DEFAULT_ALLOWED_HOSTS) | {h.strip().lower() for h in (allowed_hosts or ()) if h.strip()}
        hosts |= {h.strip().lower() for h in os.environ.get("LLM_ALLOWED_HOSTS", "").split(",") if h.strip()}
        self.allowed_hosts = frozenset(hosts)
        for model_name, url in endpoints.items():
            validate_endpoint(url, self.allowed_hosts)
        if max_calls is None and os.environ.get("LLM_MAX_CALLS"):
            try:
                max_calls = int(os.environ["LLM_MAX_CALLS"])
            except ValueError:
                raise ValueError(f"LLM_MAX_CALLS must be an integer, got {os.environ['LLM_MAX_CALLS']!r}") from None
        self.max_calls = max_calls
        self.calls_made = 0
        self.endpoints = endpoints
        self.api_keys = api_keys
        self.model_aliases = model_aliases or {}
        # Prices are USD per million input and output tokens, keyed by provider model ID.
        self.token_prices = {}
        for model_id, rates in (token_prices or {}).items():
            try:
                input_rate, output_rate = map(float, rates)
                if any(not math.isfinite(rate) or rate < 0 for rate in (input_rate, output_rate)):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"Invalid token prices for {model_id!r}: expected two non-negative rates") from None
            self.token_prices[model_id] = (input_rate, output_rate)
        self.max_tokens = max_tokens
        self.max_tokens_per_model = max_tokens_per_model or {}
        self.timeout = timeout

    def reserve_call(self) -> None:
        """Count an outgoing request against the budget shared by workers and judge."""
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise BudgetExceeded(
                f"API call budget of {self.max_calls} requests is exhausted "
                "(raise it with max_calls or LLM_MAX_CALLS)."
            )
        self.calls_made += 1

    def call(self, model_name: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """
        Send a chat completion request. Returns:
        Text, latency, model ID and optional usage and estimated USD cost.
        Raises ValueError if the model is not configured.
        Provider HTTP failures propagate as SDK errors.
        """
        base_url = self.endpoints.get(model_name)
        api_key = api_key_from_env(model_name, self.api_keys.get(model_name), base_url)
        if not base_url or not api_key:
            raise ValueError(
                f"Model '{model_name}' is not configured. "
                "Add it to LLM_ENDPOINTS and LLM_API_KEYS in llm_config.py."
            )

        validate_endpoint(base_url, self.allowed_hosts)
        remote_model = self.model_aliases.get(model_name, model_name)
        with HTTPClient(
            follow_redirects=False,
            event_hooks={"response": [_validate_chat_response]},
        ) as http_client:
            model = ChatOpenAI(
                model=remote_model,
                base_url=base_url.rstrip("/"),
                api_key=api_key,
                timeout=self.timeout,
                http_client=http_client,
                use_responses_api=False,
                max_retries=0,
                cache=False,
                # Preserve the Chat Completions token field used by existing endpoints.
                extra_body={"max_tokens": self.max_tokens_per_model.get(model_name, self.max_tokens)},
            )
            messages = [
                ("system", "You are a helpful assistant. Respond in plain text only. Do not call any functions or tools."),
                ("human", prompt),
            ]
            self.reserve_call()
            t0 = time.perf_counter()
            try:
                with tracing_context(enabled=False):
                    response = model.invoke(messages)
            except APIConnectionError as exc:
                if isinstance(exc.__cause__, LLMResponseError):
                    raise exc.__cause__ from None
                raise
            latency = time.perf_counter() - t0

        content = response.content
        if not isinstance(content, str):
            raise LLMResponseError(f"{model_name}: completion content is not text")
        usage = response.response_metadata.get("token_usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        rates = self.token_prices.get(remote_model)
        cost = None
        if rates is not None and input_tokens is not None and output_tokens is not None:
            cost = (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
        return {
            "output": content,
            "success": True,
            "latency": latency,
            "model_name": response.response_metadata.get("model_name") or remote_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        }
