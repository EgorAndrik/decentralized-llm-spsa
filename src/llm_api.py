import importlib
import logging
import os
import re
import time
import requests
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse

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
    """Key for a model, in order: LLM_API_KEY_<MODEL>, LLM_API_KEY (OpenRouter models only), the config value.

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


class LLMAPIClient:
    """
    OpenAI-compatible chat completions client.

    Works with any provider that implements /v1/chat/completions:
      - OpenRouter  (one key for all models)
      - Groq        (Llama, Mistral — free tier available)
      - Together AI (Qwen, Llama, DeepSeek)
      - DeepSeek    (deepseek-coder, deepseek-chat)
      - Mistral AI  (mistral-7b, mixtral)

    Parameters
    ----------
    endpoints : {model_name: base_url}
        Base URL up to /v1, e.g. "https://openrouter.ai/api/v1".
        For OpenRouter you can point every model at the same URL.
    api_keys : {model_name: key}
        API keys per model. For OpenRouter all keys are identical.
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
    ):
        hosts = set(DEFAULT_ALLOWED_HOSTS) | {h.strip().lower() for h in (allowed_hosts or ()) if h.strip()}
        hosts |= {h.strip().lower() for h in os.environ.get("LLM_ALLOWED_HOSTS", "").split(",") if h.strip()}
        self.allowed_hosts = frozenset(hosts)
        for model_name, url in endpoints.items():
            validate_endpoint(url, self.allowed_hosts)
        if max_calls is None and os.environ.get("LLM_MAX_CALLS"):
            max_calls = int(os.environ["LLM_MAX_CALLS"])
        self.max_calls = max_calls
        self.calls_made = 0
        self.endpoints = endpoints
        self.api_keys = api_keys
        self.model_aliases = model_aliases or {}
        self.max_tokens = max_tokens
        self.max_tokens_per_model = max_tokens_per_model or {}
        self.timeout = timeout

    def call(self, model_name: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """
        Send a chat completion request. Returns:
            {"output": str, "success": bool, "latency": float}
        Raises ValueError if the model is not configured.
        Raises requests.HTTPError on non-2xx responses.
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
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": remote_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Respond in plain text only. Do not call any functions or tools."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens_per_model.get(model_name, self.max_tokens),
        }

        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise BudgetExceeded(
                f"API call budget of {self.max_calls} requests is exhausted "
                "(raise it with max_calls or LLM_MAX_CALLS)."
            )
        self.calls_made += 1

        t0 = time.perf_counter()
        # Redirects are not followed: the Authorization header must not leave the validated host.
        response = requests.post(
            url, json=payload, headers=headers, timeout=self.timeout, allow_redirects=False,
        )
        latency = time.perf_counter() - t0

        response.raise_for_status()
        if response.is_redirect:
            raise LLMResponseError(f"{model_name}: unexpected redirect to another location")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"{model_name}: response is not a chat completion ({type(exc).__name__})") from None
        if not isinstance(content, str):
            raise LLMResponseError(f"{model_name}: completion content is {type(content).__name__}, not text")
        return {
            "output": content,
            "success": True,
            "latency": latency,
        }
