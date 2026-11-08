"""Copy to llm_config.py and set GROQ_API_KEY for the two-agent demo."""

BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL_NAME = "judge"

# The two worker keys are the first two agent profile names used by the runner.
# Values are the model IDs accepted by the configured provider.
LLM_MODEL_ALIASES = {
    "llama3-8b-instruct": "openai/gpt-oss-20b",
    "deepseek-coder-v2": "openai/gpt-oss-120b",
    # Demo only: reuse a worker model to verify the judge connection.
    JUDGE_MODEL_NAME: "openai/gpt-oss-120b",
}
LLM_ENDPOINTS = dict.fromkeys(LLM_MODEL_ALIASES, BASE_URL)

# Groq standard rates in USD per million input/output tokens.
# https://console.groq.com/docs/models
LLM_TOKEN_PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}

# Groq uses GROQ_API_KEY. For other providers set LLM_API_KEY_<INTERNAL_NAME>
# (upper-case, with punctuation replaced by underscores), or populate
# LLM_API_KEYS from a provider-specific environment variable using os.getenv.
LLM_API_KEYS = {}

# Optional web search worker: set TAVILY_API_KEY in the environment.
