# llm_config_example.py — скопируйте в llm_config.py (файл в .gitignore)
#
# КЛЮЧИ ЛУЧШЕ НЕ ХРАНИТЬ В ФАЙЛЕ: задайте их переменными окружения, они имеют
# приоритет над значениями ниже. Имя модели переводится в верхний регистр, всё
# кроме букв и цифр заменяется на «_»:
#   export LLM_API_KEY_LLAMA3_8B_INSTRUCT=...      # ключ одной модели
#   export LLM_API_KEY=...                         # ключ OpenRouter: только для моделей на openrouter.ai
#                                                  # (другим провайдерам он не отправляется)
# Ключ отправляется только на https-адреса из списка разрешённых хостов
# (openrouter.ai, api.groq.com, api.together.xyz, api.deepseek.com, api.mistral.ai,
# api.openai.com) или на http://localhost. Другой хост: export LLM_ALLOWED_HOSTS=host1,host2
# Потолок платных запросов: export LLM_MAX_CALLS=2000 (или --max-api-calls).
#
# Выберите один из двух вариантов (или смешайте):
#   ВАРИАНТ А — OpenRouter: один ключ, все модели
#   ВАРИАНТ Б — Отдельные провайдеры: свой ключ для каждой модели

# ===========================================================================
# ВАРИАНТ А: OpenRouter (https://openrouter.ai)
# Один ключ — все модели. Проще всего для старта.
# Зарегистрируйтесь на openrouter.ai, создайте ключ в Settings → API Keys.
# ===========================================================================

# LLM_ENDPOINTS = {
#     "llama3-8b-instruct":    "https://openrouter.ai/api/v1",
#     "deepseek-coder-v2":     "https://openrouter.ai/api/v1",
#     "qwen2.5-72b-instruct":  "https://openrouter.ai/api/v1",
#     "mistral-7b-instruct":   "https://openrouter.ai/api/v1",
#     "qwen2.5-tooluse":       "https://openrouter.ai/api/v1",
# }
#
# _OR_KEY = "sk-or-v1-ВАШ_КЛЮЧ_ЗДЕСЬ"
# LLM_API_KEYS = {k: _OR_KEY for k in LLM_ENDPOINTS}
#
# LLM_MODEL_ALIASES = {
#     "llama3-8b-instruct":   "meta-llama/llama-3-8b-instruct",
#     "deepseek-coder-v2":    "deepseek/deepseek-coder",
#     "qwen2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct",
#     "mistral-7b-instruct":  "mistralai/mistral-7b-instruct",
#     "qwen2.5-tooluse":      "qwen/qwen-2.5-72b-instruct",
# }
#
# JUDGE_MODEL_NAME = "qwen2.5-72b-instruct"

# ===========================================================================
# ВАРИАНТ Б: Отдельные провайдеры
#
# Groq     — https://console.groq.com  (бесплатный tier, Llama + Mistral)
# DeepSeek — https://platform.deepseek.com
# Together — https://api.together.ai   (Qwen)
# Mistral  — https://console.mistral.ai
# ===========================================================================

LLM_ENDPOINTS = {
    # Groq — llama3-8b-8192 (бесплатно, быстро)
    "llama3-8b-instruct":   "https://api.groq.com/openai/v1",
    # DeepSeek API
    "deepseek-coder-v2":    "https://api.deepseek.com/v1",
    # Together AI — Qwen 2.5 72B
    "qwen2.5-72b-instruct": "https://api.together.xyz/v1",
    # Groq — Mistral 7B
    "mistral-7b-instruct":  "https://api.groq.com/openai/v1",
    # Together AI — Qwen 2.5 Turbo (ближайший к ToolUse)
    "qwen2.5-tooluse":      "https://api.together.xyz/v1",
}

LLM_API_KEYS = {
    "llama3-8b-instruct":   "gsk_ВАШ_GROQ_КЛЮЧ",
    "deepseek-coder-v2":    "sk-ВАШ_DEEPSEEK_КЛЮЧ",
    "qwen2.5-72b-instruct": "ВАШ_TOGETHER_КЛЮЧ",
    "mistral-7b-instruct":  "gsk_ВАШ_GROQ_КЛЮЧ",
    "qwen2.5-tooluse":      "ВАШ_TOGETHER_КЛЮЧ",
}

# Маппинг внутренних имён на ID моделей у провайдера
LLM_MODEL_ALIASES = {
    "llama3-8b-instruct":   "llama3-8b-8192",                      # Groq model ID
    "deepseek-coder-v2":    "deepseek-coder",                       # DeepSeek API
    "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct-Turbo",     # Together AI
    "mistral-7b-instruct":  "mixtral-8x7b-32768",                   # Groq (Mistral-class)
    "qwen2.5-tooluse":      "Qwen/Qwen2.5-72B-Instruct-Turbo",     # Together AI
}

# Модель-судья для оценки качества ответов
JUDGE_MODEL_NAME = "qwen2.5-72b-instruct"
