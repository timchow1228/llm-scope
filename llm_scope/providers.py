"""
llm-scope providers — Provider registry, model routing, and cost calculation.
"""

import os
from typing import Optional


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "supports_stream_usage": True,
        "env_key": "DEEPSEEK_API_KEY",
        "models": {
            "deepseek-chat": {
                "input_per_1m": 0.14,
                "cache_per_1m": 0.014,
                "output_per_1m": 0.28,
                "context_window": 1_000_000,
            },
            "deepseek-reasoner": {
                "input_per_1m": 0.14,
                "cache_per_1m": 0.014,
                "output_per_1m": 0.28,
                "context_window": 1_000_000,
            },
            "deepseek-v4-flash": {
                "input_per_1m": 0.14,
                "cache_per_1m": 0.028,
                "output_per_1m": 0.28,
                "context_window": 1_000_000,
            },
            "deepseek-v4-pro": {
                "input_per_1m": 0.435,
                "cache_per_1m": 0.03625,
                "output_per_1m": 0.87,
                "context_window": 1_000_000,
            },
        },
    },
    "openai": {
        "base_url": "https://api.openai.com",
        "supports_stream_usage": True,
        "env_key": "OPENAI_API_KEY",
        "models": {
            "gpt-4o": {
                "input_per_1m": 2.50,
                "output_per_1m": 10.00,
                "context_window": 128_000,
            },
            "gpt-4o-mini": {
                "input_per_1m": 0.15,
                "output_per_1m": 0.60,
                "context_window": 128_000,
            },
        },
    },
    "groq": {
        "base_url": "https://api.groq.com/openai",
        "supports_stream_usage": False,
        "env_key": "GROQ_API_KEY",
        "models": {
            "llama-3.3-70b-versatile": {
                "input_per_1m": 0.59,
                "output_per_1m": 0.79,
                "context_window": 128_000,
            },
            "mixtral-8x7b-32768": {
                "input_per_1m": 0.24,
                "output_per_1m": 0.24,
                "context_window": 32_768,
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Local model detection
# ---------------------------------------------------------------------------

LOCAL_INDICATORS = [
    "localhost", "127.0.0.1", "0.0.0.0",
    "192.168.", "10.", ":11434", ":8000",
]


def is_local_provider(base_url: str) -> bool:
    """Check if the base_url points to a local model server."""
    return any(ind in base_url for ind in LOCAL_INDICATORS)


# ---------------------------------------------------------------------------
# Pricing defaults
# ---------------------------------------------------------------------------

LOCAL_PRICING = {"input_per_1m": 0.0, "output_per_1m": 0.0}
DEFAULT_PRICING = {"input_per_1m": 1.00, "output_per_1m": 3.00}

# ---------------------------------------------------------------------------
# Model name → prefix mapping for fallback inference
# ---------------------------------------------------------------------------

_PREFIX_MAP: dict[str, str] = {
    "gpt-": "openai",
    "deepseek-": "deepseek",
    "llama-": "groq",
    "mixtral-": "groq",
}


# ---------------------------------------------------------------------------
# Core routing function
# ---------------------------------------------------------------------------

def resolve_provider(model: str) -> tuple[str, str, dict]:
    """
    Resolve a model name to (provider_name, base_url, model_config).

    Priority:
      1. Exact match in PROVIDERS[*]["models"]
      2. Prefix inference (gpt- → openai, deepseek- → deepseek, etc.)
      3. DEVSCOPE_BASE_URL env var → detect local or use as custom upstream
      4. DEVSCOPE_PROVIDER env var fallback (default: deepseek)

    Returns:
        (provider_name, base_url, model_config)
        model_config includes input_per_1m, output_per_1m, context_window.
        Falls back to DEFAULT_PRICING if model is unknown.
    """
    # ── 1. Exact match ──
    for provider_name, config in PROVIDERS.items():
        if model in config["models"]:
            return (
                provider_name,
                config["base_url"],
                config["models"][model],
            )

    # ── 2. Prefix inference ──
    for prefix, provider_name in _PREFIX_MAP.items():
        if model.startswith(prefix) and provider_name in PROVIDERS:
            config = PROVIDERS[provider_name]
            print(f"⚠️  Unknown model '{model}', matched by prefix to {provider_name}")
            return (
                provider_name,
                config["base_url"],
                DEFAULT_PRICING,
            )

    # ── 3. DEVSCOPE_BASE_URL ──
    custom_base = os.environ.get("DEVSCOPE_BASE_URL")
    if custom_base:
        if is_local_provider(custom_base):
            print(f"⚠️  Unknown model '{model}', routing to local: {custom_base}")
            return ("local", custom_base, LOCAL_PRICING)
        else:
            print(f"⚠️  Unknown model '{model}', routing to custom: {custom_base}")
            return ("custom", custom_base, DEFAULT_PRICING)

    # ── 4. Fallback to DEVSCOPE_PROVIDER ──
    default_provider = os.environ.get("DEVSCOPE_PROVIDER", "deepseek")
    if default_provider in PROVIDERS:
        config = PROVIDERS[default_provider]
        print(f"⚠️  Unknown model '{model}', falling back to {default_provider}")
        return (
            default_provider,
            config["base_url"],
            DEFAULT_PRICING,
        )

    # Ultimate fallback (should never reach here)
    print(f"⚠️  Unknown model '{model}', no valid provider found, using deepseek")
    return ("deepseek", PROVIDERS["deepseek"]["base_url"], DEFAULT_PRICING)


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def calc_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_config: dict,
    cached_tokens: int = 0,
) -> float:
    """
    Calculate the USD cost of a single API call, considering cached tokens.
    """
    input_price = model_config.get("input_per_1m", DEFAULT_PRICING["input_per_1m"])
    output_price = model_config.get("output_per_1m", DEFAULT_PRICING["output_per_1m"])
    cache_price = model_config.get("cache_per_1m", input_price)

    # In DeepSeek and others, prompt_tokens is the total input tokens.
    uncached_tokens = max(0, prompt_tokens - cached_tokens)
    
    cost = (
        uncached_tokens * input_price +
        cached_tokens * cache_price +
        completion_tokens * output_price
    ) / 1_000_000
    
    return cost


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def get_api_key(provider_name: str) -> Optional[str]:
    """
    Get the API key for a given provider from environment variables.

    Returns None if not found.
    """
    if provider_name in PROVIDERS:
        env_var = PROVIDERS[provider_name].get("env_key")
        if env_var:
            return os.environ.get(env_var)

    # Also check generic DEVSCOPE_API_KEY
    return os.environ.get("DEVSCOPE_API_KEY")
