from __future__ import annotations

from google import genai

from config.env_config import get_env, init_config
from .model_selector import StaticModelSelector


LIGHT_MODEL = "gemini-3.1-flash-lite-preview"
BALANCED_MODEL = "gemini-3-flash-preview"
LARGE_MODEL = "gemini-3.1-pro-preview"

_client: genai.Client | None = None



gemini_model = StaticModelSelector(
    balanced_model=BALANCED_MODEL,
    large_model=LARGE_MODEL,
    light_model=LIGHT_MODEL
)


def get_client() -> genai.Client:
    global _client

    if _client is None:
        init_config()
        api_key = get_env("GEMINI_API_KEY", required=True)
        _client = genai.Client(api_key=api_key)

    return _client




__all__ = [
    "LIGHT_MODEL",
    "BALANCED_MODEL",
    "LARGE_MODEL",
    "gemini_model",
    "get_client",
]
