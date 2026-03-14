from __future__ import annotations

from dotenv import load_dotenv
from google import genai

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
        load_dotenv()
        _client = genai.Client()

    return _client




__all__ = [
    "LIGHT_MODEL",
    "BALANCED_MODEL",
    "LARGE_MODEL",
    "gemini_model",
    "get_client",
]
