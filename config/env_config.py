from __future__ import annotations

import os
from dotenv import load_dotenv


# Cached environment variables for this process
ENV_CONFIG: dict[str, str] = {}

# Ensures configuration is initialized only once
_initialized = False


def init_config() -> dict[str, str]:
    """
    Load environment variables into ENV_CONFIG once per process.

    Loads variables from `.env` (if present) and merges them with
    `os.environ`. Subsequent calls return the cached result.
    """
    global _initialized

    if _initialized:
        return ENV_CONFIG

    load_dotenv()
    ENV_CONFIG.clear()
    ENV_CONFIG.update(os.environ)
    _initialized = True
    return ENV_CONFIG


def get_env(
    key: str,
    default: str | None = None,
    *,
    required: bool = False
) -> str | None:
    """
    Get a single environment variable.

    Parameters
    ----------
    key : variable name
    default : value returned if key is missing
    required : raise ValueError if missing or empty

    Returns
    -------
    str | None
    """
    init_config()
    value = ENV_CONFIG.get(key, default)

    if required and (value is None or value == ""):
        raise ValueError(f"Missing required environment variable: {key}")

    return value


def get_env_int_list(
    key: str,
    *,
    default: list[int] | None = None,
    separator: str = ",",
    required: bool = False,
) -> list[int]:
    """
    Parse an environment variable as a list of integers.

    Expected format:
        KEY=1,2,3

    Returns default or [] if empty.
    """
    raw_value = get_env(key, required=required)

    if raw_value is None or raw_value.strip() == "":
        return default if default is not None else []

    try:
        return [
            int(item.strip())
            for item in raw_value.split(separator)
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {key} must be a comma-separated int list"
        ) from exc


def get_env_map() -> dict[str, str]:
    """
    Return the environment configuration.
    """
    init_config()
    return ENV_CONFIG