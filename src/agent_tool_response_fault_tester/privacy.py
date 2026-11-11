"""Small key-based redaction for exported trace artifacts."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_SENSITIVE_KEYS = {
    "authorization",
    "access_token",
    "refresh_token",
    "api_key",
    "password",
    "secret",
    "cookie",
}


def redact(value: Any, extra_keys: set[str] | None = None) -> Any:
    configured = (
        extra_keys
        if extra_keys is not None
        else set(filter(None, os.environ.get("TOOLFAULT_REDACT_KEYS", "").split(",")))
    )
    sensitive = DEFAULT_SENSITIVE_KEYS | {key.strip().casefold() for key in configured}
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).casefold() in sensitive else redact(item, extra_keys)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, extra_keys) for item in value]
    return value
