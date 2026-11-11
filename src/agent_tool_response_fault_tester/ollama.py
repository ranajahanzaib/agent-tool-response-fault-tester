"""Minimal Ollama native chat adapter with tool calling."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaAgent:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str | None = None,
        timeout: float = 90,
        temperature: float = 0,
        seed: int | None = 42,
        max_response_size_bytes: int = 1_000_000,
    ):
        self.model = model
        resolved_url = (
            base_url or os.getenv("TOOLFAULT_OLLAMA_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        parsed_url = urlsplit(resolved_url)
        if (
            parsed_url.scheme != "http"
            or parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed_url.username
            or parsed_url.password
        ):
            raise OllamaError("Ollama must use a local HTTP endpoint on localhost or loopback")
        self.base_url = resolved_url
        self.timeout = timeout
        self.max_response_size_bytes = max_response_size_bytes
        self.options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            self.options["seed"] = seed

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                    "options": self.options,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed at {self.base_url}: {exc}") from exc
        value = response.json()
        message = value.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama returned no chat message")
        if len(str(message).encode("utf-8")) > self.max_response_size_bytes:
            raise OllamaError(f"Ollama response exceeded {self.max_response_size_bytes} bytes")
        return {
            "message": message,
            "model": value.get("model", self.model),
            "total_duration": value.get("total_duration"),
        }

    def models(self) -> list[str]:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return [row["name"] for row in response.json().get("models", [])]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise OllamaError(f"Cannot list local Ollama models: {exc}") from exc

    def version(self) -> str | None:
        try:
            response = httpx.get(f"{self.base_url}/api/version", timeout=5)
            response.raise_for_status()
            value = response.json()
            return str(value.get("version")) if value.get("version") else None
        except (httpx.HTTPError, ValueError, TypeError):
            return None

    def ensure_model(self) -> str | None:
        names = self.models()
        if self.model not in names:
            raise OllamaError(
                f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
            )
        return self.version()
