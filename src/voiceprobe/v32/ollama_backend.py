"""Small local Ollama structured-output backend."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib import request


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    model: str = "qwen3.5:4b"
    endpoint: str = "http://127.0.0.1:11434/api/chat"
    timeout_seconds: float = 30.0
    keep_alive: str = "10m"
    num_ctx: int = 4096
    temperature: float = 0.1


class OllamaBackend:
    def __init__(
        self,
        config: OllamaConfig | None = None,
    ) -> None:
        self.config = config or OllamaConfig()

    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._generate_json_sync,
            system,
            prompt,
            schema,
        )

    def _generate_json_sync(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "stream": False,
            "think": False,
            "keep_alive": self.config.keep_alive,
            "format": schema,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
            },
        }

        req = request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        with request.urlopen(
            req,
            timeout=self.config.timeout_seconds,
        ) as response:
            envelope = json.loads(
                response.read().decode("utf-8")
            )

        content = envelope["message"]["content"]
        parsed = json.loads(content)

        if not isinstance(parsed, dict):
            raise ValueError(
                "Structured model output was not an object."
            )

        return parsed
