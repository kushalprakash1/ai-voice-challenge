"""Constrained Ollama fallback selector for VoiceProbe.

The local model is used only to classify ambiguous tested-agent turns.
It never generates final patient speech and never controls scenario facts.
"""

from __future__ import annotations

import json

import httpx

from voiceprobe.conversation.state import PatientState
from voiceprobe.planners.hybrid import (
    ResponsePlan,
    SelectorDecision,
)
from voiceprobe.scenarios.models import PatientScenario

DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_URL = "http://127.0.0.1:11434/api/chat"


class OllamaActionSelector:
    """Classify an ambiguous turn into one constrained response plan."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_URL,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._url = url
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
        )
        self._owns_client = client is None

    def close(self) -> None:
        """Close the internally owned HTTP client."""
        if self._owns_client:
            self._client.close()

    def select(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> SelectorDecision:
        """Ask Ollama for only a constrained response-plan decision."""
        available_facts = {
            key: value
            for key, value in scenario.facts.model_dump().items()
            if value is not None
        }

        recent_history = [
            {
                "speaker": message.speaker.value,
                "text": message.text,
            }
            for message in state.messages[-6:]
        ]

        context = {
            "objective": scenario.objective,
            "available_facts": available_facts,
            "recent_history": recent_history,
            "tested_agent_turn": agent_turn,
        }

        response = self._client.post(
            self._url,
            json={
                "model": self._model,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0,
                    "num_predict": 20,
                },
                "format": SelectorDecision.model_json_schema(),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Classify the tested voice agent's latest turn. "
                            "Choose exactly one response plan. "
                            "Do not write patient dialogue. "
                            "Use correct_complaint_duration when the tested "
                            "agent states patient complaint or duration facts "
                            "that conflict with the supplied facts. "
                            "Use clarify when the turn cannot be understood. "
                            "Use probe for suspicious or unusual behavior. "
                            "Use complete only when the scheduling objective "
                            "has clearly been accomplished."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            context,
                            separators=(",", ":"),
                        ),
                    },
                ],
            },
        )

        response.raise_for_status()

        payload = response.json()

        try:
            content = payload["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "Ollama response did not contain assistant content."
            ) from error

        if not isinstance(content, str):
            raise TypeError("Ollama assistant content was not text.")

        return SelectorDecision.model_validate_json(content)


__all__ = [
    "OllamaActionSelector",
    "ResponsePlan",
    "SelectorDecision",
]
