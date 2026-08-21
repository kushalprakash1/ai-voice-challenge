"""Adapter to the existing v3 fallback-resolver interface."""

from __future__ import annotations

from collections import deque

from voiceprobe.v3.flow_state import FlowSnapshot
from voiceprobe.v3.models import (
    PatientFacts,
    PolicyDecision,
)

from .ollama_backend import OllamaBackend
from .reasoner import (
    ContextualReasoner,
    ReasoningTrace,
)


class ContextualFallbackResolver:
    def __init__(
        self,
        *,
        reasoner: ContextualReasoner | None = None,
        facts: PatientFacts | None = None,
        history_limit: int = 8,
    ) -> None:
        patient = facts or PatientFacts()

        self.reasoner = (
            reasoner
            or ContextualReasoner(
                backend=OllamaBackend(),
                facts=patient,
            )
        )

        self._history: deque[str] = deque(
            maxlen=history_limit
        )

        self.last_trace: (
            ReasoningTrace | None
        ) = None

    def observe(
        self,
        speaker: str,
        text: str,
    ) -> None:
        normalized = " ".join(text.split())

        if normalized:
            self._history.append(
                f"{speaker.upper()}: {normalized}"
            )

    async def resolve_with_trace(
        self,
        remote_turn: str,
        snapshot: FlowSnapshot,
    ) -> ReasoningTrace:
        trace = await self.reasoner.reason(
            remote_turn=remote_turn,
            snapshot=snapshot,
            recent_dialogue=tuple(
                self._history
            ),
        )

        self.last_trace = trace
        return trace

    async def __call__(
        self,
        remote_turn: str,
        snapshot: FlowSnapshot,
    ) -> PolicyDecision:
        trace = await self.resolve_with_trace(
            remote_turn,
            snapshot,
        )

        return trace.decision
