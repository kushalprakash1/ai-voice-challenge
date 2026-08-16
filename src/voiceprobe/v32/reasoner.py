"""Single-pass bounded contextual reasoner for VoiceProbe v3.2."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from voiceprobe.v3.flow_state import FlowSnapshot
from voiceprobe.v3.models import (
    DecisionKind,
    PatientFacts,
    PolicyDecision,
)

from .context import ReasoningContext, normalize_history
from .schemas import (
    ContextualProposal,
    PROPOSAL_JSON_SCHEMA,
)
from .validator import ActionValidator


class StructuredBackend(Protocol):
    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class ReasoningTrace:
    proposal: ContextualProposal
    decision: PolicyDecision
    reasoning_ms: float
    validation_error: str | None = None

    @property
    def total_ms(self) -> float:
        return self.reasoning_ms


_SYSTEM = """You are VoiceProbe's bounded contextual reasoning layer.

Understand the clinic's latest utterance using conversation context and
authoritative state.

You do not control appointment state.

Rules:
- Never invent authoritative patient facts.
- Medical reason for visit and reason for rescheduling are different.
- Historical appointments are not new transaction confirmations.
- Booking, cancellation, confirmation, and authorization are transaction risk.
- Use answer_fact plus fact_key when the clinic requests an authoritative fact.
- Python, not you, will supply authoritative fact values.
- Use wait for acknowledgements/status messages needing no reply.
- For harmless conversational questions, use answer and a short natural
  response_text of at most one sentence.
- If asked why the existing appointment needs to move, explain only that the
  current time no longer works and the patient wants the already-established
  preferred day/time.
- Keep meaning extremely short.
- Never put medical symptoms into a rescheduling explanation unless explicitly
  stated as the reason in authoritative context.
"""


class ContextualReasoner:
    def __init__(
        self,
        *,
        backend: StructuredBackend,
        facts: PatientFacts | None = None,
        validator: ActionValidator | None = None,
    ) -> None:
        self.backend = backend
        self.facts = facts or PatientFacts()
        self.validator = validator or ActionValidator()

    async def reason(
        self,
        *,
        remote_turn: str,
        snapshot: FlowSnapshot,
        recent_dialogue: tuple[str, ...] = (),
    ) -> ReasoningTrace:
        context = ReasoningContext(
            facts=self.facts,
            snapshot=snapshot,
            recent_dialogue=normalize_history(
                recent_dialogue
            ),
        )

        prompt = json.dumps(
            {
                "context": context.as_payload(),
                "latest_clinic_utterance": remote_turn,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        started = time.perf_counter()

        raw = await self.backend.generate_json(
            system=_SYSTEM,
            prompt=prompt,
            schema=PROPOSAL_JSON_SCHEMA,
        )

        elapsed = (
            time.perf_counter() - started
        ) * 1000.0

        # This is the trust boundary. JSON-schema generation alone
        # is not treated as sufficient validation.
        #
        # A malformed model response must never crash the voice runtime.
        # We deliberately DO NOT coerce invalid values such as
        # confidence=100 into apparently valid values such as 1.0.
        try:
            proposal = ContextualProposal.model_validate(raw)
        except ValidationError as exc:
            validation_error = str(exc)

            # Preserve the typed trace shape for observability while making
            # the proposal explicitly non-authoritative and unusable.
            proposal = ContextualProposal.model_validate(
                {
                    "meaning": "invalid model output",
                    "risk": "uncertain",
                    "action": "clarify",
                    "grounding": "none",
                    "fact_key": "none",
                    "response_text": "",
                    "confidence": 0.0,
                }
            )

            decision = PolicyDecision(
                DecisionKind.CLARIFY,
                text="Could you clarify that question?",
                reason="v32_invalid_model_output",
                confidence=0.0,
            )

            return ReasoningTrace(
                proposal=proposal,
                decision=decision,
                reasoning_ms=elapsed,
                validation_error=validation_error,
            )

        decision = self.validator.validate(
            proposal=proposal,
            facts=self.facts,
            snapshot=snapshot,
        )

        return ReasoningTrace(
            proposal=proposal,
            decision=decision,
            reasoning_ms=elapsed,
            validation_error=None,
        )
