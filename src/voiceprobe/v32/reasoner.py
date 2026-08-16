"""Two-stage bounded contextual reasoning for VoiceProbe v3.2."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from voiceprobe.v3.flow_state import FlowSnapshot
from voiceprobe.v3.models import (
    PatientFacts,
    PolicyDecision,
)

from .context import (
    ReasoningContext,
    normalize_history,
)
from .schemas import (
    ACTION_JSON_SCHEMA,
    REWRITE_JSON_SCHEMA,
    ActionProposal,
    IntentRewrite,
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
    rewrite: IntentRewrite
    proposal: ActionProposal
    decision: PolicyDecision
    rewrite_ms: float
    planning_ms: float

    @property
    def total_ms(self) -> float:
        return (
            self.rewrite_ms
            + self.planning_ms
        )


_REWRITE_SYSTEM = """You are the semantic interpretation stage of a
task-oriented phone agent.

Do NOT answer the clinic.

Determine what the clinic's latest utterance means in the context of the
conversation and current state.

Important distinctions:
- medical reason for visit != reason for rescheduling
- acknowledgement != question
- status update != instruction
- old appointment != newly completed transaction
- never invent patient facts
- interpret semantics rather than matching exact wording

Return only the requested structured object.
"""


_PLAN_SYSTEM = """You are the bounded patient-action planning stage of
VoiceProbe.

Choose exactly one action from the supplied schema.

Rules:
1. Never invent or modify authoritative patient facts.
2. Never claim that a booking, cancellation, or reschedule completed.
3. Never grant transaction permission from this fallback layer.
4. Transaction state changes belong to deterministic VoiceProbe policy.
5. You may answer low-risk conversational questions naturally.
6. If asked why the patient needs to move an existing appointment, a safe
   response is that the current appointment time no longer works and the
   patient wants the already-established preferred day/time.
7. If an authoritative fact is requested, use answer_fact and fact_key.
   Python will render the actual fact.
8. If the clinic only acknowledges or gives a status update that needs no
   reply, choose wait.
9. proposed_state_change must always be false here.

Return only the requested structured object.
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
        self.validator = (
            validator or ActionValidator()
        )

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

        context_json = json.dumps(
            context.as_payload(),
            ensure_ascii=False,
            indent=2,
        )

        rewrite_prompt = f"""CONTEXT:
{context_json}

LATEST CLINIC UTTERANCE:
{remote_turn}

Rewrite the current semantic meaning.
"""

        started = time.perf_counter()

        rewrite_payload = (
            await self.backend.generate_json(
                system=_REWRITE_SYSTEM,
                prompt=rewrite_prompt,
                schema=REWRITE_JSON_SCHEMA,
            )
        )

        rewrite_ms = (
            time.perf_counter() - started
        ) * 1000.0

        rewrite = IntentRewrite.from_dict(
            rewrite_payload
        )

        planning_prompt = f"""AUTHORITATIVE CONTEXT:
{context_json}

LATEST CLINIC UTTERANCE:
{remote_turn}

SEMANTIC INTERPRETATION:
{json.dumps({
    "meaning": rewrite.meaning,
    "turn_kind": rewrite.turn_kind.value,
    "subject": rewrite.subject,
    "risk": rewrite.risk.value,
    "requires_response": rewrite.requires_response,
    "confidence": rewrite.confidence,
}, ensure_ascii=False, indent=2)}

Choose the safest patient action.
"""

        started = time.perf_counter()

        proposal_payload = (
            await self.backend.generate_json(
                system=_PLAN_SYSTEM,
                prompt=planning_prompt,
                schema=ACTION_JSON_SCHEMA,
            )
        )

        planning_ms = (
            time.perf_counter() - started
        ) * 1000.0

        proposal = ActionProposal.from_dict(
            proposal_payload
        )

        decision = self.validator.validate(
            rewrite=rewrite,
            proposal=proposal,
            facts=self.facts,
            snapshot=snapshot,
        )

        return ReasoningTrace(
            rewrite=rewrite,
            proposal=proposal,
            decision=decision,
            rewrite_ms=rewrite_ms,
            planning_ms=planning_ms,
        )
