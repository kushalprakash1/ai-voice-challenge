
"""Single-call contextual semantic parser."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from .semantic_frame import (
    Certainty,
    Commitment,
    Focus,
    Operation,
    SEMANTIC_FRAME_SCHEMA,
    SemanticFrame,
    SpeechAct,
)


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
class SemanticParseTrace:
    frame: SemanticFrame
    latency_ms: float
    validation_error: str | None = None


_SYSTEM = """You are the semantic parser inside a patient phone agent.

Your ONLY task is to describe the meaning of the clinic's LATEST utterance.

Do not answer the clinic.
Do not decide what VoiceProbe should say.
Do not mutate appointment state.

CRITICAL CONTEXT RULE:
The latest utterance determines its explicit subject.
Conversation history may resolve pronouns such as "it", "that appointment",
or "that provider", but history must NEVER override an explicit subject in
the latest utterance.

For example, if the conversation is about rescheduling but the clinic now
asks about insurance, focus=insurance.

FIELDS

speech_act:
ask          clinic asks for information or a choice
inform       clinic states information
acknowledge  short acknowledgement such as okay / understood
request      clinic requests the patient perform/supply something
offer        clinic presents a choice or offer
fragment     incomplete utterance that should be allowed to continue
other        none of the above

operation:
book, reschedule, cancel, keep describe appointment operations.
list_slots describes presentation of appointment options.
choose_provider describes provider selection.
Use none when no operation is being discussed.

focus:
visit_reason means WHY the patient is seeking care or why the medical visit
exists. This includes questions asking the purpose/reason for the visit or
appointment when no change/reschedule operation is being discussed.

reschedule_reason means WHY an EXISTING appointment is being changed,
moved, or rescheduled.

These concepts are mutually distinct:

- reason the patient needs medical care -> visit_reason
- reason the patient is changing an existing appointment -> reschedule_reason

Do not infer reschedule_reason merely because the broader conversation has
previously involved scheduling or rescheduling. The latest utterance must
actually concern why the appointment is being changed.

insurance means insurance carrier/plan.
provider_preference means which provider/provider availability the patient wants.
dob/name/complaint/preferred_day/preferred_time are literal patient facts.
appointment_status means information about an existing appointment.
slot_options_intro means an introduction to slot choices before a concrete choice.
Use other only when no defined focus fits.

commitment:
informational means discussing or requesting information only.
permission_request means the clinic asks whether it may BOOK, CANCEL,
RESCHEDULE, or KEEP an appointment.
authorization means the patient explicitly authorizes such an operation.
confirmation means the clinic reports that a transaction actually completed.
none means commitment is irrelevant.

IMPORTANT CONTRAST:

A reason-for-care question and a reason-for-change question are different.

Clinic asks why the patient needs the visit:
focus=visit_reason
commitment=informational

Clinic asks why the patient is moving/rescheduling an existing visit:
focus=reschedule_reason
commitment=informational

A reason-for-change question is still NOT transaction permission.

Only asking whether the clinic may actually book, cancel, keep, or
reschedule the appointment is a permission_request.

CALIBRATION EXAMPLES

Clinic: "Why does the patient need this visit?"
Frame:
speech_act=ask
operation=none
focus=visit_reason
commitment=informational
certainty=high

Clinic: "What is this appointment for?"
Frame:
speech_act=ask
operation=none
focus=visit_reason
commitment=informational
certainty=high

Clinic: "What's making you reschedule?"
Frame:
speech_act=ask
operation=reschedule
focus=reschedule_reason
commitment=informational
certainty=high

Clinic: "Which insurer do you have?"
Frame:
speech_act=ask
operation=none
focus=insurance
commitment=informational
certainty=high

Clinic: "Do you want me to finalize this booking?"
Frame:
speech_act=ask
operation=book
focus=appointment_status
commitment=permission_request
certainty=high

Clinic: "Okay, understood."
Frame:
speech_act=acknowledge
operation=none
focus=none
commitment=none
certainty=high

Clinic: "I see an existing visit next Wednesday."
Frame:
speech_act=inform
operation=none
focus=appointment_status
commitment=informational
certainty=high

Clinic: "I can show you several afternoon slots."
Frame:
speech_act=inform
operation=list_slots
focus=slot_options_intro
commitment=informational
certainty=high

Clinic: "Do you have a clinician preference?"
Frame:
speech_act=ask
operation=choose_provider
focus=provider_preference
commitment=informational
certainty=high

Return only the required structured object.
"""


class SemanticParser:
    def __init__(
        self,
        *,
        backend: StructuredBackend,
    ) -> None:
        self.backend = backend

    async def parse(
        self,
        *,
        remote_turn: str,
        recent_dialogue: tuple[str, ...] = (),
    ) -> SemanticParseTrace:

        history = [
            " ".join(item.split())
            for item in recent_dialogue[-6:]
            if item.strip()
        ]

        payload = {
            "recent_dialogue": history,
            "latest_clinic_utterance": " ".join(
                remote_turn.split()
            ),
        }

        started = time.perf_counter()

        raw = await self.backend.generate_json(
            system=_SYSTEM,
            prompt=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            schema=SEMANTIC_FRAME_SCHEMA,
        )

        latency = (
            time.perf_counter() - started
        ) * 1000.0

        try:
            frame = SemanticFrame.model_validate(raw)
        except ValidationError as exc:
            return SemanticParseTrace(
                frame=SemanticFrame(
                    speech_act=SpeechAct.OTHER,
                    operation=Operation.NONE,
                    focus=Focus.OTHER,
                    commitment=Commitment.NONE,
                    certainty=Certainty.LOW,
                ),
                latency_ms=latency,
                validation_error=str(exc),
            )

        return SemanticParseTrace(
            frame=frame,
            latency_ms=latency,
        )
