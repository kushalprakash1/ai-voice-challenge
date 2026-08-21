import asyncio

import pytest
from pydantic import ValidationError

from voiceprobe.v3.flow_state import FlowSnapshot, FlowStage
from voiceprobe.v3.models import DecisionKind, PatientFacts
from voiceprobe.v32.reasoner import ContextualReasoner
from voiceprobe.v32.schemas import ContextualProposal


class FakeBackend:
    def __init__(self, response):
        self.response = response

    async def generate_json(self, **kwargs):
        return self.response


def snapshot():
    return FlowSnapshot(
        communicated=frozenset(),
        confirmed=frozenset(),
        current_stage=FlowStage.DATE_TIME,
        complete=False,
        accepted_slot_text=None,
        booking_confirmation_text=None,
        allow_earlier_week_afternoons=False,
    )


def test_confidence_100_is_rejected():
    with pytest.raises(ValidationError):
        ContextualProposal.model_validate({
            "meaning": "insurance question",
            "risk": "authoritative_fact",
            "action": "answer_fact",
            "grounding": "authoritative_fact",
            "fact_key": "insurance",
            "response_text": "",
            "confidence": 100,
        })


def test_model_cannot_override_insurance():
    reasoner = ContextualReasoner(
        backend=FakeBackend({
            "meaning": "asks insurance",
            "risk": "authoritative_fact",
            "action": "answer_fact",
            "grounding": "authoritative_fact",
            "fact_key": "insurance",
            "response_text": "Blue Shield",
            "confidence": 0.98,
        }),
        facts=PatientFacts(insurance="Blue Cross"),
    )

    trace = asyncio.run(
        reasoner.reason(
            remote_turn="Who is your health plan through?",
            snapshot=snapshot(),
        )
    )

    assert trace.decision.text == "Blue Cross."


def test_reschedule_reason_can_generalize():
    reasoner = ContextualReasoner(
        backend=FakeBackend({
            "meaning": "reason for rescheduling",
            "risk": "low",
            "action": "answer",
            "grounding": "current_goal",
            "fact_key": "none",
            "response_text": (
                "That time no longer works for me; "
                "Friday afternoon works better."
            ),
            "confidence": 0.97,
        }),
    )

    trace = asyncio.run(
        reasoner.reason(
            remote_turn=(
                "May I document what's prompting "
                "the appointment change?"
            ),
            snapshot=snapshot(),
        )
    )

    assert trace.decision.kind is DecisionKind.ANSWER_FACT
    assert "Friday afternoon" in trace.decision.text



def test_reasoner_fails_closed_on_invalid_model_output():
    """Malformed model output must not crash the runtime."""

    reasoner = ContextualReasoner(
        backend=FakeBackend({
            "meaning": "reason for rescheduling",
            "risk": "low",
            "action": "answer",
            "grounding": "current_goal",
            "fact_key": "none",
            "response_text": (
                "That time no longer works for me. "
                "Friday afternoon works better."
            ),

            # Deliberately invalid percentage-style confidence.
            "confidence": 100,
        }),
    )

    trace = asyncio.run(
        reasoner.reason(
            remote_turn=(
                "Why do you need to move your appointment?"
            ),
            snapshot=snapshot(),
        )
    )

    assert trace.decision.kind is DecisionKind.CLARIFY
    assert (
        trace.decision.reason
        == "v32_invalid_model_output"
    )
    assert trace.decision.confidence == 0.0
    assert trace.validation_error is not None
    assert "less than or equal to 1" in trace.validation_error
