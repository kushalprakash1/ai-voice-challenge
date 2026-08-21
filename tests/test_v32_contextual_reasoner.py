"""Regression tests for the VoiceProbe v3.2 single-pass reasoner."""

import asyncio

from voiceprobe.v3.flow_state import (
    FlowSnapshot,
    FlowStage,
)
from voiceprobe.v3.models import (
    DecisionKind,
    PatientFacts,
)
from voiceprobe.v32.reasoner import ContextualReasoner


class FakeBackend:
    """Return one already-structured model proposal."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    async def generate_json(
        self,
        *,
        system,
        prompt,
        schema,
    ):
        self.calls.append(
            {
                "system": system,
                "prompt": prompt,
                "schema": schema,
            }
        )
        return self.response


def snapshot():
    return FlowSnapshot(
        communicated=frozenset(
            {
                FlowStage.PROFILE,
                FlowStage.IDENTITY,
                FlowStage.DOB,
                FlowStage.VISIT_REASON,
                FlowStage.APPOINTMENT_TYPE,
                FlowStage.DATE_TIME,
            }
        ),
        confirmed=frozenset(),
        current_stage=FlowStage.PROVIDER,
        complete=False,
        accepted_slot_text=None,
        booking_confirmation_text=None,
        allow_earlier_week_afternoons=False,
    )


def test_unseen_reschedule_reason():
    """Novel wording should not require a phrase-specific production rule."""

    backend = FakeBackend(
        {
            "meaning": "reason for rescheduling",
            "risk": "low",
            "action": "answer",
            "grounding": "current_goal",
            "fact_key": "none",
            "response_text": (
                "That appointment time no longer works for me. "
                "Friday afternoon works better."
            ),
            "confidence": 0.98,
        }
    )

    trace = asyncio.run(
        ContextualReasoner(
            backend=backend,
        ).reason(
            remote_turn=(
                "What is the reason you're "
                "changing your appointment?"
            ),
            snapshot=snapshot(),
            recent_dialogue=(
                "PGAI: You already have an appointment Tuesday at 2:15 PM.",
                "PATIENT: I'd like to move it to Friday afternoon.",
            ),
        )
    )

    assert trace.decision.kind is DecisionKind.ANSWER_FACT
    assert trace.decision.reason == "v32_contextual_answer"
    assert "Friday afternoon" in trace.decision.text
    assert "shoulder" not in trace.decision.text.casefold()
    assert len(backend.calls) == 1


def test_python_owns_insurance_fact():
    """Model interpretation may select a fact, but Python owns its value."""

    backend = FakeBackend(
        {
            "meaning": "asks insurance",
            "risk": "authoritative_fact",
            "action": "answer_fact",
            "grounding": "authoritative_fact",
            "fact_key": "insurance",

            # Deliberately malicious/wrong model-generated text.
            # It must never override authoritative PatientFacts.
            "response_text": "Blue Shield",
            "confidence": 0.99,
        }
    )

    trace = asyncio.run(
        ContextualReasoner(
            backend=backend,
            facts=PatientFacts(
                insurance="Blue Cross",
            ),
        ).reason(
            remote_turn="Who is your health plan through?",
            snapshot=snapshot(),
        )
    )

    assert trace.decision.kind is DecisionKind.ANSWER_FACT
    assert trace.decision.text == "Blue Cross."
    assert "Blue Shield" not in trace.decision.text
    assert (
        trace.decision.reason
        == "v32_authoritative_fact:insurance"
    )
    assert len(backend.calls) == 1


def test_model_cannot_change_transaction():
    """Contextual reasoning never receives booking authority."""

    backend = FakeBackend(
        {
            "meaning": "asks for booking authorization",
            "risk": "transaction",
            "action": "answer",
            "grounding": "low_risk_conversational",
            "fact_key": "none",
            "response_text": "Go ahead and book it.",
            "confidence": 0.99,
        }
    )

    trace = asyncio.run(
        ContextualReasoner(
            backend=backend,
        ).reason(
            remote_turn="Should I book that?",
            snapshot=snapshot(),
        )
    )

    assert trace.decision.kind is DecisionKind.CLARIFY
    assert (
        trace.decision.reason
        == "v32_transaction_requires_deterministic_policy"
    )
    assert "book it" not in trace.decision.text.casefold()
    assert len(backend.calls) == 1
