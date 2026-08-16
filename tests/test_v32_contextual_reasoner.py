import asyncio

from voiceprobe.v3.flow_state import (
    FlowSnapshot,
    FlowStage,
)
from voiceprobe.v3.models import (
    DecisionKind,
    PatientFacts,
)
from voiceprobe.v32.reasoner import (
    ContextualReasoner,
)


class FakeBackend:
    def __init__(self, *responses):
        self.responses = list(responses)

    async def generate_json(
        self,
        *,
        system,
        prompt,
        schema,
    ):
        del system, prompt, schema
        return self.responses.pop(0)


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
    backend = FakeBackend(
        {
            "meaning": (
                "The clinic asks why the patient "
                "wants to reschedule."
            ),
            "turn_kind": "question",
            "subject": "reschedule reason",
            "risk": "low",
            "requires_response": True,
            "confidence": 0.98,
        },
        {
            "action": "answer",
            "grounding": (
                "low_risk_conversational"
            ),
            "fact_key": "",
            "answer_text": (
                "That appointment time no longer "
                "works for me. I'd like to move "
                "it to Friday afternoon."
            ),
            "proposed_state_change": False,
            "confidence": 0.97,
        },
    )

    trace = asyncio.run(
        ContextualReasoner(
            backend=backend
        ).reason(
            remote_turn=(
                "What is the reason you're "
                "changing your appointment?"
            ),
            snapshot=snapshot(),
        )
    )

    assert (
        trace.decision.kind
        is DecisionKind.ANSWER_FACT
    )
    assert (
        trace.decision.reason
        == "v32_contextual_answer"
    )
    assert (
        "Friday afternoon"
        in trace.decision.text
    )


def test_python_owns_insurance_fact():
    backend = FakeBackend(
        {
            "meaning": (
                "The clinic asks for insurance."
            ),
            "turn_kind": "question",
            "subject": "insurance",
            "risk": "authoritative_fact",
            "requires_response": True,
            "confidence": 0.99,
        },
        {
            "action": "answer_fact",
            "grounding": (
                "authoritative_fact"
            ),
            "fact_key": "insurance",
            "answer_text": "Blue Shield",
            "proposed_state_change": False,
            "confidence": 0.99,
        },
    )

    trace = asyncio.run(
        ContextualReasoner(
            backend=backend,
            facts=PatientFacts(
                insurance="Blue Cross"
            ),
        ).reason(
            remote_turn=(
                "Who is your health plan through?"
            ),
            snapshot=snapshot(),
        )
    )

    assert trace.decision.text == "Blue Cross."
    assert (
        "Blue Shield"
        not in trace.decision.text
    )


def test_model_cannot_change_transaction():
    backend = FakeBackend(
        {
            "meaning": (
                "The clinic asks for booking "
                "authorization."
            ),
            "turn_kind": "transaction",
            "subject": "booking",
            "risk": "transaction",
            "requires_response": True,
            "confidence": 0.99,
        },
        {
            "action": "answer",
            "grounding": (
                "low_risk_conversational"
            ),
            "fact_key": "",
            "answer_text": (
                "Go ahead and book it."
            ),
            "proposed_state_change": True,
            "confidence": 0.99,
        },
    )

    trace = asyncio.run(
        ContextualReasoner(
            backend=backend
        ).reason(
            remote_turn="Should I book that?",
            snapshot=snapshot(),
        )
    )

    assert (
        trace.decision.kind
        is DecisionKind.CLARIFY
    )
    assert (
        trace.decision.reason
        == "v32_blocked_model_state_change"
    )
