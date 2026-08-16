from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from voiceprobe.reasoning.semantic_reasoner import (
    StructuredTurnReasoner,
)
from voiceprobe.reasoning.turn_frame import (
    RequestedAction,
    RequestedFact,
    SpeechAct,
    TurnFrame,
)


def make_response(
    frame: dict[str, object],
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {
                "content": json.dumps(frame),
            }
        },
    )


def test_turn_frame_supports_multiple_appointment_options() -> None:
    frame = TurnFrame.model_validate(
        {
            "speech_act": "offer",
            "workflow": "scheduling",
            "requested_action": "choose_option",
            "response_required": True,
            "requested_facts": [],
            "other_requested_facts": [],
            "appointment_options": [
                {
                    "day": "Friday",
                    "date_text": "August 21",
                    "time": "9 AM",
                    "daypart": "morning",
                    "provider": "Becker",
                    "appointment_type": None,
                },
                {
                    "day": "Friday",
                    "date_text": "August 21",
                    "time": "9:45 AM",
                    "daypart": "morning",
                    "provider": "Becker",
                    "appointment_type": None,
                },
                {
                    "day": "Friday",
                    "date_text": "August 21",
                    "time": "10:30 AM",
                    "daypart": "morning",
                    "provider": "Becker",
                    "appointment_type": None,
                },
            ],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 0.99,
        }
    )

    assert frame.speech_act is SpeechAct.OFFER

    assert (
        frame.requested_action
        is RequestedAction.CHOOSE_OPTION
    )

    assert len(frame.appointment_options) == 3


def test_requested_facts_are_typed_not_arbitrary_sentences() -> None:
    with pytest.raises(ValidationError):
        TurnFrame.model_validate(
            {
                "speech_act": "question",
                "workflow": "scheduling",
                "requested_action": "answer_fact",
                "response_required": True,
                "requested_facts": [
                    (
                        "Would you like me to check "
                        "Friday afternoon appointments?"
                    )
                ],
                "other_requested_facts": [],
                "appointment_options": [],
                "booking_confirmed": False,
                "conversation_end_requested": False,
                "agent_is_still_working": False,
                "confidence": 1.0,
            }
        )


def test_fact_request_uses_canonical_enum() -> None:
    frame = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "insurance",
            "requested_action": "answer_fact",
            "response_required": True,
            "requested_facts": [
                "insurance"
            ],
            "other_requested_facts": [],
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    assert frame.requested_facts == [
        RequestedFact.INSURANCE
    ]


def test_reasoner_payload_contains_no_patient_scenario() -> None:
    captured: dict[str, object] = {}

    result_frame = {
        "speech_act": "question",
        "workflow": "scheduling",
        "requested_action": "grant_permission",
        "response_required": True,
        "requested_facts": [],
        "other_requested_facts": [],
        "appointment_options": [],
        "booking_confirmed": False,
        "conversation_end_requested": False,
        "agent_is_still_working": False,
        "confidence": 1.0,
    }

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        captured["payload"] = json.loads(
            request.content
        )

        return make_response(
            result_frame
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            handler
        )
    )

    reasoner = StructuredTurnReasoner(
        model="qwen3:14b",
        url="http://ollama.test/api/chat",
        client=client,
    )

    try:
        result = reasoner.interpret(
            agent_turn=(
                "Would you like me to check "
                "Friday afternoon appointments?"
            ),
        )
    finally:
        reasoner.close()
        client.close()

    assert (
        result.requested_action
        is RequestedAction.GRANT_PERMISSION
    )

    assert result.requested_facts == []

    payload = captured["payload"]

    assert isinstance(
        payload,
        dict,
    )

    messages = payload["messages"]

    assert isinstance(
        messages,
        list,
    )

    user_content = messages[-1]["content"]

    decoded = json.loads(
        user_content
    )

    assert set(decoded) == {
        "recent_agent_history",
        "latest_agent_turn",
    }

    serialized = json.dumps(
        decoded
    ).casefold()

    assert "alex morgan" not in serialized
    assert "preferred_time" not in serialized
    assert "patientscenario" not in serialized


def test_wait_frame_cannot_request_fact() -> None:
    with pytest.raises(ValidationError):
        TurnFrame.model_validate(
            {
                "speech_act": "status",
                "workflow": "scheduling",
                "requested_action": "wait",
                "response_required": False,
                "requested_facts": [
                    "insurance"
                ],
                "other_requested_facts": [],
                "appointment_options": [],
                "booking_confirmed": False,
                "conversation_end_requested": False,
                "agent_is_still_working": True,
                "confidence": 1.0,
            }
        )


def test_reasoner_repairs_invalid_choose_option_frame() -> None:
    """Schema feedback should automatically repair impossible semantics."""

    calls = 0

    invalid = {
        "speech_act": "question",
        "workflow": "patient_intake",
        "requested_action": "choose_option",
        "response_required": True,
        "requested_facts": [],
        "other_requested_facts": [],
        "appointment_options": [],
        "booking_confirmed": False,
        "conversation_end_requested": False,
        "agent_is_still_working": False,
        "confidence": 0.9,
    }

    repaired = {
        "speech_act": "question",
        "workflow": "patient_intake",
        "requested_action": "answer_fact",
        "response_required": True,
        "requested_facts": [
            "provider_preference"
        ],
        "other_requested_facts": [],
        "appointment_options": [],
        "booking_confirmed": False,
        "conversation_end_requested": False,
        "agent_is_still_working": False,
        "confidence": 0.95,
    }

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        frame = (
            invalid
            if calls == 1
            else repaired
        )

        return make_response(
            frame
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            handler
        )
    )

    reasoner = StructuredTurnReasoner(
        model="qwen3:14b",
        url="http://ollama.test/api/chat",
        client=client,
    )

    try:
        result = reasoner.interpret(
            agent_turn=(
                "Do you have a specific provider "
                "you'd like to see?"
            ),
        )
    finally:
        reasoner.close()
        client.close()

    assert calls == 2

    assert (
        result.requested_action.value
        == "answer_fact"
    )

    assert [
        item.value
        for item in result.requested_facts
    ] == [
        "provider_preference"
    ]


def test_incomplete_fragment_wait_frame_is_valid() -> None:
    """Incomplete telephony fragments must be representable without guessing."""

    frame = TurnFrame.model_validate(
        {
            "speech_act": "other",
            "workflow": "unknown",
            "requested_action": "wait",
            "response_required": False,
            "requested_facts": [],
            "other_requested_facts": [],
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 0.6,
        }
    )

    assert (
        frame.requested_action.value
        == "wait"
    )

    assert frame.requested_facts == []


def test_turn_frame_can_record_untrusted_remote_fact_assertion() -> None:
    """Remote claims must be observable without becoming patient truth."""

    frame = TurnFrame.model_validate(
        {
            "speech_act": "information",
            "workflow": "patient_intake",
            "requested_action": "none",
            "response_required": False,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [
                {
                    "fact": "date_of_birth",
                    "value": "July 4th, 2000",
                }
            ],
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    assert len(
        frame.stated_facts
    ) == 1

    assertion = frame.stated_facts[0]

    assert (
        assertion.fact.value
        == "date_of_birth"
    )

    assert (
        assertion.value
        == "July 4th, 2000"
    )


def test_assertion_and_objective_request_can_coexist() -> None:
    """One utterance may contain several semantic events."""

    frame = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "patient_intake",
            "requested_action": "state_objective",
            "response_required": True,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [
                {
                    "fact": "date_of_birth",
                    "value": "July 4th, 2000",
                }
            ],
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    assert (
        frame.requested_action.value
        == "state_objective"
    )

    assert (
        frame.stated_facts[0].fact.value
        == "date_of_birth"
    )


def test_turn_frame_can_represent_optional_workflow_proposal() -> None:
    frame = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "profile_setup",
            "requested_action": "grant_permission",
            "response_required": True,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [],
            "proposed_workflow": {
                "kind": "profile_setup",
                "description": "create a demo patient profile",
                "requirement": "optional",
            },
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    assert (
        frame.requested_action.value
        == "grant_permission"
    )

    assert frame.proposed_workflow is not None

    assert (
        frame.proposed_workflow.kind.value
        == "profile_setup"
    )

    assert (
        frame.proposed_workflow.requirement.value
        == "optional"
    )


def test_normal_availability_permission_needs_no_side_workflow() -> None:
    frame = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "scheduling",
            "requested_action": "grant_permission",
            "response_required": True,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [],
            "proposed_workflow": None,
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    assert frame.proposed_workflow is None


def test_confirmed_appointment_is_not_patient_fact_assertion() -> None:
    """Booking state belongs to scheduling state, not caller-profile truth."""

    frame = TurnFrame.model_validate(
        {
            "speech_act": "information",
            "workflow": "scheduling",
            "requested_action": "none",
            "response_required": False,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [],
            "proposed_workflow": None,
            "appointment_options": [],
            "confirmed_appointment": {
                "day": "Friday",
                "date_text": None,
                "time": "2:30 PM",
                "daypart": None,
                "provider": None,
                "appointment_type": None,
            },
            "booking_confirmed": True,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    assert frame.stated_facts == []
    assert frame.appointment_options == []
    assert frame.booking_confirmed is True
    assert frame.confirmed_appointment is not None
    assert frame.confirmed_appointment.day == "Friday"
    assert frame.confirmed_appointment.time == "2:30 PM"


def test_confirmed_slot_requires_booking_confirmed() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TurnFrame.model_validate(
            {
                "speech_act": "information",
                "workflow": "scheduling",
                "requested_action": "none",
                "response_required": False,
                "requested_facts": [],
                "other_requested_facts": [],
                "stated_facts": [],
                "proposed_workflow": None,
                "appointment_options": [],
                "confirmed_appointment": {
                    "day": "Friday",
                    "date_text": None,
                    "time": "2:30 PM",
                    "daypart": None,
                    "provider": None,
                    "appointment_type": None,
                },
                "booking_confirmed": False,
                "conversation_end_requested": False,
                "agent_is_still_working": False,
                "confidence": 1.0,
            }
        )


def test_source_grounding_drops_assertion_missing_from_current_turn() -> None:
    """Historical or prompt contamination must not become a correction."""

    from voiceprobe.reasoning.semantic_reasoner import (
        source_ground_turn_frame,
    )

    frame = TurnFrame.model_validate(
        {
            "speech_act": "information",
            "workflow": "scheduling",
            "requested_action": "none",
            "response_required": False,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [
                {
                    "fact": "date_of_birth",
                    "value": "July 4th, 2000",
                }
            ],
            "proposed_workflow": None,
            "appointment_options": [],
            "confirmed_appointment": {
                "day": "Friday",
                "date_text": None,
                "time": "2:30 PM",
                "daypart": None,
                "provider": None,
                "appointment_type": None,
            },
            "booking_confirmed": True,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    grounded = source_ground_turn_frame(
        frame=frame,
        agent_turn=(
            "Great. You're booked for Friday at 2:30 PM."
        ),
    )

    assert grounded.stated_facts == []

    # The guard removes only unsupported assertions.
    # Legitimate scheduling semantics must survive unchanged.
    assert grounded.booking_confirmed is True
    assert grounded.confirmed_appointment is not None
    assert grounded.confirmed_appointment.day == "Friday"
    assert grounded.confirmed_appointment.time == "2:30 PM"


def test_source_grounding_keeps_supported_name_assertion() -> None:
    """A genuine current-turn assertion must survive provenance checking."""

    from voiceprobe.reasoning.semantic_reasoner import (
        source_ground_turn_frame,
    )

    frame = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "patient_intake",
            "requested_action": "answer_fact",
            "response_required": True,
            "requested_facts": [
                "last_name",
            ],
            "other_requested_facts": [],
            "stated_facts": [
                {
                    "fact": "first_name",
                    "value": "Martin",
                }
            ],
            "proposed_workflow": None,
            "appointment_options": [],
            "confirmed_appointment": None,
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    grounded = source_ground_turn_frame(
        frame=frame,
        agent_turn=(
            "Thanks, Martin. What is your last name?"
        ),
    )

    assert len(
        grounded.stated_facts
    ) == 1

    assert (
        grounded.stated_facts[0].fact.value
        == "first_name"
    )

    assert (
        grounded.stated_facts[0].value
        == "Martin"
    )


def test_source_grounding_keeps_supported_dob_assertion() -> None:
    """Ordinal/punctuation normalization must not erase real assertions."""

    from voiceprobe.reasoning.semantic_reasoner import (
        source_ground_turn_frame,
    )

    frame = TurnFrame.model_validate(
        {
            "speech_act": "information",
            "workflow": "profile_setup",
            "requested_action": "none",
            "response_required": False,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [
                {
                    "fact": "date_of_birth",
                    "value": "July 4th, 2000",
                }
            ],
            "proposed_workflow": None,
            "appointment_options": [],
            "confirmed_appointment": None,
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    grounded = source_ground_turn_frame(
        frame=frame,
        agent_turn=(
            "Your date of birth is July 4th, 2000."
        ),
    )

    assert len(
        grounded.stated_facts
    ) == 1

    assert (
        grounded.stated_facts[0].fact.value
        == "date_of_birth"
    )


def test_booking_confirmed_requires_confirmed_appointment() -> None:
    """A booking claim without a structured booked slot is incomplete."""

    import pytest
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError
    ):
        TurnFrame.model_validate(
            {
                "speech_act": "information",
                "workflow": "scheduling",
                "requested_action": "none",
                "response_required": False,
                "requested_facts": [],
                "other_requested_facts": [],
                "stated_facts": [],
                "proposed_workflow": None,
                "appointment_options": [],
                "confirmed_appointment": None,
                "booking_confirmed": True,
                "conversation_end_requested": False,
                "agent_is_still_working": False,
                "confidence": 1.0,
            }
        )


def test_semantic_booking_payload_is_repaired_from_current_source() -> None:
    """Explicit booking speech overrides an incomplete model slot."""

    import json

    import httpx

    from voiceprobe.reasoning.semantic_reasoner import (
        StructuredTurnReasoner,
    )

    calls = 0

    model_payload = {
        "speech_act": "confirmation",
        "workflow": "scheduling",
        "requested_action": "none",
        "response_required": False,
        "requested_facts": [],
        "other_requested_facts": [],
        "stated_facts": [],
        "proposed_workflow": None,
        "appointment_options": [],
        "confirmed_appointment": None,
        "booking_confirmed": True,
        "conversation_end_requested": False,
        "agent_is_still_working": False,
        "confidence": 1.0,
    }

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        model_payload
                    )
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            handler
        )
    )

    reasoner = StructuredTurnReasoner(
        model="qwen3:14b",
        url="http://ollama.test/api/chat",
        client=client,
    )

    try:
        frame = reasoner.interpret(
            agent_turn=(
                "Okay, you're booked for Friday at 4.30 p.m."
            ),
            recent_history=(
                "Friday.",
            ),
        )
    finally:
        reasoner.close()
        client.close()

    # Deterministic provenance repair should avoid another LLM call.
    assert calls == 1

    assert frame.booking_confirmed is True
    assert frame.confirmed_appointment is not None
    assert frame.confirmed_appointment.day == "Friday"
    assert frame.confirmed_appointment.time == "4:30 PM"


def test_semantic_booking_source_overrides_hallucinated_slot() -> None:
    """Model-generated slot values cannot override explicit remote speech."""

    from voiceprobe.reasoning.semantic_reasoner import (
        source_repair_semantic_payload,
    )

    payload = {
        "speech_act": "confirmation",
        "workflow": "scheduling",
        "requested_action": "none",
        "response_required": False,
        "requested_facts": [],
        "other_requested_facts": [],
        "stated_facts": [],
        "proposed_workflow": None,
        "appointment_options": [],
        "confirmed_appointment": {
            "day": "Friday",
            "date_text": None,
            "time": "9:45 AM",
            "daypart": None,
            "provider": "Becker",
            "appointment_type": None,
        },
        "booking_confirmed": True,
        "conversation_end_requested": False,
        "agent_is_still_working": False,
        "confidence": 1.0,
    }

    repaired = source_repair_semantic_payload(
        payload=payload,
        agent_turn=(
            "Okay, you're booked for Friday at 4.30 p.m."
        ),
        recent_history=(
            "Friday.",
        ),
    )

    slot = repaired[
        "confirmed_appointment"
    ]

    assert slot[
        "day"
    ] == "Friday"

    assert slot[
        "time"
    ] == "4:30 PM"

    # Becker was not stated anywhere in the supplied speech.
    assert slot[
        "provider"
    ] is None



def test_booking_confirmation_does_not_leak_into_later_goodbye() -> None:
    """Recent booking history cannot make a goodbye another confirmation."""

    from voiceprobe.reasoning.semantic_reasoner import (
        source_repair_semantic_payload,
    )

    payload = {
        "speech_act": "goodbye",
        "workflow": "scheduling",
        "requested_action": "none",
        "response_required": False,
        "requested_facts": [],
        "other_requested_facts": [],
        "stated_facts": [],
        "proposed_workflow": None,
        "appointment_options": [],
        "confirmed_appointment": {
            "day": "Friday",
            "date_text": None,
            "time": None,
            "daypart": None,
            "provider": None,
            "appointment_type": None,
        },
        "booking_confirmed": True,
        "conversation_end_requested": True,
        "agent_is_still_working": False,
        "confidence": 1.0,
    }

    repaired = source_repair_semantic_payload(
        payload=payload,
        agent_turn="Okay, bye.",
        recent_history=(
            "Okay, you're booked for Friday at 4.30 p.m.",
        ),
    )

    assert repaired[
        "booking_confirmed"
    ] is False

    assert repaired[
        "confirmed_appointment"
    ] is None


def test_elliptical_current_confirmation_can_inherit_recent_slot() -> None:
    """A real current confirmation may use immediately relevant slot context."""

    from voiceprobe.reasoning.semantic_reasoner import (
        source_repair_semantic_payload,
    )

    payload = {
        "speech_act": "confirmation",
        "workflow": "scheduling",
        "requested_action": "none",
        "response_required": False,
        "requested_facts": [],
        "other_requested_facts": [],
        "stated_facts": [],
        "proposed_workflow": None,
        "appointment_options": [],
        "confirmed_appointment": None,
        "booking_confirmed": True,
        "conversation_end_requested": False,
        "agent_is_still_working": False,
        "confidence": 1.0,
    }

    repaired = source_repair_semantic_payload(
        payload=payload,
        agent_turn="You're all set.",
        recent_history=(
            "Friday at 4.30 p.m.",
        ),
    )

    assert repaired[
        "booking_confirmed"
    ] is True

    slot = repaired[
        "confirmed_appointment"
    ]

    assert slot is not None
    assert slot[
        "day"
    ] == "Friday"
    assert slot[
        "time"
    ] == "4:30 PM"
