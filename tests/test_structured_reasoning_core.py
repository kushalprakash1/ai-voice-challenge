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
