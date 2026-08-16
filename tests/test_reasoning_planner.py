from __future__ import annotations

import json

import httpx

from voiceprobe.reasoning.action_plan import (
    ActionPlan,
    PatientActionKind,
)
from voiceprobe.reasoning.constraint_validator import (
    ConstraintValidator,
)
from voiceprobe.reasoning.planner import (
    QwenPatientPlanner,
)
from voiceprobe.reasoning.turn_frame import (
    TurnFrame,
)
from voiceprobe.reasoning.world_model import (
    build_world_model,
)
from voiceprobe.scenarios.catalog import (
    get_scenario,
)


def build_multi_slot_turn() -> TurnFrame:
    return TurnFrame.model_validate(
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
            "confidence": 1.0,
        }
    )


def test_world_model_derives_generic_constraints() -> None:
    world = build_world_model(
        get_scenario(
            "autonomous-phone-diagnostic"
        )
    )

    constraints = {
        item.field: item.value
        for item in world.constraints
    }

    assert constraints["day"].casefold() == "friday"
    assert constraints["time"].casefold() == "afternoon"

    serialized = json.dumps(
        world.model_dump(
            mode="json",
        )
    )

    # Data may contain the patient's name, but the CONSTRAINT MODEL
    # itself must not be implemented by checking the name.
    assert '"field": "day"' in serialized
    assert '"field": "time"' in serialized


def test_validator_rejects_morning_slot_for_afternoon_constraint() -> None:
    world = build_world_model(
        get_scenario(
            "autonomous-phone-diagnostic"
        )
    )

    turn = build_multi_slot_turn()

    plan = ActionPlan(
        action=PatientActionKind.SELECT_OPTION,
        selected_option_index=0,
        reason_code="select_first_option",
        confidence=1.0,
    )

    violations = ConstraintValidator().validate(
        world=world,
        turn=turn,
        plan=plan,
    )

    assert any(
        item.code
        == "hard_constraint_conflict"
        for item in violations
    )


def test_validator_accepts_matching_afternoon_slot() -> None:
    world = build_world_model(
        get_scenario(
            "autonomous-phone-diagnostic"
        )
    )

    turn = TurnFrame.model_validate(
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
                    "time": "2:30 PM",
                    "daypart": "afternoon",
                    "provider": "Becker",
                    "appointment_type": None,
                }
            ],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    plan = ActionPlan(
        action=PatientActionKind.SELECT_OPTION,
        selected_option_index=0,
        reason_code="compatible_slot",
        confidence=1.0,
    )

    assert (
        ConstraintValidator().validate(
            world=world,
            turn=turn,
            plan=plan,
        )
        == ()
    )


def test_action_vocabulary_has_no_generic_agree() -> None:
    values = {
        item.value
        for item in PatientActionKind
    }

    assert "agree" not in values
    assert "yes" not in values

    assert "select_option" in values
    assert "request_alternative" in values


def test_planner_uses_structured_action_schema() -> None:
    """Multiple compatible options should invoke the Qwen planner."""

    captured: dict[str, object] = {}

    response_plan = {
        "action": "select_option",
        "selected_option_index": 0,
        "facts_to_answer": [],
        "reason_code": "choose_compatible_option",
        "confidence": 1.0,
    }

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        captured["payload"] = json.loads(
            request.content
        )

        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        response_plan
                    )
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            handler
        )
    )

    planner = QwenPatientPlanner(
        model="qwen3:14b",
        url="http://ollama.test/api/chat",
        client=client,
    )

    # Both options satisfy the scenario's HARD constraints:
    #
    # day  = Friday
    # time = afternoon
    #
    # Because more than one compatible option remains, deterministic
    # filtering cannot uniquely decide and Qwen should be invoked.
    ambiguous_turn = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "scheduling",
            "requested_action": "choose_option",
            "response_required": True,
            "requested_facts": [],
            "other_requested_facts": [],
            "appointment_options": [
                {
                    "day": "Friday",
                    "date_text": None,
                    "time": "2:30 PM",
                    "daypart": None,
                    "provider": None,
                    "appointment_type": None,
                },
                {
                    "day": "Friday",
                    "date_text": None,
                    "time": "4 PM",
                    "daypart": None,
                    "provider": None,
                    "appointment_type": None,
                },
            ],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    try:
        plan, violations = planner.plan(
            world=build_world_model(
                get_scenario(
                    "autonomous-phone-diagnostic"
                )
            ),
            turn=ambiguous_turn,
        )
    finally:
        planner.close()
        client.close()

    assert (
        plan.action
        is PatientActionKind.SELECT_OPTION
    )

    assert plan.selected_option_index == 0
    assert violations == ()

    # This is the key assertion for this test:
    # unlike zero/one-compatible-option cases, Qwen was actually called.
    payload = captured["payload"]

    assert isinstance(
        payload,
        dict,
    )

    assert payload["model"] == "qwen3:14b"
    assert payload["stream"] is False
    assert payload["think"] is False

    assert (
        payload["format"]
        == ActionPlan.model_json_schema()
    )

    assert payload["options"] == {
        "temperature": 0,
    }
def test_fact_request_bypasses_planner_llm() -> None:
    """Structured fact requests should not be re-interpreted by Qwen."""

    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        raise AssertionError(
            "Semantically settled fact request reached planner LLM: "
            f"{request.method} {request.url}"
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            forbidden
        )
    )

    planner = QwenPatientPlanner(
        model="qwen3:14b",
        url="http://ollama.test/api/chat",
        client=client,
    )

    turn = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "patient_intake",
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

    try:
        plan, violations = planner.plan(
            world=build_world_model(
                get_scenario(
                    "autonomous-phone-diagnostic"
                )
            ),
            turn=turn,
        )
    finally:
        planner.close()
        client.close()

    assert calls == 0

    assert (
        plan.action
        is PatientActionKind.ANSWER_FACT
    )

    assert [
        item.value
        for item in plan.facts_to_answer
    ] == [
        "insurance"
    ]

    assert violations == ()


def test_wait_bypasses_planner_llm() -> None:
    """A semantic WAIT should cost no second model inference."""

    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        raise AssertionError(
            "Semantically settled WAIT reached planner LLM."
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            forbidden
        )
    )

    planner = QwenPatientPlanner(
        model="qwen3:14b",
        url="http://ollama.test/api/chat",
        client=client,
    )

    turn = TurnFrame.model_validate(
        {
            "speech_act": "status",
            "workflow": "scheduling",
            "requested_action": "wait",
            "response_required": False,
            "requested_facts": [],
            "other_requested_facts": [],
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": True,
            "confidence": 1.0,
        }
    )

    try:
        plan, violations = planner.plan(
            world=build_world_model(
                get_scenario(
                    "autonomous-phone-diagnostic"
                )
            ),
            turn=turn,
        )
    finally:
        planner.close()
        client.close()

    assert calls == 0
    assert plan.action is PatientActionKind.WAIT
    assert violations == ()


def test_unavailable_requested_fact_fails_closed() -> None:
    """A semantic hallucination must never become invented patient data."""

    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        raise AssertionError(
            "Unavailable fact fallback reached planner LLM."
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            forbidden
        )
    )

    planner = QwenPatientPlanner(
        model="qwen3:14b",
        url="http://ollama.test/api/chat",
        client=client,
    )

    turn = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "patient_intake",
            "requested_action": "answer_fact",
            "response_required": True,
            "requested_facts": [
                "phone_number"
            ],
            "other_requested_facts": [],
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 0.9,
        }
    )

    try:
        plan, violations = planner.plan(
            world=build_world_model(
                get_scenario(
                    "autonomous-phone-diagnostic"
                )
            ),
            turn=turn,
        )
    finally:
        planner.close()
        client.close()

    assert calls == 0

    assert (
        plan.action
        is PatientActionKind.CLARIFY
    )

    assert (
        plan.reason_code
        == "requested_fact_unavailable"
    )

    assert violations == ()


def test_zero_compatible_options_bypass_planner_llm() -> None:
    """Morning-only offers must deterministically request an alternative."""

    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        raise AssertionError(
            "Zero-compatible appointment set reached planner LLM."
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            forbidden
        )
    )

    planner = QwenPatientPlanner(
        model="qwen3:14b",
        url="http://ollama.test/api/chat",
        client=client,
    )

    try:
        plan, violations = planner.plan(
            world=build_world_model(
                get_scenario(
                    "autonomous-phone-diagnostic"
                )
            ),
            turn=build_multi_slot_turn(),
        )
    finally:
        planner.close()
        client.close()

    assert calls == 0

    assert (
        plan.action
        is PatientActionKind.REQUEST_ALTERNATIVE
    )

    assert plan.selected_option_index is None
    assert violations == ()


def test_exactly_one_compatible_option_bypasses_planner_llm() -> None:
    """A single policy-valid option is mechanically selected."""

    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        raise AssertionError(
            "Single-compatible appointment set reached planner LLM."
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            forbidden
        )
    )

    planner = QwenPatientPlanner(
        model="qwen3:14b",
        url="http://ollama.test/api/chat",
        client=client,
    )

    turn = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "scheduling",
            "requested_action": "choose_option",
            "response_required": True,
            "requested_facts": [],
            "other_requested_facts": [],
            "appointment_options": [
                {
                    "day": "Friday",
                    "date_text": None,
                    "time": "9 AM",
                    "daypart": None,
                    "provider": None,
                    "appointment_type": None,
                },
                {
                    "day": "Friday",
                    "date_text": None,
                    "time": "2:30 PM",
                    "daypart": None,
                    "provider": None,
                    "appointment_type": None,
                },
            ],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    try:
        plan, violations = planner.plan(
            world=build_world_model(
                get_scenario(
                    "autonomous-phone-diagnostic"
                )
            ),
            turn=turn,
        )
    finally:
        planner.close()
        client.close()

    assert calls == 0

    assert (
        plan.action
        is PatientActionKind.SELECT_OPTION
    )

    assert plan.selected_option_index == 1
    assert violations == ()


def test_constraint_validator_lists_compatible_options() -> None:
    world = build_world_model(
        get_scenario(
            "autonomous-phone-diagnostic"
        )
    )

    turn = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "scheduling",
            "requested_action": "choose_option",
            "response_required": True,
            "requested_facts": [],
            "other_requested_facts": [],
            "appointment_options": [
                {
                    "day": "Friday",
                    "date_text": None,
                    "time": "9 AM",
                    "daypart": None,
                    "provider": None,
                    "appointment_type": None,
                },
                {
                    "day": "Friday",
                    "date_text": None,
                    "time": "2:30 PM",
                    "daypart": None,
                    "provider": None,
                    "appointment_type": None,
                },
                {
                    "day": "Friday",
                    "date_text": None,
                    "time": "4 PM",
                    "daypart": None,
                    "provider": None,
                    "appointment_type": None,
                },
            ],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    assert (
        ConstraintValidator().compatible_option_indices(
            world=world,
            turn=turn,
        )
        == (
            1,
            2,
        )
    )
