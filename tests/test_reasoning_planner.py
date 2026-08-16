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


def test_workflow_proposal_reaches_planner_context() -> None:
    """Planner must reason over the proposed workflow, not raw phrase rules."""

    captured: dict[str, object] = {}

    response_plan = {
        "action": "grant_permission",
        "selected_option_index": None,
        "facts_to_answer": [],
        "reason_code": "workflow_enables_objective",
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

    turn = TurnFrame.model_validate(
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

    assert (
        plan.action
        is PatientActionKind.GRANT_PERMISSION
    )

    assert violations == ()

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

    context = json.loads(
        messages[-1]["content"]
    )

    proposal = (
        context[
            "remote_turn"
        ][
            "proposed_workflow"
        ]
    )

    assert proposal == {
        "kind": "profile_setup",
        "description": "create a demo patient profile",
        "requirement": "optional",
    }


def test_joint_first_last_request_can_use_authoritative_full_name() -> None:
    """Do not guess name components when authoritative full name suffices."""

    from voiceprobe.reasoning.world_model import (
        PatientWorldModel,
    )

    caller_world = PatientWorldModel(
        scenario_id="full-name-test",
        objective="Schedule an appointment.",
        facts={
            "name": "Jordan Lee",
        },
        constraints=[],
    )

    turn = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "patient_intake",
            "requested_action": "answer_fact",
            "response_required": True,
            "requested_facts": [
                "first_name",
                "last_name",
            ],
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

    planner = QwenPatientPlanner(
        model="qwen3:14b",
        url="http://unused.invalid/api/chat",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(
                    AssertionError(
                        "Deterministic name resolution reached Qwen."
                    )
                )
            )
        ),
    )

    try:
        plan, violations = planner.plan(
            world=caller_world,
            turn=turn,
        )
    finally:
        planner.close()

    assert (
        plan.action
        is PatientActionKind.ANSWER_FACT
    )

    assert [
        fact.value
        for fact in plan.facts_to_answer
    ] == [
        "full_name"
    ]

    assert violations == ()


def test_permission_plan_can_carry_requested_full_name() -> None:
    """Workflow consent and a fact answer may coexist in one ActionPlan."""

    from voiceprobe.reasoning.world_model import (
        PatientWorldModel,
    )

    caller_world = PatientWorldModel(
        scenario_id="compound-turn-test",
        objective="Schedule an appointment.",
        facts={
            "name": "Maya Patel",
        },
        constraints=[],
    )

    response_plan = {
        "action": "grant_permission",
        "selected_option_index": None,
        "facts_to_answer": [],
        "reason_code": "workflow_enables_objective",
        "confidence": 1.0,
    }

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
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

    turn = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "profile_setup",
            "requested_action": "grant_permission",
            "response_required": True,
            "requested_facts": [
                "first_name",
                "last_name",
            ],
            "other_requested_facts": [],
            "stated_facts": [],
            "proposed_workflow": {
                "kind": "profile_setup",
                "description": "create a patient profile",
                "requirement": "optional",
            },
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    try:
        plan, violations = planner.plan(
            world=caller_world,
            turn=turn,
        )
    finally:
        planner.close()
        client.close()

    assert (
        plan.action
        is PatientActionKind.GRANT_PERMISSION
    )

    assert [
        fact.value
        for fact in plan.facts_to_answer
    ] == [
        "full_name"
    ]

    assert violations == ()


def test_model_fact_payload_is_replaced_by_authoritative_resolution() -> None:
    """Qwen may choose an action but never which patient facts are disclosed."""

    from voiceprobe.reasoning.world_model import (
        PatientWorldModel,
    )

    caller_world = PatientWorldModel(
        scenario_id="maya-compound-test",
        objective="Complete an identity and insurance check.",
        facts={
            "name": "Maya Patel",
        },
        constraints=[],
    )

    # Deliberately simulate the bad model output observed in historical
    # replay: it tries to disclose unavailable component fields.
    qwen_plan = {
        "action": "grant_permission",
        "selected_option_index": None,
        "facts_to_answer": [
            "first_name",
            "last_name",
        ],
        "reason_code": "workflow_enables_objective",
        "confidence": 1.0,
    }

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        qwen_plan
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

    turn = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "profile_setup",
            "requested_action": "grant_permission",
            "response_required": True,
            "requested_facts": [
                "first_name",
                "last_name",
            ],
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

    try:
        plan, violations = planner.plan(
            world=caller_world,
            turn=turn,
        )
    finally:
        planner.close()
        client.close()

    assert (
        plan.action
        is PatientActionKind.GRANT_PERMISSION
    )

    assert [
        fact.value
        for fact in plan.facts_to_answer
    ] == [
        "full_name"
    ]

    assert violations == ()


def test_confirmed_booking_ends_cleanly_without_reselecting_slot() -> None:
    """A booked slot is not another invitation to select that slot."""

    turn = TurnFrame.model_validate(
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

    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(
            "Confirmed booking should not require Qwen planning."
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            forbidden
        )
    )

    planner = QwenPatientPlanner(
        model="qwen3:14b",
        url="http://unused.invalid/api/chat",
        client=client,
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
        is PatientActionKind.END_CONVERSATION
    )

    assert plan.selected_option_index is None
    assert violations == ()


def test_select_option_is_invalid_without_choice_request() -> None:
    turn = TurnFrame.model_validate(
        {
            "speech_act": "information",
            "workflow": "scheduling",
            "requested_action": "none",
            "response_required": False,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [],
            "proposed_workflow": None,
            "appointment_options": [
                {
                    "day": "Friday",
                    "date_text": None,
                    "time": "2:30 PM",
                    "daypart": None,
                    "provider": None,
                    "appointment_type": None,
                }
            ],
            "confirmed_appointment": None,
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    plan = ActionPlan(
        action=PatientActionKind.SELECT_OPTION,
        selected_option_index=0,
        reason_code="bad_reselection",
        confidence=1.0,
    )

    violations = ConstraintValidator().validate(
        world=build_world_model(
            get_scenario(
                "autonomous-phone-diagnostic"
            )
        ),
        turn=turn,
        plan=plan,
    )

    assert any(
        item.code
        == "selection_without_choice_request"
        for item in violations
    )


def test_passive_none_turn_waits_instead_of_ending_conversation() -> None:
    """A passive acknowledgement is not evidence that the mission is done."""

    turn = TurnFrame.model_validate(
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
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(
            "Passive NONE turn should bypass Qwen."
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            forbidden
        )
    )

    planner = QwenPatientPlanner(
        model="qwen3:14b",
        url="http://unused.invalid/api/chat",
        client=client,
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
        is PatientActionKind.WAIT
    )

    assert (
        plan.reason_code
        == "passive_turn_requires_no_response"
    )

    assert violations == ()


def test_explicit_conversation_end_still_ends() -> None:
    """The passive-turn WAIT rule must not swallow a real goodbye."""

    turn = TurnFrame.model_validate(
        {
            "speech_act": "goodbye",
            "workflow": "scheduling",
            "requested_action": "none",
            "response_required": False,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [],
            "proposed_workflow": None,
            "appointment_options": [],
            "confirmed_appointment": None,
            "booking_confirmed": False,
            "conversation_end_requested": True,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(
                AssertionError(
                    "Explicit conversation end should bypass Qwen."
                )
            )
        )
    )

    planner = QwenPatientPlanner(
        model="qwen3:14b",
        url="http://unused.invalid/api/chat",
        client=client,
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

    assert (
        plan.action
        is PatientActionKind.END_CONVERSATION
    )

    assert violations == ()
