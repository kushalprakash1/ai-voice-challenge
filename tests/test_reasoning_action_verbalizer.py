from voiceprobe.reasoning.action_plan import (
    ActionPlan,
    PatientActionKind,
)
from voiceprobe.reasoning.action_verbalizer import (
    GenericActionVerbalizer,
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


def world():
    return build_world_model(
        get_scenario(
            "autonomous-phone-diagnostic"
        )
    )


def frame(
    *,
    options=None,
):
    return TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "scheduling",
            "requested_action": (
                "choose_option"
                if options
                else "clarify"
            ),
            "response_required": True,
            "requested_facts": [],
            "other_requested_facts": [],
            "appointment_options": (
                options or []
            ),
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )


def test_wait_produces_silence() -> None:
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

    plan = ActionPlan(
        action=PatientActionKind.WAIT,
        reason_code="wait",
        confidence=1.0,
    )

    assert (
        GenericActionVerbalizer().verbalize(
            world=world(),
            turn=turn,
            plan=plan,
        )
        == ""
    )


def test_insurance_answer_comes_from_world_truth() -> None:
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

    plan = ActionPlan(
        action=PatientActionKind.ANSWER_FACT,
        facts_to_answer=[
            "insurance"
        ],
        reason_code="fact",
        confidence=1.0,
    )

    assert (
        GenericActionVerbalizer().verbalize(
            world=world(),
            turn=turn,
            plan=plan,
        )
        == "Blue Cross."
    )


def test_request_alternative_states_generic_constraints() -> None:
    turn = frame(
        options=[
            {
                "day": "Friday",
                "date_text": None,
                "time": "9 AM",
                "daypart": None,
                "provider": None,
                "appointment_type": None,
            }
        ]
    )

    plan = ActionPlan(
        action=PatientActionKind.REQUEST_ALTERNATIVE,
        reason_code="no_compatible_option",
        confidence=1.0,
    )

    assert (
        GenericActionVerbalizer().verbalize(
            world=world(),
            turn=turn,
            plan=plan,
        )
        ==
        (
            "Those options don't work for me. "
            "Do you have anything Friday afternoon?"
        )
    )


def test_select_option_names_the_actual_option() -> None:
    turn = frame(
        options=[
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
        ]
    )

    plan = ActionPlan(
        action=PatientActionKind.SELECT_OPTION,
        selected_option_index=1,
        reason_code="compatible_option",
        confidence=1.0,
    )

    assert (
        GenericActionVerbalizer().verbalize(
            world=world(),
            turn=turn,
            plan=plan,
        )
        == "Friday at 2:30 PM works for me."
    )


def test_state_objective_is_generic() -> None:
    turn = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "unknown",
            "requested_action": "state_objective",
            "response_required": True,
            "requested_facts": [],
            "other_requested_facts": [],
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )

    plan = ActionPlan(
        action=PatientActionKind.STATE_OBJECTIVE,
        reason_code="objective_requested",
        confidence=1.0,
    )

    assert (
        GenericActionVerbalizer().verbalize(
            world=world(),
            turn=turn,
            plan=plan,
        )
        ==
        "I need to schedule an appointment for Friday afternoon."
    )


def test_permission_and_full_name_are_composed() -> None:
    from voiceprobe.reasoning.world_model import (
        PatientWorldModel,
    )

    caller_world = PatientWorldModel(
        scenario_id="compound-verbalizer-test",
        objective="Schedule an appointment.",
        facts={
            "name": "Maya Patel",
        },
        constraints=[],
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

    plan = ActionPlan(
        action=PatientActionKind.GRANT_PERMISSION,
        facts_to_answer=[
            "full_name"
        ],
        reason_code="workflow_enables_objective",
        confidence=1.0,
    )

    assert (
        GenericActionVerbalizer().verbalize(
            world=caller_world,
            turn=turn,
            plan=plan,
        )
        == "Yes, please. Maya Patel."
    )
