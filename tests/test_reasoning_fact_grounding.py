from voiceprobe.reasoning.action_plan import (
    ActionPlan,
    PatientActionKind,
)
from voiceprobe.reasoning.action_verbalizer import (
    GenericActionVerbalizer,
)
from voiceprobe.reasoning.fact_grounding import (
    ground_fact_assertions,
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


def objective_turn(
    dob: str,
) -> TurnFrame:
    return TurnFrame.model_validate(
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
                    "value": dob,
                }
            ],
            "appointment_options": [],
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )


def test_wrong_dob_is_grounded_as_conflict() -> None:
    result = ground_fact_assertions(
        world=world(),
        turn=objective_turn(
            "July 4th, 2000"
        ),
    )

    assert len(
        result.conflicts
    ) == 1

    conflict = result.conflicts[0]

    assert (
        conflict.fact.value
        == "date_of_birth"
    )

    assert (
        conflict.asserted_value
        == "July 4th, 2000"
    )

    assert (
        conflict.authoritative_value
        == "April 12, 1998"
    )


def test_equivalent_dob_format_does_not_create_conflict() -> None:
    result = ground_fact_assertions(
        world=world(),
        turn=objective_turn(
            "April 12th, 1998"
        ),
    )

    assert result.conflicts == ()

    assert [
        item.value
        for item in result.matched_facts
    ] == [
        "date_of_birth"
    ]


def test_correction_and_primary_action_are_both_verbalized() -> None:
    caller_world = world()

    turn = objective_turn(
        "July 4th, 2000"
    )

    grounding = ground_fact_assertions(
        world=caller_world,
        turn=turn,
    )

    plan = ActionPlan(
        action=PatientActionKind.STATE_OBJECTIVE,
        reason_code="objective_requested",
        confidence=1.0,
    )

    text = (
        GenericActionVerbalizer().verbalize(
            world=caller_world,
            turn=turn,
            plan=plan,
            corrections=grounding.conflicts,
        )
    )

    assert text == (
        "Actually, my date of birth is April 12, 1998. "
        "I need to schedule an appointment for Friday afternoon."
    )


def test_correct_assertion_does_not_add_correction() -> None:
    caller_world = world()

    turn = objective_turn(
        "April 12, 1998"
    )

    grounding = ground_fact_assertions(
        world=caller_world,
        turn=turn,
    )

    plan = ActionPlan(
        action=PatientActionKind.STATE_OBJECTIVE,
        reason_code="objective_requested",
        confidence=1.0,
    )

    text = (
        GenericActionVerbalizer().verbalize(
            world=caller_world,
            turn=turn,
            plan=plan,
            corrections=grounding.conflicts,
        )
    )

    assert text == (
        "I need to schedule an appointment for Friday afternoon."
    )
