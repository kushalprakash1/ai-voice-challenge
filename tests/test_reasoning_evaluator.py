from voiceprobe.reasoning.action_plan import (
    ActionPlan,
    PatientActionKind,
)
from voiceprobe.reasoning.evaluator import (
    ConversationEvaluationState,
    EvaluationSeverity,
    evaluate_turn,
)
from voiceprobe.reasoning.fact_grounding import (
    FactConflict,
    FactGrounding,
)
from voiceprobe.reasoning.turn_frame import (
    RequestedFact,
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


def passive_turn() -> TurnFrame:
    return TurnFrame.model_validate(
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


def test_passive_wait_is_clean() -> None:
    findings, _ = evaluate_turn(
        world=world(),
        turn=passive_turn(),
        plan=ActionPlan(
            action=PatientActionKind.WAIT,
            reason_code="passive_turn_requires_no_response",
            confidence=1.0,
        ),
        grounding=FactGrounding(),
        patient_text="",
    )

    assert findings == ()


def test_premature_end_is_critical() -> None:
    findings, _ = evaluate_turn(
        world=world(),
        turn=passive_turn(),
        plan=ActionPlan(
            action=PatientActionKind.END_CONVERSATION,
            reason_code="bad_end",
            confidence=1.0,
        ),
        grounding=FactGrounding(),
        patient_text="Okay, bye.",
    )

    assert any(
        item.code
        == "premature_end_conversation"
        and item.severity
        is EvaluationSeverity.CRITICAL
        for item in findings
    )


def test_missing_fact_correction_is_reported() -> None:
    turn = TurnFrame.model_validate(
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
                    "value": "July 4, 2000",
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

    grounding = FactGrounding(
        conflicts=(
            FactConflict(
                fact=RequestedFact.DATE_OF_BIRTH,
                asserted_value="July 4, 2000",
                authoritative_value="April 12, 1998",
            ),
        )
    )

    findings, _ = evaluate_turn(
        world=world(),
        turn=turn,
        plan=ActionPlan(
            action=PatientActionKind.STATE_OBJECTIVE,
            reason_code="objective_requested",
            confidence=1.0,
        ),
        grounding=grounding,
        patient_text=(
            "I need to schedule an appointment for Friday afternoon."
        ),
    )

    assert any(
        item.code
        == "authoritative_correction_not_realized"
        for item in findings
    )


def test_selected_and_confirmed_slot_can_be_checked_across_turns() -> None:
    choose_turn = TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "scheduling",
            "requested_action": "choose_option",
            "response_required": True,
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

    _, state = evaluate_turn(
        world=world(),
        turn=choose_turn,
        plan=ActionPlan(
            action=PatientActionKind.SELECT_OPTION,
            selected_option_index=0,
            reason_code="only_compatible_option",
            confidence=1.0,
        ),
        grounding=FactGrounding(),
        patient_text="Friday at 2:30 PM works for me.",
        state=ConversationEvaluationState(),
    )

    confirmed_turn = TurnFrame.model_validate(
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
                "time": "4:00 PM",
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

    findings, _ = evaluate_turn(
        world=world(),
        turn=confirmed_turn,
        plan=ActionPlan(
            action=PatientActionKind.END_CONVERSATION,
            reason_code="booking_confirmed",
            confidence=1.0,
        ),
        grounding=FactGrounding(),
        patient_text="Okay, thank you. Bye.",
        state=state,
    )

    assert any(
        item.code
        == "confirmed_slot_differs_from_selected"
        and item.severity
        is EvaluationSeverity.CRITICAL
        for item in findings
    )
