from __future__ import annotations

import httpx

from voiceprobe.reasoning.action_plan import PatientActionKind
from voiceprobe.reasoning.action_verbalizer import GenericActionVerbalizer
from voiceprobe.reasoning.constraint_validator import ConstraintValidator
from voiceprobe.reasoning.planner import QwenPatientPlanner
from voiceprobe.reasoning.turn_frame import RequestedAction, TurnFrame
from voiceprobe.reasoning.world_model import build_world_model
from voiceprobe.scenarios.catalog import get_scenario


def failed_live_turn_frame() -> TurnFrame:
    return TurnFrame.model_validate(
        {
            "speech_act": "question",
            "workflow": "scheduling",
            "requested_action": "choose_presented_choice",
            "response_required": True,
            "requested_facts": [],
            "other_requested_facts": [],
            "stated_facts": [],
            "proposed_workflow": None,
            "appointment_options": [],
            "presented_choices": [
                {
                    "label": "look at afternoon options on another day",
                    "kind": "search_availability",
                    "day": None,
                    "date_text": None,
                    "time": None,
                    "daypart": "afternoon",
                    "provider": None,
                    "appointment_type": None,
                },
                {
                    "label": "check the following Friday, August 28th",
                    "kind": "search_availability",
                    "day": None,
                    "date_text": "August 28th",
                    "time": None,
                    "daypart": None,
                    "provider": None,
                    "appointment_type": None,
                },
            ],
            "confirmed_appointment": None,
            "booking_confirmed": False,
            "conversation_end_requested": False,
            "agent_is_still_working": False,
            "confidence": 1.0,
        }
    )


def test_general_choice_is_separate_from_booking_slot_choice() -> None:
    turn = failed_live_turn_frame()
    assert turn.requested_action is RequestedAction.CHOOSE_PRESENTED_CHOICE
    assert turn.appointment_options == []
    assert len(turn.presented_choices) == 2


def test_other_day_conflicts_but_next_friday_is_compatible() -> None:
    world = build_world_model(get_scenario("autonomous-phone-diagnostic"))
    turn = failed_live_turn_frame()
    compatible = ConstraintValidator().compatible_presented_choice_indices(
        world=world,
        turn=turn,
    )
    assert compatible == (1,)


def test_planner_selects_only_compatible_general_choice_without_qwen() -> None:
    def fail_if_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Qwen should not be called here")

    client = httpx.Client(transport=httpx.MockTransport(fail_if_called))
    try:
        planner = QwenPatientPlanner(
            model="test-model",
            url="http://ollama.invalid/api/chat",
            client=client,
        )
        world = build_world_model(get_scenario("autonomous-phone-diagnostic"))
        turn = failed_live_turn_frame()
        plan, repaired = planner.plan(
            world=world,
            turn=turn,
            recent_actions=(),
        )
        assert repaired == ()
        assert plan.action is PatientActionKind.SELECT_PRESENTED_CHOICE
        assert plan.selected_choice_index == 1
    finally:
        client.close()


def test_general_search_choice_carries_forward_afternoon() -> None:
    def fail_if_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Qwen should not be called here")

    client = httpx.Client(transport=httpx.MockTransport(fail_if_called))
    try:
        planner = QwenPatientPlanner(
            model="test-model",
            url="http://ollama.invalid/api/chat",
            client=client,
        )
        world = build_world_model(get_scenario("autonomous-phone-diagnostic"))
        turn = failed_live_turn_frame()
        plan, _ = planner.plan(
            world=world,
            turn=turn,
            recent_actions=(),
        )
    finally:
        client.close()

    patient_text = GenericActionVerbalizer().verbalize(
        world=world,
        turn=turn,
        plan=plan,
    )
    assert patient_text == (
        "Please check Friday, August 28th for afternoon appointments."
    )
