from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy
from voiceprobe.v3.flow_controller import SchedulingFlowController
from voiceprobe.v3.flow_state import SchedulingFlowTracker
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.personas import (
    PersonaDecisionOverlay,
    PersonaRuntime,
    get_persona,
)


EXISTING_APPOINTMENT = (
    "You already have a new patient consultation booked for Tuesday, "
    "August twenty fifth at two fifteen PM. Would you like to keep this "
    "appointment, reschedule it to a different time, or cancel it?"
)

MULTIPLE_OPTIONS = (
    "Friday afternoon I have two fifteen PM, three PM, and three forty five PM. "
    "Which time works best for you?"
)


def test_live_failure_existing_appointment_is_not_new_booking_confirmation() -> None:
    tracker = SchedulingFlowTracker()

    snapshot = tracker.observe_remote_turn(EXISTING_APPOINTMENT)

    assert snapshot.complete is False
    assert snapshot.accepted_slot_text is None
    assert snapshot.booking_confirmation_text is None


def test_existing_appointment_prompt_requests_reschedule() -> None:
    decision = RoutineSchedulingPolicy().decide(EXISTING_APPOINTMENT)

    assert decision.kind is DecisionKind.STATE_OBJECTIVE
    assert decision.reason == "existing_appointment_reschedule"
    assert "reschedule" in decision.text.casefold()
    assert "friday afternoon" in decision.text.casefold()


def test_controller_continues_instead_of_ending_existing_appointment_call() -> None:
    controller = SchedulingFlowController()

    result = controller.decide_burst([EXISTING_APPOINTMENT])

    assert result.after.complete is False
    assert result.after.accepted_slot_text is None
    assert result.decision.kind is DecisionKind.STATE_OBJECTIVE
    assert result.decision.reason == "existing_appointment_reschedule"
    assert "reschedule" in result.decision.text.casefold()


def test_existing_appointment_can_reach_option_confuser_experiment() -> None:
    persona = PersonaRuntime(
        get_persona("option_confuser"),
        seed=6,
        sequence_id="exclude_then_restore",
    )

    controller = SchedulingFlowController(
        decision_overlay=PersonaDecisionOverlay(persona)
    )

    existing = controller.decide_burst([EXISTING_APPOINTMENT])

    assert existing.after.complete is False
    assert existing.decision.kind is DecisionKind.STATE_OBJECTIVE
    assert persona.selected_sequence_id is None

    options = controller.decide_burst([MULTIPLE_OPTIONS])

    assert options.after.complete is False
    assert options.decision.kind is DecisionKind.CLARIFY
    assert "anything except the earliest" in options.decision.text.casefold()
    assert "don't book" in options.decision.text.casefold()
    assert persona.selected_sequence_id == "exclude_then_restore"


def test_real_booking_confirmation_still_completes_after_slot_acceptance() -> None:
    tracker = SchedulingFlowTracker()

    tracker.record_slot_acceptance("2:15 PM")

    snapshot = tracker.observe_remote_turn(
        "Your appointment is booked for Friday at 2:15 PM."
    )

    assert snapshot.complete is True
    assert snapshot.accepted_slot_text == "2:15 PM"
    assert snapshot.booking_confirmation_text is not None
