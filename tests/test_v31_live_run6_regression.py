from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy
from voiceprobe.v3.flow_controller import SchedulingFlowController
from voiceprobe.v3.models import DecisionKind


BOOKING_CONFIRMATION = (
    "You're booked for a new patient consultation with Doovie Hauser "
    "on Tuesday, August twenty fifth at two fifteen PM at Pivot Point "
    "Orthopedics. Please bring your photo ID, insurance card, a list "
    "of current medications, and any imaging discs if you have them."
)

TRAILING_HELP = (
    "Is there anything else I can help you with today?"
)


def test_booking_confirmation_outranks_incidental_insurance_card():
    decision = RoutineSchedulingPolicy().decide(
        BOOKING_CONFIRMATION
    )

    assert decision.kind == DecisionKind.WAIT
    assert decision.reason == "booking_confirmation"
    assert decision.text == ""


def test_completed_booking_burst_suppresses_trailing_help_question():
    controller = SchedulingFlowController()

    controller.tracker.record_slot_acceptance(
        "two fifteen PM"
    )

    result = controller.decide_burst(
        [
            BOOKING_CONFIRMATION,
            TRAILING_HELP,
        ]
    )

    assert result.after.complete is True

    assert result.after.accepted_slot_text == (
        "two fifteen PM"
    )

    assert result.after.booking_confirmation_text == (
        BOOKING_CONFIRMATION
    )

    assert result.decision.kind == DecisionKind.WAIT
    assert result.decision.reason == "booking_confirmation"
    assert result.actionable_turn is None


def test_booking_confirmation_does_not_answer_blue_cross():
    controller = SchedulingFlowController()

    controller.tracker.record_slot_acceptance(
        "two fifteen PM"
    )

    result = controller.decide_burst(
        [
            BOOKING_CONFIRMATION,
            TRAILING_HELP,
        ]
    )

    assert result.decision.text != "Blue Cross."
    assert result.decision.kind != DecisionKind.ANSWER_FACT


def test_capability_menu_does_not_request_insurance():
    decision = RoutineSchedulingPolicy().decide(
        "I can help with appointments, medication refills, insurance updates, and more."
    )

    assert decision.reason != "insurance_requested"
    assert decision.text != "Blue Cross."
