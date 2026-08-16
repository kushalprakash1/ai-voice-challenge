from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy
from voiceprobe.v3.flow_state import FlowStage, SchedulingFlowTracker
from voiceprobe.v3.models import DecisionKind, PolicyDecision


def test_profile_decision_communicates_profile_and_identity() -> None:
    tracker = SchedulingFlowTracker()
    decision = RoutineSchedulingPolicy().decide(
        (
            "Would you like to create a demo patient profile? "
            "I just need your first and last name to get started."
        )
    )

    snapshot = tracker.apply_decision(decision)

    assert FlowStage.PROFILE in snapshot.communicated
    assert FlowStage.IDENTITY in snapshot.communicated
    assert FlowStage.PROFILE not in snapshot.confirmed


def test_remote_profile_acknowledgement_confirms_profile_and_identity() -> None:
    tracker = SchedulingFlowTracker()

    snapshot = tracker.observe_remote_turn(
        "Your demo patient profile is set up."
    )

    assert FlowStage.PROFILE in snapshot.confirmed
    assert FlowStage.IDENTITY in snapshot.confirmed


def test_correct_dob_confirmation_is_recorded_but_wrong_dob_is_not() -> None:
    tracker = SchedulingFlowTracker()

    tracker.observe_remote_turn(
        "Your date of birth is July 4th, 2000."
    )
    assert FlowStage.DOB not in tracker.snapshot().confirmed

    tracker.observe_remote_turn(
        "Thanks for confirming your date of birth as April 12th, 1998."
    )
    assert FlowStage.DOB in tracker.snapshot().confirmed


def test_reason_and_type_combined_decision_updates_both_stages() -> None:
    tracker = SchedulingFlowTracker()
    decision = RoutineSchedulingPolicy().decide(
        (
            "Can you tell me the reason for your visit? "
            "Is this a routine checkup, new patient consultation, "
            "follow-up, or something else?"
        )
    )

    assert decision.kind == DecisionKind.ANSWER_VISIT_DETAILS

    snapshot = tracker.apply_decision(decision)

    assert FlowStage.VISIT_REASON in snapshot.communicated
    assert FlowStage.APPOINTMENT_TYPE in snapshot.communicated


def test_provider_preference_marks_provider_communicated() -> None:
    tracker = SchedulingFlowTracker()

    snapshot = tracker.apply_decision(
        PolicyDecision(
            kind=DecisionKind.ANSWER_PROVIDER_PREFERENCE,
            text="First available is fine.",
            reason="provider_preference_requested",
        )
    )

    assert FlowStage.PROVIDER in snapshot.communicated


def test_booking_completion_requires_explicit_remote_confirmation_with_time() -> None:
    tracker = SchedulingFlowTracker()

    tracker.record_slot_acceptance(
        "Friday, August 28th at 2:30 PM"
    )

    assert not tracker.snapshot().complete
    assert FlowStage.SLOT in tracker.snapshot().communicated
    assert FlowStage.CONFIRMATION not in tracker.snapshot().confirmed

    snapshot = tracker.observe_remote_turn(
        (
            "Your appointment is confirmed for Friday, August 28th "
            "at 2:30 PM."
        )
    )

    assert snapshot.complete
    assert FlowStage.SLOT in snapshot.confirmed
    assert FlowStage.CONFIRMATION in snapshot.confirmed
    assert snapshot.current_stage == FlowStage.COMPLETE


def test_status_with_time_does_not_false_confirm_booking() -> None:
    tracker = SchedulingFlowTracker()

    snapshot = tracker.observe_remote_turn(
        "I found an appointment Friday at 2:30 PM."
    )

    assert not snapshot.complete
    assert FlowStage.CONFIRMATION not in snapshot.confirmed
