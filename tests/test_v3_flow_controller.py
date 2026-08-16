from voiceprobe.v3.flow_controller import SchedulingFlowController
from voiceprobe.v3.flow_state import FlowStage
from voiceprobe.v3.models import DecisionKind


def test_latest_live_failure_sequence_advances_correct_fields() -> None:
    controller = SchedulingFlowController()

    first = controller.decide_burst(
        [
            (
                "Thank you for calling Pivot Point Orthopedics. "
                "Would you like to create a demo patient profile? "
                "I just need your first and last name to get started."
            )
        ]
    )

    assert first.decision.kind == DecisionKind.CREATE_PROFILE
    assert FlowStage.PROFILE in first.after.communicated
    assert FlowStage.IDENTITY in first.after.communicated

    second = controller.decide_burst(
        [
            "Your demo patient profile is set up and your date of birth is July 4th, 2000. How can I help you today?"
        ]
    )

    assert second.decision.kind == DecisionKind.CORRECT_AND_STATE_OBJECTIVE
    assert FlowStage.PROFILE in second.after.confirmed
    assert FlowStage.DOB in second.after.communicated
    assert FlowStage.DATE_TIME in second.after.communicated

    third = controller.decide_burst(
        [
            "Thanks, Alex.",
            "Let me check available appointments for you on Friday afternoon.",
            "Thanks for confirming your date of birth of April 12th, 1998.",
            (
                "Can you tell me the reason for your visit? "
                "For example, is this for a routine checkup, "
                "a new patient consultation, a follow-up, or something else?"
            ),
        ]
    )

    assert third.decision.kind == DecisionKind.ANSWER_VISIT_DETAILS
    assert FlowStage.DOB in third.after.confirmed
    assert FlowStage.VISIT_REASON in third.after.communicated
    assert FlowStage.APPOINTMENT_TYPE in third.after.communicated


def test_provider_prompt_updates_provider_without_repeating_datetime() -> None:
    controller = SchedulingFlowController()

    result = controller.decide_burst(
        [
            (
                "We have openings on Friday afternoon with two providers. "
                "Would you prefer Dr. A or Dr. B, or is the first available okay?"
            )
        ]
    )

    assert result.decision.kind == DecisionKind.ANSWER_PROVIDER_PREFERENCE
    assert result.decision.text == "First available is fine."
    assert FlowStage.PROVIDER in result.after.communicated


def test_august_28_branch_keeps_datetime_progress_without_booking() -> None:
    controller = SchedulingFlowController()

    result = controller.decide_burst(
        [
            "There are no Friday afternoon openings on August 21st.",
            (
                "Would you like to see afternoon openings on Friday, "
                "August 28th, or check other days in the future?"
            ),
        ]
    )

    assert result.decision.kind == DecisionKind.CHOOSE_SEARCH_BRANCH
    assert FlowStage.DATE_TIME in result.after.communicated
    assert FlowStage.SLOT not in result.after.confirmed
    assert not result.after.complete

def test_concrete_slot_acceptance_then_remote_confirmation_completes_flow() -> None:
    controller = SchedulingFlowController()

    accepted = controller.decide_burst(
        [
            (
                "I have Friday at 2:30 PM with the first available provider. "
                "Would that work for you?"
            )
        ]
    )

    assert accepted.decision.kind == DecisionKind.GRANT_PERMISSION
    assert accepted.after.accepted_slot_text == "2:30 PM"
    assert FlowStage.SLOT in accepted.after.communicated
    assert not accepted.after.complete

    confirmed = controller.decide_burst(
        ["Great, you're confirmed for Friday at 2:30 PM."]
    )

    assert confirmed.decision.kind == DecisionKind.WAIT
    assert confirmed.after.complete
    assert FlowStage.CONFIRMATION in confirmed.after.confirmed
    assert confirmed.after.accepted_slot_text == "2:30 PM"
    assert confirmed.after.booking_confirmation_text is not None


def test_spoken_slot_confirmation_completes_flow() -> None:
    controller = SchedulingFlowController()

    controller.decide_burst(
        ["I can schedule you Friday at two thirty PM. Would that work?"]
    )
    confirmed = controller.decide_burst(
        ["You're booked for Friday at two thirty PM."]
    )

    assert confirmed.after.complete
    assert confirmed.after.accepted_slot_text.casefold() == "two thirty pm"


def test_live_afternoon_fallback_reaches_non_friday_booking() -> None:
    controller = SchedulingFlowController()

    branch = controller.decide_burst(
        [
            (
                "Would you like me to check afternoon options on a different "
                "day or check with a different provider?"
            )
        ]
    )

    assert branch.decision.kind == DecisionKind.CHOOSE_SEARCH_BRANCH
    assert branch.decision.reason == "choose_earlier_week_afternoon_search"
    assert branch.after.allow_earlier_week_afternoons
    assert "earlier in the week" in branch.decision.text.casefold()

    permission = controller.decide_burst(
        ["Would you like me to check afternoon options earlier in the week?"]
    )

    assert permission.decision.kind == DecisionKind.GRANT_PERMISSION
    assert permission.after.allow_earlier_week_afternoons

    accepted = controller.decide_burst(
        ["I have Tuesday at 2:30 PM. Would that work for you?"]
    )

    assert accepted.decision.kind == DecisionKind.GRANT_PERMISSION
    assert accepted.decision.reason == "compatible_concrete_slot_offered"
    assert accepted.after.accepted_slot_text == "2:30 PM"
    assert FlowStage.SLOT in accepted.after.communicated
    assert not accepted.after.complete

    confirmed = controller.decide_burst(
        ["Great, you're confirmed for Tuesday at 2:30 PM."]
    )

    assert confirmed.decision.kind == DecisionKind.WAIT
    assert confirmed.after.complete
    assert FlowStage.CONFIRMATION in confirmed.after.confirmed
    assert confirmed.after.accepted_slot_text == "2:30 PM"
    assert confirmed.after.booking_confirmation_text is not None


def test_same_burst_fallback_then_earlier_week_slot_is_accepted() -> None:
    controller = SchedulingFlowController()

    result = controller.decide_burst(
        [
            (
                "Would you like me to check afternoon options on a different "
                "day or check with a different provider?"
            ),
            "I have Wednesday at 3:15 PM. Would that work for you?",
        ]
    )

    assert result.after.allow_earlier_week_afternoons
    assert result.decision.kind == DecisionKind.GRANT_PERMISSION
    assert result.decision.reason == "compatible_concrete_slot_offered"
    assert result.after.accepted_slot_text == "3:15 PM"


def test_relaxed_day_never_relaxes_afternoon_constraint() -> None:
    controller = SchedulingFlowController()

    controller.decide_burst(
        ["Should I check afternoon options earlier in the week?"]
    )

    result = controller.decide_burst(
        ["I have Monday at 9 AM. Would that work for you?"]
    )

    assert result.after.allow_earlier_week_afternoons
    assert result.decision.kind == DecisionKind.DECLINE_INCOMPATIBLE_OFFER
    assert "afternoon" in result.decision.text.casefold()
    assert result.after.accepted_slot_text is None
    assert not result.after.complete

def test_live_run2_split_open_intent_recovers_without_fallback() -> None:
    controller = SchedulingFlowController()

    first = controller.decide_burst(
        [
            (
                "This call may be recorded for quality and training purposes. "
                "Thank you for calling Pivot Point Orthopedics. "
                "Would you like to create a demo patient profile?"
            ),
            "I just need your first and last name to get started.",
        ]
    )
    assert first.decision.kind == DecisionKind.ANSWER_FACT

    second = controller.decide_burst(
        [
            (
                "Your patient profile is set up, and your date of birth is "
                "July fourth two thousand for demo purposes. "
                "How may I help you today,"
            )
        ]
    )

    assert second.decision.kind == DecisionKind.CORRECT_AND_STATE_OBJECTIVE
    assert second.decision.reason == "correct_remote_fact_then_answer_open_intent"
    assert FlowStage.DOB in second.after.communicated
    assert FlowStage.DATE_TIME in second.after.communicated

    third = controller.decide_burst(
        [
            "Thanks, Alex.",
            "can I help you today?",
        ]
    )

    assert third.decision.kind == DecisionKind.STATE_OBJECTIVE
    assert third.decision.reason == "open_ended_intent_question"
    assert third.decision.text
    assert "Friday afternoon" in third.decision.text

    fourth = controller.decide_burst(["Are you still there?"])

    assert fourth.decision.kind == DecisionKind.STATE_OBJECTIVE
    assert fourth.decision.reason == "presence_check_restate_objective"
    assert fourth.decision.text
