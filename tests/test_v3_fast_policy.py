from voiceprobe.v3.corpus import load_regression_cases
from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy


def test_all_annotated_live_call_regressions() -> None:
    policy = RoutineSchedulingPolicy()

    for case in load_regression_cases():
        decision = policy.decide(case["agent_turn"])

        assert decision.kind.value == case["expected_kind"], (
            case["call_uuid"],
            case["ordinal"],
            case["agent_turn"],
            decision,
        )

        for expected_piece in case["expected_text_contains"]:
            assert expected_piece.casefold() in decision.text.casefold(), (
                case["call_uuid"],
                case["ordinal"],
                expected_piece,
                decision.text,
            )


def test_generic_objective_is_not_used_for_reason_for_visit() -> None:
    decision = RoutineSchedulingPolicy().decide("What is the reason for your visit?")

    assert "shoulder" in decision.text.casefold()
    assert "friday afternoon" not in decision.text.casefold()


def test_provider_choice_uses_stored_provider_preference() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "We have openings on Friday afternoon with two providers. "
        "Would you prefer Dr. A or Dr. B, or is the first available okay?"
    )

    assert decision.text == "First available is fine."


def test_provider_choice_handles_generic_offer_first_available_wording() -> None:
    policy = RoutineSchedulingPolicy()

    decision = policy.decide(
        "We have openings on Friday afternoon with doctor A and doctor B. "
        "Do you have a preference, or should I offer the first available?"
    )

    assert decision.kind.value == "answer_provider_preference"
    assert decision.text == "First available is fine."
    assert decision.reason == "provider_preference_requested"


def test_compatible_concrete_friday_afternoon_slot_is_accepted() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "I have Friday at 2:30 PM with the first available provider. "
        "Would that work for you?"
    )

    assert decision.kind.value == "grant_permission"
    assert decision.text == "Yes, please book the 2:30 PM slot."
    assert decision.reason == "compatible_concrete_slot_offered"


def test_concrete_non_friday_slot_is_declined() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "I have Thursday at 2:30 PM. Would that work for you?"
    )

    assert decision.kind.value == "decline_incompatible_offer"
    assert "Friday afternoon" in decision.text


def test_booking_confirmation_with_concrete_slot_does_not_fallback() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "Great, you're confirmed for Friday at 2:30 PM."
    )

    assert decision.kind.value == "wait"
    assert decision.reason == "booking_confirmation"


def test_spoken_pm_slot_is_accepted() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "I can schedule you Friday at two thirty PM. Would that work?"
    )

    assert decision.kind.value == "grant_permission"
    assert decision.reason == "compatible_concrete_slot_offered"


def test_day_or_provider_fallback_explicitly_chooses_earlier_week_afternoon() -> None:
    policy = RoutineSchedulingPolicy()

    decision = policy.decide(
        "Would you like me to check afternoon options on a different day "
        "or check with a different provider?"
    )

    assert decision.kind.value == "choose_search_branch"
    assert decision.reason == "choose_earlier_week_afternoon_search"
    assert "afternoon" in decision.text.casefold()
    assert "earlier in the week" in decision.text.casefold()
    assert decision.text != "Yes, please."


def test_direct_earlier_week_afternoon_search_is_explicit() -> None:
    policy = RoutineSchedulingPolicy()

    decision = policy.decide(
        "Would you like me to check afternoon options earlier in the week?"
    )

    assert decision.kind.value == "grant_permission"
    assert decision.reason == "allow_earlier_week_afternoon_search"
    assert "afternoon options earlier in the week" in decision.text.casefold()


def test_earlier_week_pm_slot_is_accepted_only_after_relaxation() -> None:
    policy = RoutineSchedulingPolicy()

    before = policy.decide("I have Tuesday at 2:30 PM. Would that work for you?")
    assert before.kind.value == "decline_incompatible_offer"

    policy.relax_day_constraint_for_afternoon()

    after = policy.decide("I have Tuesday at 2:30 PM. Would that work for you?")
    assert after.kind.value == "grant_permission"
    assert after.reason == "compatible_concrete_slot_offered"


def test_morning_remains_incompatible_after_day_relaxation() -> None:
    policy = RoutineSchedulingPolicy()
    policy.relax_day_constraint_for_afternoon()

    decision = policy.decide("I have Tuesday at 9 AM. Would that work for you?")

    assert decision.kind.value == "decline_incompatible_offer"
    assert "afternoon" in decision.text.casefold()
    assert "earlier in the week" in decision.text.casefold()


def test_weekend_pm_is_not_accepted_by_earlier_week_relaxation() -> None:
    policy = RoutineSchedulingPolicy()
    policy.relax_day_constraint_for_afternoon()

    decision = policy.decide("I have Saturday at 2:30 PM. Would that work for you?")

    assert decision.kind.value == "decline_incompatible_offer"
    assert decision.reason == "offer_outside_relaxed_earlier_week_window"


def test_live_run2_wrong_dob_trailing_comma_is_actionable() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "Your patient profile is set up, and your date of birth is "
        "July fourth two thousand for demo purposes. "
        "How may I help you today,"
    )

    assert decision.kind.value == "correct_and_state_objective"
    assert decision.reason == "correct_remote_fact_then_answer_open_intent"
    assert "April 12, 1998" in decision.text
    assert "Friday afternoon" in decision.text


def test_live_run2_can_i_help_you_today_restates_objective() -> None:
    decision = RoutineSchedulingPolicy().decide("can I help you today?")

    assert decision.kind.value == "state_objective"
    assert decision.reason == "open_ended_intent_question"
    assert "Friday afternoon" in decision.text


def test_live_run2_presence_check_recovers_with_objective() -> None:
    decision = RoutineSchedulingPolicy().decide("Are you still there?")

    assert decision.kind.value == "state_objective"
    assert decision.reason == "presence_check_restate_objective"
    assert "I'm here" in decision.text
    assert "Friday afternoon" in decision.text
