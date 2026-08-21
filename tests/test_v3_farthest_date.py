from voiceprobe.v3.farthest_date import (
    ALTERNATIVE_SEARCH_RESPONSE,
    BOOKING_ACCEPTANCE,
    DATE_PREFERENCE_RESPONSE,
    FALLBACK_TEXT,
    FIRST_VISIT_RESPONSE,
    OBJECTIVE_TEXT,
    REASON_RESPONSE,
    RESCHEDULE_CONFIRMATION_RESPONSE,
    RESCHEDULE_RESPONSE,
    TIME_PREFERENCE_RESPONSE,
    VERIFICATION_TEXT,
    CONTINUE_LATEST_SEARCH_RESPONSE,
    FarthestDatePolicy,
    farthest_date_phrase_inventory,
)
from voiceprobe.v3.flow_controller import SchedulingFlowController
from voiceprobe.scenarios.catalog import get_scenario


def _respond(controller, policy, target):
    result = controller.decide_burst([target])
    policy.mark_decision_spoken(result.decision)
    return result


def test_exact_call6_transcript_uses_stable_scheduling_controller():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)
    exchanges = (
        (("This call may be recorded for quality assurance. Would you like to create a demo patient profile?",), "Yes, please. My name is Chitragupta Subramnian Singh."),
        (("Your patient profile is set up. How may I help you today?",), OBJECTIVE_TEXT),
        (("You mentioned wanting a new patient consultation. Just to confirm, is this your first time visiting Pivot Point Orthopedics?",), FIRST_VISIT_RESPONSE),
        (("Do you have a provider preference, or is first available okay?",), "First available is fine."),
        ((
            "You already have a new patient consultation booked for Tuesday, August twenty fifth at two fifteen PM.",
            "Would you like to keep this appointment, reschedule it, or cancel it?",
        ), RESCHEDULE_RESPONSE),
        (("Why do you need to reschedule?",), REASON_RESPONSE),
        (("What date would you prefer?",), DATE_PREFERENCE_RESPONSE),
        (("What time would you prefer?",), TIME_PREFERENCE_RESPONSE),
        (("I can offer December 18, 2026 at 2:15 PM. Would that work?",), VERIFICATION_TEXT),
        (("Yes, that is the latest date I can currently book.",), BOOKING_ACCEPTANCE),
    )
    spoken = []
    for target_burst, expected in exchanges:
        result = controller.decide_burst(target_burst)
        policy.mark_decision_spoken(result.decision)
        spoken.append(result.decision.text)
        assert result.decision.text == expected
        assert result.decision.kind.value != "fallback"

    confirmed = controller.decide_burst(
        ["Your appointment for December 18, 2026 at 2:15 PM is booked and confirmed."]
    )
    assert confirmed.decision.requires_response is False
    assert confirmed.after.complete is True
    assert confirmed.after.accepted_slot_text == "2:15 PM"
    assert len(spoken) == len(set(spoken))


def test_first_time_visiting_routes_to_first_visit_not_appointment_type():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)
    result = _respond(
        controller,
        policy,
        "You mentioned wanting a new patient consultation. Just to confirm, is this your first time visiting Pivot Point Orthopedics?",
    )
    assert result.decision.reason == "call6:first_visit"
    assert result.decision.text == FIRST_VISIT_RESPONSE
    assert result.decision.reason != "appointment_type_requested"


def test_establish_care_question_routes_to_first_visit():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)
    result = _respond(controller, policy, "Are you looking to establish care?")
    assert result.decision.reason == "call6:first_visit"
    assert result.decision.text == FIRST_VISIT_RESPONSE


def test_keep_reschedule_cancel_choice_is_deterministic_not_fallback():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)
    result = _respond(controller, policy, "Would you like to keep this appointment, reschedule it, or cancel it?")
    assert result.decision.reason == "call6:reschedule_choice"
    assert result.decision.text == RESCHEDULE_RESPONSE
    assert result.decision.kind.value != "fallback"


def test_existing_appointment_reschedule_confirmation_precedes_slot_policy():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)
    for prompt in (
        "Is this the appointment you'd like to reschedule for a later date?",
        "Would you like to move this appointment to a later date?",
        "Do you want to reschedule this appointment?",
        "You have an appointment on Tuesday, August twenty fifth at two fifteen PM. Is this the appointment you want to reschedule?",
    ):
        result = controller.decide_burst([prompt])
        assert result.decision.text == RESCHEDULE_CONFIRMATION_RESPONSE
        assert result.decision.reason == "call6:confirm_existing_appointment_to_reschedule"


def test_call6_has_no_day_or_time_constraint():
    facts = get_scenario("farthest-date-scheduling").facts
    assert facts.preferred_day is None
    assert facts.preferred_time is None


def test_alternative_day_or_time_continues_search_instead_of_transfer():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)
    prompts = (
        "There are no Friday afternoon appointments available. Would you like to try a different day or time, or be transferred?",
        "Would you like to look for openings on a different day or time, or speak with someone?",
        "Would you like me to check another day or time?",
    )
    for prompt in prompts:
        result = controller.decide_burst([prompt])
        assert result.decision.text == ALTERNATIVE_SEARCH_RESPONSE
        assert result.decision.reason == "call6:continue_unconstrained_latest_search"


def test_non_friday_concrete_offers_never_leak_friday_constraint():
    for weekday in ("Tuesday", "Wednesday", "Thursday"):
        policy = FarthestDatePolicy()
        controller = SchedulingFlowController(decision_overlay=policy)
        result = controller.decide_burst(
            [f"I can offer {weekday}, December 18, 2026 at 2:15 PM. Would that work?"]
        )
        assert result.decision.reason != "non_friday_offer_conflicts_with_day_constraint"
        assert "Friday afternoon" not in result.decision.text


def test_exact_live_later_date_choices_use_complete_burst_before_shared_policy():
    prompts = (
        (
            "The furthest date I can currently book is Tuesday, August twenty fifth. I have openings at ten thirty AM, eleven fifteen AM, and twelve PM. Would you like to move your appointment to one of these times,",
            "or would you like to try for a later date?",
            "or a different day of the week.",
        ),
        (
            "Would you like to move your new patient appointment to ten thirty AM, eleven fifteen AM, or twelve PM on Tuesday, August twenty fifth?",
            "or would you prefer to look for dates further in the future?",
        ),
    )
    for burst in prompts:
        policy = FarthestDatePolicy()
        result = SchedulingFlowController(decision_overlay=policy).decide_burst(burst)
        assert result.decision.reason == "call6:continue_latest_search"
        assert result.decision.text == CONTINUE_LATEST_SEARCH_RESPONSE
        assert result.decision.kind.value != "fallback"
        assert "Friday" not in result.decision.text


def test_latest_asserted_final_offer_accepts_grounded_slot_instead_of_transfer():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)
    result = controller.decide_burst((
        "The latest date I can currently book is Tuesday, August twenty fifth. The available times are ten thirty AM, eleven fifteen AM, and twelve PM.",
        "Would you like to move your appointment to one of these times,",
        "or would you like to speak with someone at the clinic for more options?",
    ))
    assert result.decision.reason == "compatible_concrete_slot_offered"
    assert result.decision.text == "Yes, please book the ten thirty AM slot."
    assert result.after.accepted_slot_text == "ten thirty AM"
    assert result.decision.kind.value != "fallback"


def test_reachable_inventory_contains_all_exact_call6_responses_and_no_broken_ones():
    inventory = farthest_date_phrase_inventory()
    assert "First available is fine." in inventory
    assert RESCHEDULE_CONFIRMATION_RESPONSE in inventory
    assert ALTERNATIVE_SEARCH_RESPONSE in inventory
    assert "Could you rephrase that scheduling question?" not in inventory
    assert "That day doesn't work for me. I need a Friday afternoon appointment." not in inventory


def test_single_grounded_date_verifies_once_then_uses_slot_acceptance_machinery():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)
    offered = _respond(
        controller,
        policy,
        "I can offer December 18, 2026 at 2:15 PM. Would that work?",
    )
    assert offered.decision.text == VERIFICATION_TEXT
    accepted = _respond(controller, policy, "Yes, that is the latest date I can currently book.")
    assert accepted.decision.reason == "compatible_concrete_slot_offered"
    assert accepted.after.accepted_slot_text == "2:15 PM"


def test_multiple_dates_selects_latest_grounded_offer():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)
    result = _respond(
        controller,
        policy,
        "I can offer November 20, 2026 or December 18, 2026. Which date works?",
    )
    assert result.decision.text == "The later of those dates works for me. What times are available?"
    assert policy.latest_offered_date == "December 18, 2026"


def test_absolute_latest_fallback_is_asked_only_once():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)
    first = _respond(controller, policy, "I cannot search for the absolute latest date.")
    second = _respond(controller, policy, "I cannot search for the absolute latest date.")
    assert first.decision.text == FALLBACK_TEXT
    assert second.decision.text != FALLBACK_TEXT
    assert BOOKING_ACCEPTANCE != FALLBACK_TEXT


def test_suppressed_objective_retries_once_after_capability_menu():
    policy = FarthestDatePolicy()
    controller = SchedulingFlowController(decision_overlay=policy)

    prepared = controller.decide_burst(
        ["Your patient profile has been created. How may I help you today?"]
    )
    assert prepared.decision.reason == "call6:objective"
    policy.mark_decision_suppressed(prepared.decision)
    assert policy.objective_stated is False

    retried = controller.decide_burst(
        [
            "Thanks. Your profile is set up for this demo. What would you like to do next?",
            "I can help with appointments, medication refills, insurance updates, and more.",
        ]
    )
    assert retried.decision.reason == "call6:objective"
    assert retried.decision.text == OBJECTIVE_TEXT
    assert retried.decision.text != "Blue Cross."
    policy.mark_decision_suppressed(retried.decision)

    third = controller.decide_burst(["What would you like to do?"])
    assert third.decision.reason != "call6:objective"
