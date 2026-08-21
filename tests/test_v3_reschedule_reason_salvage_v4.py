from __future__ import annotations

from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy
from voiceprobe.v3.models import DecisionKind, PatientFacts


EXISTING_APPOINTMENT_PROMPT = (
    "You already have a new patient consultation booked for Tuesday, "
    "August twenty fifth at two fifteen PM. Would you like to keep this "
    "appointment, reschedule it to Friday afternoon, or cancel it?"
)


def test_reschedule_choice_arms_reason_expectation() -> None:
    policy = RoutineSchedulingPolicy()

    decision = policy.decide(EXISTING_APPOINTMENT_PROMPT)

    assert decision.kind == DecisionKind.STATE_OBJECTIVE
    assert decision.reason == "existing_appointment_reschedule"
    assert policy._awaiting_reschedule_reason is True


def test_exact_failed_call_reschedule_reason_is_deterministic() -> None:
    policy = RoutineSchedulingPolicy()
    policy.decide(EXISTING_APPOINTMENT_PROMPT)

    decision = policy.decide(
        "Can you tell me the reason you need to reschedule?"
    )

    assert decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert decision.text == "I have right shoulder pain."
    assert decision.reason == "reschedule_reason_requested"
    assert decision.confidence == 1.0
    assert policy._awaiting_reschedule_reason is False


def test_failed_call_rephrase_is_deterministic() -> None:
    policy = RoutineSchedulingPolicy()
    policy.decide(EXISTING_APPOINTMENT_PROMPT)

    decision = policy.decide(
        "Of course. Can you share why you need to change your appointment "
        "from Tuesday, August twenty fifth to a different day?"
    )

    assert decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert decision.text == "I have right shoulder pain."
    assert decision.reason == "reschedule_reason_requested"


def test_reschedule_reason_uses_authoritative_patient_facts() -> None:
    facts = PatientFacts(complaint="left knee pain")
    policy = RoutineSchedulingPolicy(facts)
    policy.decide(EXISTING_APPOINTMENT_PROMPT)

    decision = policy.decide(
        "Why do you need to reschedule your appointment?"
    )

    assert decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert decision.text == "I have left knee pain."


def test_normal_recognized_question_still_wins_after_reschedule() -> None:
    policy = RoutineSchedulingPolicy()
    policy.decide(EXISTING_APPOINTMENT_PROMPT)

    decision = policy.decide("What insurance do you have?")

    assert decision.kind == DecisionKind.ANSWER_FACT
    assert "Blue Cross" in decision.text
    # The expectation is intentionally retained until the remote scheduler
    # actually asks the reschedule-reason question.
    assert policy._awaiting_reschedule_reason is True


def test_unrelated_why_question_is_not_hijacked() -> None:
    policy = RoutineSchedulingPolicy()
    policy.decide(EXISTING_APPOINTMENT_PROMPT)

    decision = policy.decide("Why is your office closed today?")

    assert decision.kind == DecisionKind.FALLBACK
    assert decision.reason == "novel_or_ambiguous_turn"
    assert policy._awaiting_reschedule_reason is True


def test_general_visit_reason_clears_reschedule_expectation_too() -> None:
    policy = RoutineSchedulingPolicy()
    policy.decide(EXISTING_APPOINTMENT_PROMPT)

    decision = policy.decide("What is the reason for your visit?")

    assert decision.kind == DecisionKind.ANSWER_COMPLAINT
    assert decision.text == "I have right shoulder pain."
    assert policy._awaiting_reschedule_reason is False
