"""Deterministic behavioral-probe policy tests."""

from voiceprobe.agents.brain import CommunicationDecision, CommunicationKind
from voiceprobe.agents.probes import ProbeProgress, apply_probe_policy
from voiceprobe.conversation.objective import AppointmentProgress
from voiceprobe.scenarios.catalog import get_scenario
from voiceprobe.scenarios.models import ProbeKind


def test_repetition_scenario_requests_one_repeat_on_later_fact_question() -> None:
    scenario = get_scenario("repetition-clarification")
    progress = ProbeProgress()

    decision, progress = apply_probe_policy(
        scenario=scenario,
        appointment=AppointmentProgress(),
        probe_progress=progress,
        prior_agent_turn_count=1,
        base_decision=CommunicationDecision(
            kind=CommunicationKind.ANSWER,
            facts_to_communicate=("insurance",),
        ),
    )

    assert decision.kind is CommunicationKind.ASK_AGENT_TO_REPEAT
    assert decision.probe is ProbeKind.REQUEST_AGENT_REPEAT_ONCE
    assert progress.has_fired(ProbeKind.REQUEST_AGENT_REPEAT_ONCE)

    second_decision, second_progress = apply_probe_policy(
        scenario=scenario,
        appointment=AppointmentProgress(),
        probe_progress=progress,
        prior_agent_turn_count=2,
        base_decision=CommunicationDecision(
            kind=CommunicationKind.ANSWER,
            facts_to_communicate=("insurance",),
        ),
    )

    assert second_decision.kind is CommunicationKind.ANSWER
    assert second_progress == progress


def test_repeat_probe_never_overrides_patient_correction() -> None:
    scenario = get_scenario("repetition-clarification")

    decision, progress = apply_probe_policy(
        scenario=scenario,
        appointment=AppointmentProgress(),
        probe_progress=ProbeProgress(),
        prior_agent_turn_count=3,
        base_decision=CommunicationDecision(
            kind=CommunicationKind.CORRECT,
            facts_to_communicate=("name",),
        ),
    )

    assert decision.kind is CommunicationKind.CORRECT
    assert not progress.fired


def test_booking_probe_requests_confirmation_before_goodbye() -> None:
    scenario = get_scenario("booking-confirmation-robustness")
    appointment = AppointmentProgress(
        offered_day="Friday",
        offered_time="2:30 PM",
        offer_accepted=True,
        booking_confirmed=False,
    )

    decision, progress = apply_probe_policy(
        scenario=scenario,
        appointment=appointment,
        probe_progress=ProbeProgress(),
        prior_agent_turn_count=4,
        base_decision=CommunicationDecision(
            kind=CommunicationKind.END_CONVERSATION,
        ),
    )

    assert decision.kind is CommunicationKind.VERIFY_BOOKING
    assert decision.offered_day == "Friday"
    assert decision.offered_time == "2:30 PM"
    assert decision.probe is ProbeKind.VERIFY_BOOKING_BEFORE_END
    assert progress.has_fired(ProbeKind.VERIFY_BOOKING_BEFORE_END)


def test_booking_probe_does_not_fire_after_confirmation() -> None:
    scenario = get_scenario("booking-confirmation-robustness")
    appointment = AppointmentProgress(
        offered_day="Friday",
        offered_time="2:30 PM",
        offer_accepted=True,
        booking_confirmed=True,
    )
    base = CommunicationDecision(kind=CommunicationKind.END_CONVERSATION)

    decision, progress = apply_probe_policy(
        scenario=scenario,
        appointment=appointment,
        probe_progress=ProbeProgress(),
        prior_agent_turn_count=4,
        base_decision=base,
    )

    assert decision == base
    assert not progress.fired


def test_non_probe_scenario_keeps_original_decision() -> None:
    scenario = get_scenario("identity-insurance-check")
    base = CommunicationDecision(
        kind=CommunicationKind.ANSWER,
        facts_to_communicate=("date_of_birth",),
    )

    decision, progress = apply_probe_policy(
        scenario=scenario,
        appointment=AppointmentProgress(),
        probe_progress=ProbeProgress(),
        prior_agent_turn_count=4,
        base_decision=base,
    )

    assert decision == base
    assert not progress.fired


def test_catalog_has_only_two_intended_active_probes() -> None:
    assert get_scenario("repetition-clarification").probes == (
        ProbeKind.REQUEST_AGENT_REPEAT_ONCE,
    )
    assert get_scenario("booking-confirmation-robustness").probes == (
        ProbeKind.VERIFY_BOOKING_BEFORE_END,
    )


def test_booking_probe_skips_confirmation_already_given_this_turn() -> None:
    scenario = get_scenario("booking-confirmation-robustness")

    appointment = AppointmentProgress(
        offered_day="Friday",
        offered_time="2:30 PM",
        offer_accepted=True,
        booking_confirmed=False,
    )

    base = CommunicationDecision(
        kind=CommunicationKind.END_CONVERSATION,
    )

    decision, progress = apply_probe_policy(
        scenario=scenario,
        appointment=appointment,
        probe_progress=ProbeProgress(),
        prior_agent_turn_count=4,
        base_decision=base,
        booking_confirmed_this_turn=True,
    )

    assert decision == base
    assert not progress.fired
