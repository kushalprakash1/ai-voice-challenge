from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.state import build_initial_state
from voiceprobe.scenarios.models import PatientFacts, PatientScenario
from voiceprobe.verbalizers.deterministic import (
    DeterministicNaturalVerbalizer,
)


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="deterministic-verbalizer",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            date_of_birth="April 12, 1998",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )


def verbalize(
    decision: CommunicationDecision,
) -> str:
    scenario = build_scenario()

    return DeterministicNaturalVerbalizer().verbalize(
        scenario=scenario,
        state=build_initial_state(scenario),
        decision=decision,
    )


def test_date_of_birth_answer_cannot_become_scheduling_claim() -> None:
    text = verbalize(
        CommunicationDecision(
            kind=CommunicationKind.ANSWER,
            facts_to_communicate=("date_of_birth",),
        )
    )

    assert text == "April 12, 1998."


def test_insurance_answer_cannot_invent_appointment() -> None:
    text = verbalize(
        CommunicationDecision(
            kind=CommunicationKind.ANSWER,
            facts_to_communicate=("insurance",),
        )
    )

    assert text == "Blue Cross."


def test_workflow_responses_are_stable() -> None:
    assert (
        verbalize(
            CommunicationDecision(
                kind=CommunicationKind.AGREE,
            )
        )
        == "Yes, please."
    )

    assert (
        verbalize(
            CommunicationDecision(
                kind=CommunicationKind.DECLINE_WORKFLOW,
            )
        )
        == "No, I need an appointment."
    )


def test_offer_acceptance_is_stable() -> None:
    assert (
        verbalize(
            CommunicationDecision(
                kind=CommunicationKind.ACCEPT_OFFER,
                offered_day="Friday",
                offered_time="2:30 PM",
            )
        )
        == "Yes, that works."
    )


def test_booking_verification_uses_only_authoritative_slot() -> None:
    assert (
        verbalize(
            CommunicationDecision(
                kind=CommunicationKind.VERIFY_BOOKING,
                offered_day="Friday",
                offered_time="2:30 PM",
            )
        )
        == "Just to confirm, am I booked for Friday at 2:30 PM?"
    )


def test_normal_end_is_stable() -> None:
    assert (
        verbalize(
            CommunicationDecision(
                kind=CommunicationKind.END_CONVERSATION,
            )
        )
        == "Okay, thank you. Bye."
    )
