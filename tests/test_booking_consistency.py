from __future__ import annotations

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
    PatientBrain,
)
from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    TurnMeaning,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.conversation.state import PatientState
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)


class SequenceInterpreter:
    def __init__(
        self,
        meanings: list[TurnMeaning],
    ) -> None:
        self._meanings = iter(meanings)

    def interpret(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> TurnMeaning:
        del scenario, state, agent_turn
        return next(self._meanings)


class StubVerbalizer:
    def verbalize(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        decision: CommunicationDecision,
    ) -> str:
        del scenario, state

        if decision.kind is CommunicationKind.ACKNOWLEDGE_COMPLETE:
            return "Perfect, thank you."

        if decision.kind is CommunicationKind.DECLINE_OFFER:
            return "That time doesn't match what we agreed on."

        return "That works for me."


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="booking-consistency",
        objective="Schedule Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )


def test_wrong_confirmation_preserves_accepted_slot_then_correction_completes() -> None:
    interpreter = SequenceInterpreter(
        [
            TurnMeaning(
                appointment_offer=AppointmentOffer(
                    day="Friday",
                    time="2.30 p.m.",
                )
            ),
            TurnMeaning(
                appointment_offer=AppointmentOffer(
                    day="Friday",
                    time="10 a.m.",
                ),
                booking_confirmed=True,
            ),
            # Simulate the exact Qwen miss we observed: it extracts the
            # slot but forgets booking_confirmed. Deterministic
            # normalization must recover it from "you're booked".
            TurnMeaning(
                appointment_offer=AppointmentOffer(
                    day="Friday",
                    time="2.30 p.m.",
                ),
                booking_confirmed=False,
            ),
        ]
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
        brain=PatientBrain(),
    )

    accepted = session.handle_agent_turn("How about Friday at 2.30 p.m.?")

    assert accepted.decision.kind is CommunicationKind.ACCEPT_OFFER
    assert accepted.progress.offered_day == "Friday"
    assert accepted.progress.offered_time == "2.30 p.m."
    assert accepted.progress.offer_accepted
    assert not accepted.progress.booking_confirmed
    assert not accepted.progress.objective_complete

    wrong_confirmation = session.handle_agent_turn(
        "Great, you're booked for Friday at 10 a.m."
    )

    assert wrong_confirmation.decision.kind is CommunicationKind.DECLINE_OFFER

    # The wrong booking must NOT replace the already accepted slot.
    assert wrong_confirmation.progress.offered_day == "Friday"
    assert wrong_confirmation.progress.offered_time == "2.30 p.m."
    assert wrong_confirmation.progress.offer_accepted
    assert not wrong_confirmation.progress.booking_confirmed
    assert not wrong_confirmation.progress.objective_complete

    corrected_confirmation = session.handle_agent_turn(
        "Sorry, you're booked for Friday at 2.30 p.m."
    )

    assert corrected_confirmation.meaning.booking_confirmed
    assert (
        corrected_confirmation.decision.kind is CommunicationKind.ACKNOWLEDGE_COMPLETE
    )
    assert corrected_confirmation.progress.offered_day == "Friday"
    assert corrected_confirmation.progress.offered_time == "2.30 p.m."
    assert corrected_confirmation.progress.offer_accepted
    assert corrected_confirmation.progress.booking_confirmed
    assert corrected_confirmation.progress.objective_complete
