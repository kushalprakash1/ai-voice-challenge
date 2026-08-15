from __future__ import annotations

from collections import deque

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    TurnMeaning,
)
from voiceprobe.conversation.normalization import (
    recover_asr_booking_confirmation,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.conversation.state import PatientState
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="asr-booking-recovery",
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


def test_your_book_corruption_recovers_with_matching_accepted_slot() -> None:
    meaning = TurnMeaning(
        appointment_offer=AppointmentOffer(
            day="Friday",
            time="2.30 p.m.",
        ),
        booking_confirmed=False,
    )

    recovered = recover_asr_booking_confirmation(
        meaning,
        agent_turn=("Great. Your book for Friday at 2.30 p.m."),
        accepted_offer_matches=True,
    )

    assert recovered.booking_confirmed


def test_your_book_corruption_cannot_confirm_without_slot_match() -> None:
    meaning = TurnMeaning(
        appointment_offer=AppointmentOffer(
            day="Tuesday",
            time="9 a.m.",
        ),
        booking_confirmed=False,
    )

    recovered = recover_asr_booking_confirmation(
        meaning,
        agent_turn=("Great. Your book for Tuesday at 9 a.m."),
        accepted_offer_matches=False,
    )

    assert not recovered.booking_confirmed


class SequenceInterpreter:
    def __init__(
        self,
        meanings: list[TurnMeaning],
    ) -> None:
        self._meanings = deque(meanings)

    def interpret(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> TurnMeaning:
        del scenario, state, agent_turn

        return self._meanings.popleft()


class DecisionVerbalizer:
    def verbalize(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        decision: CommunicationDecision,
    ) -> str:
        del scenario, state

        if decision.kind is CommunicationKind.ACCEPT_OFFER:
            return "Friday at 2:30 works for me."

        if decision.kind is CommunicationKind.ACKNOWLEDGE_COMPLETE:
            return "Okay, see you then."

        return decision.kind.value


def test_session_recovers_real_your_book_asr_corruption() -> None:
    scenario = build_scenario()

    interpreter = SequenceInterpreter(
        [
            TurnMeaning(
                appointment_offer=AppointmentOffer(
                    day="Friday",
                    time="2.30 p.m.",
                ),
            ),
            # Reproduce the real telephone failure:
            # Qwen extracts the matching slot but ASR corruption causes
            # booking_confirmed to remain false.
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
        scenario=scenario,
        interpreter=interpreter,
        verbalizer=DecisionVerbalizer(),
    )

    accepted = session.handle_agent_turn("How about Friday at 2.30 p.m.?")

    assert accepted.decision.kind is CommunicationKind.ACCEPT_OFFER
    assert accepted.progress.offer_accepted
    assert not accepted.progress.booking_confirmed
    assert not accepted.progress.objective_complete

    confirmed = session.handle_agent_turn("Great. Your book for Friday at 2.30 p.m.")

    assert confirmed.meaning.booking_confirmed
    assert confirmed.decision.kind is CommunicationKind.ACKNOWLEDGE_COMPLETE
    assert confirmed.progress.offer_accepted
    assert confirmed.progress.booking_confirmed
    assert confirmed.progress.objective_complete
