from __future__ import annotations

from voiceprobe.agents.brain import CommunicationKind
from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    TurnMeaning,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.conversation.state import PatientState
from voiceprobe.scenarios.catalog import get_scenario
from voiceprobe.scenarios.models import PatientScenario


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


class DeterministicVerbalizer:
    def verbalize(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        decision: object,
    ) -> str:
        del scenario, state

        kind = decision.kind

        if kind is CommunicationKind.ACCEPT_PARTIAL_OFFER:
            return "That works. What is the missing slot detail?"

        if kind is CommunicationKind.ACCEPT_OFFER:
            return "That works for me."

        if kind is CommunicationKind.DECLINE_OFFER:
            return "That does not work for me."

        if kind is CommunicationKind.CLARIFY:
            return "Could you clarify?"

        return "Okay."


def test_friday_completes_previously_accepted_430_partial_offer() -> None:
    session = PatientSession(
        scenario=get_scenario("autonomous-phone-diagnostic"),
        interpreter=SequenceInterpreter(
            [
                TurnMeaning(
                    appointment_offer=AppointmentOffer(
                        day=None,
                        time="4:30 PM",
                    )
                ),
                # Reproduce the real failure: Qwen may consider a bare
                # "Friday." locally ambiguous. Trusted pending-slot state,
                # not model history, supplies its conversational meaning.
                TurnMeaning(
                    unclear=True,
                ),
            ]
        ),
        verbalizer=DeterministicVerbalizer(),
    )

    first = session.handle_agent_turn("Can you schedule an appointment for 4:30 PM?")

    assert first.decision.kind is CommunicationKind.ACCEPT_PARTIAL_OFFER
    assert first.progress.offered_day is None
    assert first.progress.offered_time == "4:30 PM"
    assert first.progress.offer_accepted is False

    second = session.handle_agent_turn("Friday.")

    assert second.meaning.appointment_offer == AppointmentOffer(
        day="Friday",
        time="4:30 PM",
    )
    assert second.decision.kind is CommunicationKind.ACCEPT_OFFER
    assert second.progress.offered_day == "Friday"
    assert second.progress.offered_time == "4:30 PM"
    assert second.progress.offer_accepted is True


def test_time_completes_previously_accepted_friday_partial_offer() -> None:
    session = PatientSession(
        scenario=get_scenario("autonomous-phone-diagnostic"),
        interpreter=SequenceInterpreter(
            [
                TurnMeaning(
                    appointment_offer=AppointmentOffer(
                        day="Friday",
                        time=None,
                    )
                ),
                TurnMeaning(
                    unclear=True,
                ),
            ]
        ),
        verbalizer=DeterministicVerbalizer(),
    )

    first = session.handle_agent_turn("I have an opening Friday.")

    assert first.decision.kind is CommunicationKind.ACCEPT_PARTIAL_OFFER

    second = session.handle_agent_turn("2:30 PM.")

    assert second.meaning.appointment_offer == AppointmentOffer(
        day="Friday",
        time="2:30 PM",
    )
    assert second.decision.kind is CommunicationKind.ACCEPT_OFFER
    assert second.progress.offered_day == "Friday"
    assert second.progress.offered_time == "2:30 PM"
    assert second.progress.offer_accepted is True


def test_declined_partial_offer_does_not_become_pending_context() -> None:
    session = PatientSession(
        scenario=get_scenario("autonomous-phone-diagnostic"),
        interpreter=SequenceInterpreter(
            [
                TurnMeaning(
                    appointment_offer=AppointmentOffer(
                        day=None,
                        time="11:30 AM",
                    )
                ),
                TurnMeaning(
                    unclear=True,
                ),
            ]
        ),
        verbalizer=DeterministicVerbalizer(),
    )

    first = session.handle_agent_turn("How about 11:30 AM?")

    assert first.decision.kind is CommunicationKind.DECLINE_OFFER

    second = session.handle_agent_turn("Friday.")

    # Because 11:30 AM was rejected, it must not silently survive and be
    # merged into a later bare weekday.
    assert second.meaning.appointment_offer is None
    assert second.decision.kind is CommunicationKind.CLARIFY
