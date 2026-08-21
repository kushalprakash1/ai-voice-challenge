from __future__ import annotations

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.autonomous_phone import build_scenario
from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    TurnMeaning,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.conversation.state import PatientState
from voiceprobe.scenarios.models import PatientScenario
from voiceprobe.verbalizers.ollama import OllamaNaturalVerbalizer


class FixedInterpreter:
    def __init__(
        self,
        meaning: TurnMeaning,
    ) -> None:
        self._meaning = meaning

    def interpret(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> TurnMeaning:
        del scenario, state, agent_turn
        return self._meaning


class RegressionVerbalizer:
    def verbalize(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        decision: CommunicationDecision,
    ) -> str:
        del scenario, state

        if decision.kind is CommunicationKind.END_CONVERSATION:
            return "Okay, thank you. Bye."

        if decision.kind is CommunicationKind.ACCEPT_PARTIAL_OFFER:
            if decision.offered_day is None:
                return "That time works. What day would that be?"

            return "That day works. What time would that be?"

        if decision.kind is CommunicationKind.ACCEPT_OFFER:
            return "That works for me."

        return "Okay."


def make_session(
    meaning: TurnMeaning,
) -> PatientSession:
    return PatientSession(
        scenario=build_scenario(),
        interpreter=FixedInterpreter(meaning),
        verbalizer=RegressionVerbalizer(),
    )


def test_goodbye_before_objective_completion_does_not_end_conversation() -> None:
    session = make_session(
        TurnMeaning(
            conversation_end_requested=True,
        )
    )

    result = session.handle_agent_turn("Okay, bye.")

    assert result.meaning.conversation_end_requested is True
    assert result.decision.kind is CommunicationKind.DECLINE_WORKFLOW
    assert not result.progress.booking_confirmed
    assert not result.progress.objective_complete
    assert not session.state.objective_complete
    # A premature goodbye must not make the simulated patient
    # reciprocate with the old terminal goodbye response.
    assert result.patient_text != "Okay, thank you. Bye."
    assert result.progress.objective_complete is False


def test_matching_time_only_offer_is_partial_not_complete_acceptance() -> None:
    session = make_session(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day=None,
                time="4:30 PM",
            )
        )
    )

    result = session.handle_agent_turn("Can you schedule an appointment for 4:30 PM?")

    assert result.decision.kind is CommunicationKind.ACCEPT_PARTIAL_OFFER
    assert result.progress.offered_time == "4:30 PM"
    assert result.progress.offered_day is None
    assert result.progress.offer_accepted is False
    assert result.progress.objective_complete is False


def test_matching_day_only_offer_is_partial_not_complete_acceptance() -> None:
    session = make_session(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time=None,
            )
        )
    )

    result = session.handle_agent_turn("I have an opening Friday.")

    assert result.decision.kind is CommunicationKind.ACCEPT_PARTIAL_OFFER
    assert result.progress.offered_day == "Friday"
    assert result.progress.offered_time is None
    assert result.progress.offer_accepted is False


def test_complete_matching_offer_still_accepts_normally() -> None:
    session = make_session(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            )
        )
    )

    result = session.handle_agent_turn("How about Friday at 2:30 PM?")

    assert result.decision.kind is CommunicationKind.ACCEPT_OFFER
    assert result.progress.offer_accepted is True


def test_partial_offer_verbalizer_goal_requires_missing_detail() -> None:
    goal = OllamaNaturalVerbalizer._speech_goal(
        CommunicationDecision(
            kind=CommunicationKind.ACCEPT_PARTIAL_OFFER,
            offered_time="4:30 PM",
        )
    )

    assert "missing day or time" in goal
    assert "complete appointment" in goal
    assert "at most 12 spoken words" in goal


def test_end_conversation_verbalizer_goal_forbids_reopening() -> None:
    goal = OllamaNaturalVerbalizer._speech_goal(
        CommunicationDecision(
            kind=CommunicationKind.END_CONVERSATION,
        )
    )

    assert "end the conversation" in goal
    assert "Do not ask another question" in goal
    assert "do not reopen scheduling" in goal
