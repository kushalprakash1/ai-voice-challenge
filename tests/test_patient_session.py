from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    FactAssertion,
    TurnMeaning,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.conversation.state import PatientState
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)


class StubInterpreter:
    def __init__(
        self,
        *meanings: TurnMeaning,
    ) -> None:
        self._meanings = list(meanings)
        self.seen_states: list[PatientState] = []

    def interpret(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
        progress=None,
    ) -> TurnMeaning:
        del scenario, agent_turn, progress

        self.seen_states.append(state)

        if not self._meanings:
            raise RuntimeError("No stub meaning remaining.")

        return self._meanings.pop(0)


class StubVerbalizer:
    def __init__(self) -> None:
        self.seen_states: list[PatientState] = []
        self.seen_decisions: list[CommunicationDecision] = []

    def verbalize(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        decision: CommunicationDecision,
    ) -> str:
        del scenario

        self.seen_states.append(state)
        self.seen_decisions.append(decision)

        if decision.kind is CommunicationKind.CORRECT:
            return "Actually, it's my right shoulder, and it's been five days."

        if decision.kind is CommunicationKind.ACCEPT_OFFER:
            return "Friday at 2:30 PM works for me."

        if decision.kind is CommunicationKind.ACKNOWLEDGE_COMPLETE:
            return "Perfect, thank you."

        if decision.kind is CommunicationKind.CLARIFY:
            return "Could you clarify that for me?"

        return "Here is my answer."


class FailingVerbalizer:
    def verbalize(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        decision: CommunicationDecision,
    ) -> str:
        del scenario, state, decision
        raise RuntimeError("verbalizer failure")


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="shoulder-friday",
        objective=("Schedule an appointment for Friday afternoon."),
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )


def test_interpreter_receives_pre_turn_state() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            requested_facts=("insurance",),
        )
    )
    verbalizer = StubVerbalizer()

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=verbalizer,
    )

    session.handle_agent_turn("What insurance do you have?")

    assert interpreter.seen_states[0].messages == ()

    # The verbalizer sees the receptionist message because it needs
    # conversational context for realizing the patient response.
    assert len(verbalizer.seen_states[0].messages) == 1

    assert len(session.state.messages) == 2
    assert "insurance" in session.state.answered_facts


def test_grounded_correction_is_recorded() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            requested_facts=("complaint",),
            stated_facts=(
                FactAssertion(
                    fact="complaint",
                    value="left knee",
                ),
            ),
        )
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    result = session.handle_agent_turn("It's your left knee, correct?")

    assert result.decision.kind is CommunicationKind.CORRECT
    assert len(session.state.corrections) == 1
    assert session.state.corrections[0].claim == "complaint='left knee'"


def test_matching_offer_is_recorded_and_accepted() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            ),
        )
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    session.handle_agent_turn("I have Friday at 2:30 PM available.")

    assert session.progress.offered_day == "Friday"
    assert session.progress.offered_time == "2:30 PM"
    assert session.progress.offer_accepted
    assert not session.progress.objective_complete


def test_preference_answer_updates_progress() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            requested_facts=(
                "preferred_day",
                "preferred_time",
            ),
        )
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    session.handle_agent_turn("What day and time work best for you?")

    assert session.progress.preferred_day_shared
    assert session.progress.preferred_time_shared


def test_same_slot_confirmation_completes_objective() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            ),
        ),
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            ),
            booking_confirmed=True,
        ),
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    session.handle_agent_turn("Friday at 2:30 PM is available.")

    result = session.handle_agent_turn("Great, you're booked for Friday at 2:30 PM.")

    assert result.decision.kind is CommunicationKind.ACKNOWLEDGE_COMPLETE
    assert session.progress.offer_accepted
    assert session.progress.booking_confirmed
    assert session.progress.objective_complete
    assert session.state.objective_complete


def test_booking_without_acceptance_does_not_complete() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            booking_confirmed=True,
        )
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    result = session.handle_agent_turn("Okay, you're booked.")

    assert result.decision.kind is CommunicationKind.CLARIFY
    assert not session.progress.booking_confirmed
    assert not session.progress.objective_complete
    assert not session.state.objective_complete


def test_failed_verbalizer_does_not_partially_commit_turn() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            requested_facts=("insurance",),
        )
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=FailingVerbalizer(),
    )

    try:
        session.handle_agent_turn("What insurance do you have?")
    except RuntimeError as error:
        assert str(error) == "verbalizer failure"
    else:
        raise AssertionError("Expected verbalizer failure.")

    assert session.state.messages == ()
    assert not session.state.answered_facts
    assert not session.progress.preferred_day_shared
    assert not session.progress.preferred_time_shared
    assert not session.progress.has_offer


def test_incompatible_offer_is_declined_not_corrected() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            stated_facts=(
                FactAssertion(
                    fact="preferred_day",
                    value="Tuesday",
                ),
                FactAssertion(
                    fact="preferred_time",
                    value="9:00 AM",
                ),
            ),
            appointment_offer=AppointmentOffer(
                day="Tuesday",
                time="9:00 AM",
            ),
        )
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    result = session.handle_agent_turn(
        "I have Tuesday at 9:00 AM available. Would that work?"
    )

    assert result.decision.kind is CommunicationKind.DECLINE_OFFER

    assert result.grounded.conflicts == ()

    assert result.decision.facts_to_communicate == (
        "preferred_day",
        "preferred_time",
    )

    assert session.progress.offered_day == "Tuesday"
    assert session.progress.offered_time == "9:00 AM"
    assert not session.progress.offer_accepted


def test_non_actionable_turn_waits_without_verbalizer() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            topic="none",
        )
    )
    verbalizer = StubVerbalizer()

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=verbalizer,
    )

    result = session.handle_agent_turn("One moment.")

    assert result.decision.kind is CommunicationKind.WAIT
    assert result.patient_text == ""
    assert result.meaning.topic is None
    assert result.timings.verbalizer_seconds == 0.0

    # The receptionist turn remains in conversation history, but no
    # synthetic patient turn is created.
    assert len(session.state.messages) == 1
    assert verbalizer.seen_states == []
    assert verbalizer.seen_decisions == []
    assert not session.state.answered_facts


def test_substantive_statement_does_not_become_wait() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            topic="agent's availability",
        )
    )
    verbalizer = StubVerbalizer()

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=verbalizer,
    )

    result = session.handle_agent_turn("I don't have Friday afternoon available.")

    assert result.decision.kind is CommunicationKind.CLARIFY
    assert result.patient_text != ""
    assert len(verbalizer.seen_decisions) == 1


def test_matching_booking_and_goodbye_completes_before_ending() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            ),
        ),
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            ),
            booking_confirmed=True,
            conversation_end_requested=True,
        ),
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    session.handle_agent_turn("Friday at 2:30 PM is available.")

    result = session.handle_agent_turn(
        "You're booked for Friday at 2:30 PM. Have a good day, goodbye."
    )

    assert result.decision.kind is CommunicationKind.END_CONVERSATION
    assert session.progress.offered_day == "Friday"
    assert session.progress.offered_time == "2:30 PM"
    assert session.progress.offer_accepted
    assert session.progress.booking_confirmed
    assert session.progress.objective_complete
    assert session.state.objective_complete


def test_wrong_booking_and_goodbye_is_rejected_before_ending() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            ),
        ),
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Tuesday",
                time="9 AM",
            ),
            booking_confirmed=True,
            conversation_end_requested=True,
        ),
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    session.handle_agent_turn("Friday at 2:30 PM is available.")

    result = session.handle_agent_turn("You're all set for Tuesday at 9 AM. Goodbye.")

    assert result.decision.kind is CommunicationKind.DECLINE_OFFER

    # A contradictory confirmation must never replace the slot the patient
    # already accepted.
    assert session.progress.offered_day == "Friday"
    assert session.progress.offered_time == "2:30 PM"
    assert session.progress.offer_accepted

    assert not session.progress.booking_confirmed
    assert not session.progress.objective_complete
    assert not session.state.objective_complete


def test_plain_goodbye_before_booking_does_not_end_conversation() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            conversation_end_requested=True,
        )
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    result = session.handle_agent_turn("Okay, goodbye.")

    assert result.decision.kind is CommunicationKind.DECLINE_WORKFLOW
    assert not session.progress.booking_confirmed
    assert not session.progress.objective_complete
    assert not session.state.objective_complete


def test_goodbye_after_accepted_offer_requests_booking_confirmation() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            ),
        ),
        TurnMeaning(
            conversation_end_requested=True,
        ),
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    first = session.handle_agent_turn(
        "Friday at 2:30 PM is available."
    )

    assert first.decision.kind is CommunicationKind.ACCEPT_OFFER
    assert session.progress.offer_accepted

    result = session.handle_agent_turn("Okay, goodbye.")

    assert result.decision.kind is CommunicationKind.VERIFY_BOOKING
    assert result.decision.offered_day == "Friday"
    assert result.decision.offered_time == "2:30 PM"

    assert session.progress.offer_accepted
    assert not session.progress.booking_confirmed
    assert not session.progress.objective_complete
    assert not session.state.objective_complete


def test_offer_and_goodbye_same_turn_processes_offer_before_ending() -> None:
    interpreter = StubInterpreter(
        TurnMeaning(
            appointment_offer=AppointmentOffer(
                day="Friday",
                time="2:30 PM",
            ),
            conversation_end_requested=True,
        )
    )

    session = PatientSession(
        scenario=build_scenario(),
        interpreter=interpreter,
        verbalizer=StubVerbalizer(),
    )

    result = session.handle_agent_turn(
        "Friday at 2:30 PM is available. Goodbye."
    )

    assert result.decision.kind is CommunicationKind.ACCEPT_OFFER
    assert session.progress.offered_day == "Friday"
    assert session.progress.offered_time == "2:30 PM"
    assert session.progress.offer_accepted

    assert not session.progress.booking_confirmed
    assert not session.progress.objective_complete
    assert not session.state.objective_complete
