from __future__ import annotations

import pytest

from voiceprobe.agents.brain import CommunicationKind
from voiceprobe.conversation.meaning import (
    FactAssertion,
    QuestionKind,
    ResponseExpectation,
    TurnMeaning,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.interpreters.semantic_gate import deterministic_turn_meaning
from voiceprobe.scenarios.catalog import get_scenario
from voiceprobe.target_memory import (
    matching_target_memory,
    should_tolerate_target_conflict,
)
from voiceprobe.verbalizers.deterministic import (
    DeterministicNaturalVerbalizer,
)


class StaticInterpreter:
    def __init__(self, meaning: TurnMeaning) -> None:
        self._meaning = meaning

    def interpret(
        self,
        *,
        scenario,
        state,
        agent_turn: str,
    ) -> TurnMeaning:
        return self._meaning


def build_session(
    monkeypatch: pytest.MonkeyPatch,
    meaning: TurnMeaning,
) -> PatientSession:
    monkeypatch.setenv("VOICEPROBE_EXPLORATION_MODE", "1")

    return PatientSession(
        scenario=get_scenario("autonomous-phone-diagnostic"),
        interpreter=StaticInterpreter(meaning),
        verbalizer=DeterministicNaturalVerbalizer(),
    )


def test_first_name_is_not_misclassified_as_full_name() -> None:
    scenario = get_scenario("autonomous-phone-diagnostic")

    meaning = deterministic_turn_meaning(
        scenario=scenario,
        agent_turn="What is your first name?",
    )

    assert meaning is not None
    assert meaning.requested_facts == ("first_name",)


def test_profile_context_does_not_turn_first_name_into_permission() -> None:
    scenario = get_scenario("autonomous-phone-diagnostic")

    meaning = deterministic_turn_meaning(
        scenario=scenario,
        agent_turn=(
            "Sure, patient profile. I just need your first name. "
            "What should I enter for you?"
        ),
    )

    assert meaning is not None
    assert meaning.requested_facts == ("first_name",)


def test_appointment_type_can_be_answered_without_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = build_session(
        monkeypatch,
        TurnMeaning(
            response_expectation=ResponseExpectation.CHOICE,
            question_kind=QuestionKind.PATIENT_ATTRIBUTE,
            requested_facts=("appointment_type",),
            topic="appointment type",
        ),
    )

    result = session.handle_agent_turn(
        "What type of appointment do you need?"
    )

    assert result.decision.kind is CommunicationKind.ANSWER
    assert result.patient_text == "I need a new patient consultation."


def test_new_patient_and_visit_history_answer_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = build_session(
        monkeypatch,
        TurnMeaning(
            response_expectation=ResponseExpectation.FACT,
            question_kind=QuestionKind.PATIENT_ATTRIBUTE,
            requested_facts=(
                "patient_status",
                "visited_before",
            ),
            topic="patient status and prior visits",
        ),
    )

    result = session.handle_agent_turn(
        "Are you a patient, or have you visited us before?"
    )

    assert result.decision.kind is CommunicationKind.ANSWER
    assert "I'm a new patient." in result.patient_text
    assert "I haven't visited before." in result.patient_text


def test_demo_dob_memory_is_scoped_and_retrievable() -> None:
    text = (
        "Your date of birth is July 4, 2000 "
        "for demo purposes."
    )

    matches = matching_target_memory(text)

    assert tuple(entry.memory_id for entry in matches) == ("TB-002",)
    assert should_tolerate_target_conflict(text) is True


def test_known_demo_dob_does_not_overwrite_truth_or_force_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = build_session(
        monkeypatch,
        TurnMeaning(
            stated_facts=(
                FactAssertion(
                    fact="date_of_birth",
                    value="July 4, 2000",
                ),
            ),
        ),
    )

    result = session.handle_agent_turn(
        "Your date of birth is July 4, 2000 for demo purposes."
    )

    assert result.decision.kind is CommunicationKind.WAIT
    assert result.patient_text == ""
    assert result.state.corrections == ()

    scenario = get_scenario("autonomous-phone-diagnostic")
    assert scenario.facts.date_of_birth == "April 12, 1998"
