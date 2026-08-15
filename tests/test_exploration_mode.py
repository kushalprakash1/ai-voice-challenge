from __future__ import annotations

import pytest

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.meaning import (
    QuestionKind,
    ResponseExpectation,
    TurnMeaning,
    WorkflowDirection,
    WorkflowRelation,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.scenarios.catalog import get_scenario


class StaticInterpreter:
    def __init__(
        self,
        meaning: TurnMeaning,
    ) -> None:
        self.meaning = meaning

    def interpret(
        self,
        *,
        scenario,
        state,
        agent_turn: str,
    ) -> TurnMeaning:
        return self.meaning


class DecisionEchoVerbalizer:
    def verbalize(
        self,
        *,
        scenario,
        state,
        decision: CommunicationDecision,
    ) -> str:
        if decision.state_objective:
            return "I need to schedule an appointment for Friday afternoon."

        return decision.kind.value


def build_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    meaning: TurnMeaning,
    exploration: bool,
) -> PatientSession:
    if exploration:
        monkeypatch.setenv(
            "VOICEPROBE_EXPLORATION_MODE",
            "1",
        )
    else:
        monkeypatch.delenv(
            "VOICEPROBE_EXPLORATION_MODE",
            raising=False,
        )

    return PatientSession(
        scenario=get_scenario("autonomous-phone-diagnostic"),
        interpreter=StaticInterpreter(meaning),
        verbalizer=DecisionEchoVerbalizer(),
    )


def test_normal_mode_still_declines_optional_side_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meaning = TurnMeaning(
        response_expectation=ResponseExpectation.YES_NO,
        workflow_relation=WorkflowRelation.NONE,
        question_kind=QuestionKind.WORKFLOW_PERMISSION,
        workflow_direction=WorkflowDirection.CONTINUE,
        topic="demo patient profile",
    )

    session = build_session(
        monkeypatch,
        meaning=meaning,
        exploration=False,
    )

    result = session.handle_agent_turn(
        "Would you like to create a demo patient profile?"
    )

    assert session.exploration_mode is False
    assert result.decision.kind is CommunicationKind.DECLINE_WORKFLOW


def test_exploration_mode_agrees_to_optional_side_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meaning = TurnMeaning(
        response_expectation=ResponseExpectation.YES_NO,
        workflow_relation=WorkflowRelation.NONE,
        question_kind=QuestionKind.WORKFLOW_PERMISSION,
        workflow_direction=WorkflowDirection.CONTINUE,
        topic="demo patient profile",
    )

    session = build_session(
        monkeypatch,
        meaning=meaning,
        exploration=True,
    )

    result = session.handle_agent_turn(
        "Would you like to create a demo patient profile?"
    )

    assert session.exploration_mode is True
    assert result.decision.kind is CommunicationKind.AGREE


def test_exploration_discloses_known_name_inside_profile_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meaning = TurnMeaning(
        response_expectation=ResponseExpectation.FACT,
        workflow_relation=WorkflowRelation.NONE,
        question_kind=QuestionKind.PATIENT_ATTRIBUTE,
        workflow_direction=WorkflowDirection.CONTINUE,
        topic="demo patient profile",
        requested_facts=("name",),
    )

    session = build_session(
        monkeypatch,
        meaning=meaning,
        exploration=True,
    )

    result = session.handle_agent_turn(
        "I just need your first and last name to get started."
    )

    assert result.decision.kind is CommunicationKind.ANSWER
    assert result.decision.facts_to_communicate == ("name",)


def test_exploration_does_not_agree_to_stop_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meaning = TurnMeaning(
        response_expectation=ResponseExpectation.YES_NO,
        workflow_relation=WorkflowRelation.OPPOSES_OBJECTIVE,
        question_kind=QuestionKind.WORKFLOW_PERMISSION,
        workflow_direction=WorkflowDirection.STOP,
        topic="cancel appointment workflow",
    )

    session = build_session(
        monkeypatch,
        meaning=meaning,
        exploration=True,
    )

    result = session.handle_agent_turn(
        "Would you like to cancel and stop?"
    )

    assert result.decision.kind is CommunicationKind.DECLINE_WORKFLOW


def test_exploration_restates_objective_on_premature_goodbye(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meaning = TurnMeaning(
        conversation_end_requested=True,
    )

    session = build_session(
        monkeypatch,
        meaning=meaning,
        exploration=True,
    )

    result = session.handle_agent_turn(
        "Have a great day."
    )

    assert result.decision.kind is CommunicationKind.ANSWER
    assert result.decision.state_objective is True


def test_invalid_exploration_environment_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VOICEPROBE_EXPLORATION_MODE",
        "maybe",
    )

    with pytest.raises(
        ValueError,
        match="must be exactly '0' or '1'",
    ):
        PatientSession(
            scenario=get_scenario("autonomous-phone-diagnostic"),
            interpreter=StaticInterpreter(TurnMeaning()),
            verbalizer=DecisionEchoVerbalizer(),
        )
