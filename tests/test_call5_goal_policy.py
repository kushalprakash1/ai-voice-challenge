import httpx

from voiceprobe.agents.brain import (
    CommunicationKind,
    PatientBrain,
)
from voiceprobe.conversation.goal_policy import WorkflowFocus
from voiceprobe.conversation.session import PatientSession
from voiceprobe.interpreters.ollama import OllamaConversationInterpreter
from voiceprobe.scenarios.catalog import get_scenario
from voiceprobe.verbalizers.deterministic import (
    DeterministicNaturalVerbalizer,
)


OBJECTIVE_TEXT = (
    "I need to schedule an appointment for Friday afternoon."
)


def build_session() -> tuple[
    PatientSession,
    OllamaConversationInterpreter,
    httpx.Client,
]:
    def forbidden_ollama(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "Call 5 replay unexpectedly reached Ollama: "
            f"{request.method} {request.url}"
        )

    client = httpx.Client(
        transport=httpx.MockTransport(forbidden_ollama),
    )

    interpreter = OllamaConversationInterpreter(
        model="qwen3:1.7b",
        client=client,
    )

    session = PatientSession(
        scenario=get_scenario("autonomous-phone-diagnostic"),
        interpreter=interpreter,
        verbalizer=DeterministicNaturalVerbalizer(),
        brain=PatientBrain(),
    )

    return session, interpreter, client


def test_literal_call5_failure_sequence_is_goal_directed() -> None:
    session, interpreter, client = build_session()

    turns = (
        (
            (
                "This call may be recorded for quality and training purposes. "
                "Para Español, Oprima y Dos."
            ),
            CommunicationKind.WAIT,
            "",
            WorkflowFocus.NONE,
        ),
        (
            "Thank you for calling.",
            CommunicationKind.WAIT,
            "",
            WorkflowFocus.NONE,
        ),
        (
            (
                "I can help you create a demo patient profile. "
                "I just need your first and last name."
            ),
            CommunicationKind.DECLINE_WORKFLOW,
            "No, I need an appointment.",
            WorkflowFocus.SIDE_WORKFLOW,
        ),
        (
            "Would you like to do that?",
            CommunicationKind.DECLINE_WORKFLOW,
            "No, I need an appointment.",
            WorkflowFocus.SIDE_WORKFLOW,
        ),
        (
            (
                "created and your date of birth is July 4, 2000 "
                "for demo purposes. How may I help you today?"
            ),
            CommunicationKind.ANSWER,
            OBJECTIVE_TEXT,
            WorkflowFocus.SCHEDULING,
        ),
        (
            (
                "You are now set up as a demo patient named Alex Morgan. "
                "with a birth date of July 4, 2000. "
                "This lets me show you how the system works. "
                "What would you like to try or ask about next?"
            ),
            CommunicationKind.ANSWER,
            OBJECTIVE_TEXT,
            WorkflowFocus.SCHEDULING,
        ),
    )

    patient_outputs: list[str] = []

    try:
        for (
            agent_turn,
            expected_kind,
            expected_text,
            expected_focus,
        ) in turns:
            result = session.handle_agent_turn(agent_turn)

            patient_outputs.append(result.patient_text)

            assert result.decision.kind is expected_kind
            assert result.patient_text == expected_text
            assert session.goal_context.focus is expected_focus

        # The demo-profile workflow must never receive the patient name.
        assert "Alex Morgan." not in patient_outputs

        # The two open-ended prompts must be mission restatements, not CLARIFY.
        assert patient_outputs[-2:] == [
            OBJECTIVE_TEXT,
            OBJECTIVE_TEXT,
        ]

        assert not session.progress.objective_complete

    finally:
        interpreter.close()
        client.close()


def test_side_workflow_context_blocks_followup_fact_leak() -> None:
    session, interpreter, client = build_session()

    try:
        first = session.handle_agent_turn(
            "I can create a demo patient profile for you."
        )

        assert first.decision.kind is CommunicationKind.DECLINE_WORKFLOW
        assert session.goal_context.focus is WorkflowFocus.SIDE_WORKFLOW

        followup = session.handle_agent_turn(
            "I just need your first and last name."
        )

        assert (
            followup.decision.kind
            is CommunicationKind.DECLINE_WORKFLOW
        )
        assert followup.patient_text == "No, I need an appointment."
        assert "name" not in session.state.answered_facts

    finally:
        interpreter.close()
        client.close()


def test_explicit_scheduling_intake_can_exit_side_workflow() -> None:
    session, interpreter, client = build_session()

    try:
        session.handle_agent_turn(
            "I can create a demo patient profile for you."
        )

        result = session.handle_agent_turn(
            "Before I can schedule you, I need your date of birth."
        )

        assert result.decision.kind is CommunicationKind.ANSWER
        assert result.patient_text == "April 12, 1998."
        assert session.goal_context.focus is WorkflowFocus.SCHEDULING

    finally:
        interpreter.close()
        client.close()
