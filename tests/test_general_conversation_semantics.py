import json

import httpx

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
    PatientBrain,
)
from voiceprobe.conversation.grounding import ground_turn_meaning
from voiceprobe.conversation.meaning import (
    QuestionKind,
    ResponseExpectation,
    TurnMeaning,
    WorkflowDirection,
    WorkflowRelation,
)
from voiceprobe.conversation.objective import AppointmentProgress
from voiceprobe.conversation.state import build_initial_state
from voiceprobe.interpreters.ollama import OllamaConversationInterpreter
from voiceprobe.scenarios.models import PatientFacts, PatientScenario
from voiceprobe.verbalizers.ollama import OllamaNaturalVerbalizer


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="shoulder-friday",
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


def decide(meaning: TurnMeaning) -> CommunicationDecision:
    scenario = build_scenario()

    return PatientBrain().decide(
        scenario=scenario,
        grounded=ground_turn_meaning(
            scenario=scenario,
            meaning=meaning,
        ),
        progress=AppointmentProgress(),
    )


def test_goal_aligned_yes_no_workflow_is_agreed_to() -> None:
    decision = decide(
        TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=WorkflowRelation.ADVANCES_OBJECTIVE,
            question_kind=QuestionKind.WORKFLOW_PERMISSION,
            workflow_direction=WorkflowDirection.CONTINUE,
            topic="creating a patient profile",
        )
    )

    assert decision.kind is CommunicationKind.AGREE
    assert decision.facts_to_communicate == ()


def test_goal_opposing_continue_workflow_is_declined() -> None:
    decision = decide(
        TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=WorkflowRelation.OPPOSES_OBJECTIVE,
            question_kind=QuestionKind.WORKFLOW_PERMISSION,
            workflow_direction=WorkflowDirection.CONTINUE,
            topic="performing an objective-opposing side workflow",
        )
    )

    assert decision.kind is CommunicationKind.DECLINE_WORKFLOW


def test_optional_continue_workflow_is_declined() -> None:
    decision = decide(
        TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=WorkflowRelation.NONE,
            question_kind=QuestionKind.WORKFLOW_PERMISSION,
            workflow_direction=WorkflowDirection.CONTINUE,
            topic="performing an optional side workflow",
        )
    )

    assert decision.kind is CommunicationKind.DECLINE_WORKFLOW


def test_uncertain_continue_workflow_requests_clarification() -> None:
    decision = decide(
        TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=WorkflowRelation.UNCERTAIN,
            question_kind=QuestionKind.WORKFLOW_PERMISSION,
            workflow_direction=WorkflowDirection.CONTINUE,
            topic="performing a workflow with unclear objective relevance",
        )
    )

    assert decision.kind is CommunicationKind.CLARIFY


def test_goal_opposing_yes_no_workflow_is_declined() -> None:
    decision = decide(
        TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=WorkflowRelation.OPPOSES_OBJECTIVE,
            question_kind=QuestionKind.WORKFLOW_PERMISSION,
            workflow_direction=WorkflowDirection.STOP,
            topic="stopping the scheduling process",
        )
    )

    assert decision.kind is CommunicationKind.DECLINE_WORKFLOW


def test_unknown_factual_yes_no_is_not_blindly_agreed_to() -> None:
    decision = decide(
        TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=WorkflowRelation.NONE,
            question_kind=QuestionKind.PATIENT_ATTRIBUTE,
            workflow_direction=WorkflowDirection.NONE,
            topic="whether the patient is a new patient",
        )
    )

    assert decision.kind is CommunicationKind.CLARIFY


def test_requested_fact_has_priority_over_generic_workflow_semantics() -> None:
    decision = decide(
        TurnMeaning(
            response_expectation=ResponseExpectation.YES_NO,
            workflow_relation=WorkflowRelation.ADVANCES_OBJECTIVE,
            requested_facts=("name",),
            topic="confirming patient identity",
        )
    )

    assert decision.kind is CommunicationKind.ANSWER
    assert decision.facts_to_communicate == ("name",)


def test_ollama_fallback_accepts_general_semantics_and_receives_objective() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.update(body)

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "response_expectation": "yes_no",
                            "workflow_relation": "advances_objective",
                            "question_kind": "workflow_permission",
                            "workflow_direction": "continue",
                            "topic": "creating a patient profile",
                            "requested_facts": [],
                            "stated_facts": [],
                            "appointment_offer": None,
                            "booking_confirmed": False,
                            "conversation_end_requested": False,
                            "requests_repetition": False,
                            "unclear": False,
                        }
                    ),
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )
    scenario = build_scenario()
    interpreter = OllamaConversationInterpreter(client=client)

    try:
        meaning = interpreter.interpret(
            scenario=scenario,
            state=build_initial_state(scenario),
            agent_turn=(
                "There is an auxiliary intake pathway available. "
                "Please tell me whether that makes sense for you."
            ),
        )
    finally:
        interpreter.close()
        client.close()

    assert meaning.response_expectation is ResponseExpectation.YES_NO
    assert meaning.workflow_relation is WorkflowRelation.ADVANCES_OBJECTIVE
    assert meaning.question_kind is QuestionKind.WORKFLOW_PERMISSION
    assert meaning.workflow_direction is WorkflowDirection.CONTINUE
    assert meaning.topic == "creating a patient profile"

    messages = captured["messages"]
    assert isinstance(messages, list)

    user_message = messages[-1]
    assert isinstance(user_message, dict)

    context = json.loads(user_message["content"])
    assert context["conversation_objective"] == scenario.objective
    assert (
        context["latest_tested_agent_turn"]
        == "There is an auxiliary intake pathway available. Please tell me whether that makes sense for you."
    )

    serialized_prompt = json.dumps(messages)
    assert "response_expectation" in serialized_prompt
    assert "workflow_relation" in serialized_prompt


def test_agree_verbalizer_does_not_receive_patient_facts() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.update(body)

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"text": "Yes, please."}),
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )
    scenario = build_scenario()
    verbalizer = OllamaNaturalVerbalizer(client=client)

    try:
        response = verbalizer.verbalize(
            scenario=scenario,
            state=build_initial_state(scenario),
            decision=CommunicationDecision(
                kind=CommunicationKind.AGREE,
            ),
        )
    finally:
        verbalizer.close()
        client.close()

    assert response == "Yes, please."

    serialized = json.dumps(captured["messages"])

    assert "Alex Morgan" not in serialized
    assert "right shoulder pain" not in serialized
    assert "Blue Cross" not in serialized
    assert "Friday" not in serialized

    messages = captured["messages"]
    assert isinstance(messages, list)
    context = json.loads(messages[-1]["content"])
    assert context["communication_kind"] == "agree"
    assert "agree" in context["speech_goal"].casefold()
