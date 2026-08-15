import json

import httpx
import pytest

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.state import (
    ActionKind,
    PatientAction,
    apply_patient_action,
    build_initial_state,
)
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)
from voiceprobe.verbalizers.ollama import (
    OllamaNaturalVerbalizer,
)


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="shoulder-friday",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )


def test_verbalizes_only_approved_facts() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "text": (
                                "My right shoulder has been hurting "
                                "for about five days."
                            )
                        }
                    ),
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    scenario = build_scenario()

    verbalizer = OllamaNaturalVerbalizer(
        client=client,
    )

    response = verbalizer.verbalize(
        scenario=scenario,
        state=build_initial_state(scenario),
        decision=CommunicationDecision(
            kind=CommunicationKind.ANSWER,
            facts_to_communicate=(
                "complaint",
                "duration",
            ),
        ),
    )

    assert "right shoulder" in response
    assert "five days" in response

    messages = captured_body["messages"]
    serialized = json.dumps(messages)

    assert "right shoulder pain" in serialized
    assert "five days" in serialized
    assert "Blue Cross" not in serialized
    assert "Alex Morgan" not in serialized

    client.close()


def test_rejects_unapproved_fact_leakage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"text": ("My right shoulder hurts, and I have Blue Cross.")}
                    ),
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    scenario = build_scenario()

    verbalizer = OllamaNaturalVerbalizer(
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="unapproved scenario fact",
    ):
        verbalizer.verbalize(
            scenario=scenario,
            state=build_initial_state(scenario),
            decision=CommunicationDecision(
                kind=CommunicationKind.ANSWER,
                facts_to_communicate=("complaint",),
            ),
        )

    client.close()


def test_repeat_exposes_previous_patient_message() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "text": (
                                "My right shoulder has been hurting "
                                "for about five days."
                            )
                        }
                    ),
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    scenario = build_scenario()
    state = build_initial_state(scenario)

    state = apply_patient_action(
        state,
        scenario,
        PatientAction(
            kind=ActionKind.ANSWER,
            response=("My right shoulder has been hurting for about five days."),
            facts_used=(
                "complaint",
                "duration",
            ),
        ),
    )

    verbalizer = OllamaNaturalVerbalizer(
        client=client,
    )

    verbalizer.verbalize(
        scenario=scenario,
        state=state,
        decision=CommunicationDecision(
            kind=CommunicationKind.REPEAT,
        ),
    )

    serialized = json.dumps(captured_body["messages"])

    assert "My right shoulder has been hurting for about five days." in serialized

    client.close()


def test_rejects_state_from_different_scenario() -> None:
    scenario = build_scenario()

    other = PatientScenario(
        scenario_id="other",
        objective="Schedule another appointment.",
        facts=PatientFacts(
            name="Jordan Lee",
            complaint="ankle pain",
            duration="three days",
        ),
    )

    verbalizer = OllamaNaturalVerbalizer(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(500))
        )
    )

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        verbalizer.verbalize(
            scenario=scenario,
            state=build_initial_state(other),
            decision=CommunicationDecision(
                kind=CommunicationKind.CLARIFY,
            ),
        )


def test_repeat_rejects_new_fact_not_in_previous_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "text": (
                                "My right shoulder has been hurting "
                                "for about five days, and I have Blue Cross."
                            )
                        }
                    ),
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    scenario = build_scenario()
    state = build_initial_state(scenario)

    state = apply_patient_action(
        state,
        scenario,
        PatientAction(
            kind=ActionKind.ANSWER,
            response=("My right shoulder has been hurting for about five days."),
            facts_used=("complaint", "duration"),
        ),
    )

    verbalizer = OllamaNaturalVerbalizer(
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="unapproved scenario fact: insurance",
    ):
        verbalizer.verbalize(
            scenario=scenario,
            state=state,
            decision=CommunicationDecision(
                kind=CommunicationKind.REPEAT,
            ),
        )

    client.close()


def test_correction_request_contains_patient_fact_goal() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "text": (
                                "Actually, it's right shoulder pain, "
                                "and it's been five days."
                            )
                        }
                    ),
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    scenario = build_scenario()

    verbalizer = OllamaNaturalVerbalizer(
        client=client,
    )

    verbalizer.verbalize(
        scenario=scenario,
        state=build_initial_state(scenario),
        decision=CommunicationDecision(
            kind=CommunicationKind.CORRECT,
            facts_to_communicate=(
                "complaint",
                "duration",
            ),
        ),
    )

    messages = captured_body["messages"]
    assert isinstance(messages, list)

    user_message = messages[-1]
    assert isinstance(user_message, dict)

    user_content = user_message["content"]
    assert isinstance(user_content, str)

    context = json.loads(user_content)
    goal = context["speech_goal"].casefold()

    assert context["communication_kind"] == "correct"
    assert "correct" in goal
    assert "patient information" in goal
    assert "appointment" in goal

    client.close()


def test_accept_offer_request_requires_conversational_tone() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"text": ("Yeah, Friday at 2:30 PM works for me.")}
                    ),
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    scenario = build_scenario()

    verbalizer = OllamaNaturalVerbalizer(
        client=client,
    )

    verbalizer.verbalize(
        scenario=scenario,
        state=build_initial_state(scenario),
        decision=CommunicationDecision(
            kind=CommunicationKind.ACCEPT_OFFER,
            offered_day="Friday",
            offered_time="2:30 PM",
        ),
    )

    messages = captured_body["messages"]
    assert isinstance(messages, list)

    user_message = messages[-1]
    assert isinstance(user_message, dict)

    user_content = user_message["content"]
    assert isinstance(user_content, str)

    context = json.loads(user_content)
    goal = context["speech_goal"].casefold()

    assert context["communication_kind"] == "accept_offer"
    assert "confirm" in goal
    assert "conversational" in goal
    assert "formal" in goal

    client.close()
