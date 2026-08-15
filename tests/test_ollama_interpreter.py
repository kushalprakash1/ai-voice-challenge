import json

import httpx

from voiceprobe.conversation.state import build_initial_state
from voiceprobe.interpreters.ollama import (
    OllamaConversationInterpreter,
)
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
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


def test_parses_semantic_interpretation() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        body = json.loads(request.content)

        assert body["stream"] is False
        assert body["think"] is False
        assert body["options"]["num_predict"] == 256

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": (
                        "{"
                        '"requested_facts":'
                        '["complaint","duration"],'
                        '"stated_facts":['
                        "{"
                        '"fact":"complaint",'
                        '"value":"left knee pain"'
                        "},"
                        "{"
                        '"fact":"duration",'
                        '"value":"two weeks"'
                        "}"
                        "],"
                        '"appointment_offer":null,'
                        '"booking_confirmed":false,'
                        '"requests_repetition":false,'
                        '"unclear":false'
                        "}"
                    ),
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    scenario = build_scenario()

    interpreter = OllamaConversationInterpreter(
        client=client,
    )

    meaning = interpreter.interpret(
        scenario=scenario,
        state=build_initial_state(scenario),
        agent_turn=("So your left knee has been hurting for two weeks, right?"),
    )

    assert meaning.requested_facts == (
        "complaint",
        "duration",
    )

    assert len(meaning.stated_facts) == 2

    complaint = meaning.stated_facts[0]
    assert complaint.fact == "complaint"
    assert complaint.value == "left knee pain"

    duration = meaning.stated_facts[1]
    assert duration.fact == "duration"
    assert duration.value == "two weeks"

    assert not meaning.booking_confirmed
    assert not meaning.requests_repetition
    assert not meaning.unclear

    client.close()
