import json

import httpx

from voiceprobe.conversation.state import (
    build_initial_state,
    record_agent_turn,
)
from voiceprobe.interpreters.ollama import (
    OllamaConversationInterpreter,
)
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)


def test_interpreter_does_not_send_prior_turn_history() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "requested_facts": [
                                "preferred_day",
                                "preferred_time",
                            ],
                            "stated_facts": [],
                            "appointment_offer": None,
                            "booking_confirmed": False,
                            "requests_repetition": False,
                            "unclear": False,
                        }
                    ),
                }
            },
        )

    scenario = PatientScenario(
        scenario_id="turn-locality",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )

    state = build_initial_state(scenario)

    state = record_agent_turn(
        state,
        "HISTORY_SENTINEL_7319",
    )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    interpreter = OllamaConversationInterpreter(
        client=client,
    )

    meaning = interpreter.interpret(
        scenario=scenario,
        state=state,
        agent_turn=("What day and time would work best for you?"),
    )

    serialized = json.dumps(captured)

    assert "HISTORY_SENTINEL_7319" not in serialized
    assert "What day and time would work best for you?" in serialized

    assert meaning.requested_facts == (
        "preferred_day",
        "preferred_time",
    )

    options = captured["options"]
    assert isinstance(options, dict)
    assert options["num_predict"] == 256

    client.close()
