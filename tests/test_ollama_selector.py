import json

import httpx

from voiceprobe.conversation.state import build_initial_state
from voiceprobe.planners.hybrid import ResponsePlan
from voiceprobe.planners.ollama import OllamaActionSelector
from voiceprobe.scenarios.models import PatientFacts, PatientScenario


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="shoulder-friday",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
        ),
    )


def test_parses_structured_ollama_decision() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)

        assert body["think"] is False
        assert body["stream"] is False
        assert body["options"]["num_predict"] == 20

        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": ('{"plan":"correct_complaint_duration"}'),
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    selector = OllamaActionSelector(
        client=client,
    )

    decision = selector.select(
        scenario=build_scenario(),
        state=build_initial_state(build_scenario()),
        agent_turn=("So your knee has been hurting for two weeks?"),
    )

    assert decision.plan is ResponsePlan.CORRECT_COMPLAINT_DURATION

    client.close()
