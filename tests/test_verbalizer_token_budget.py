from __future__ import annotations

import json

import httpx

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.state import build_initial_state
from voiceprobe.scenarios.catalog import get_scenario
from voiceprobe.verbalizers.ollama import OllamaNaturalVerbalizer


def test_partial_offer_has_enough_structured_output_budget() -> None:
    captured_num_predict: list[int] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))

        captured_num_predict.append(payload["options"]["num_predict"])

        return httpx.Response(
            200,
            json={"message": {"content": ('{"text":"4:30 works. What day is that?"}')}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    scenario = get_scenario("autonomous-phone-diagnostic")
    state = build_initial_state(scenario)

    verbalizer = OllamaNaturalVerbalizer(
        client=client,
    )

    try:
        text = verbalizer.verbalize(
            scenario=scenario,
            state=state,
            decision=CommunicationDecision(
                kind=(CommunicationKind.ACCEPT_PARTIAL_OFFER),
                offered_time="4:30 PM",
            ),
        )
    finally:
        verbalizer.close()
        client.close()

    assert text == "4:30 works. What day is that?"
    assert captured_num_predict == [96]
