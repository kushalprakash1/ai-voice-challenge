from __future__ import annotations

from typing import Any, cast

import httpx

from voiceprobe.conversation.state import (
    build_initial_state,
)
from voiceprobe.interpreters.ollama import (
    OllamaConversationInterpreter,
)
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "message": {
                "content": (
                    '{"requested_facts":["insurance"],'
                    '"stated_facts":[],'
                    '"appointment_offer":null,'
                    '"booking_confirmed":false,'
                    '"requests_repetition":false,'
                    '"unclear":false}'
                )
            }
        }


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def post(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> FakeResponse:
        del args, kwargs
        self.calls += 1
        return FakeResponse()


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="prefetch-test",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            insurance="Blue Cross",
        ),
    )


def test_matching_prefetch_avoids_second_request() -> None:
    scenario = build_scenario()
    state = build_initial_state(scenario)
    fake = FakeClient()

    interpreter = OllamaConversationInterpreter(
        client=cast(httpx.Client, fake),
    )

    try:
        assert interpreter.prefetch(
            scenario=scenario,
            state=state,
            agent_turn="What insurance do you have?",
        )

        result = interpreter.interpret(
            scenario=scenario,
            state=state,
            agent_turn="What insurance do you have?",
        )

        assert result.requested_facts == ("insurance",)
        assert fake.calls == 1
    finally:
        interpreter.close()


def test_invalidated_prefetch_is_never_consumed() -> None:
    scenario = build_scenario()
    state = build_initial_state(scenario)
    fake = FakeClient()

    interpreter = OllamaConversationInterpreter(
        client=cast(httpx.Client, fake),
    )

    try:
        assert interpreter.prefetch(
            scenario=scenario,
            state=state,
            agent_turn="What insurance do you have?",
        )

        interpreter.invalidate_prefetch()

        result = interpreter.interpret(
            scenario=scenario,
            state=state,
            agent_turn="What insurance do you have?",
        )

        assert result.requested_facts == ("insurance",)

        # One speculative request plus one authoritative rerun.
        assert fake.calls == 2
    finally:
        interpreter.close()
