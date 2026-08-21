import httpx

from voiceprobe.conversation.session import PatientSession
from voiceprobe.interpreters.ollama import (
    OllamaConversationInterpreter,
)
from voiceprobe.interpreters.semantic_gate import (
    deterministic_turn_meaning,
)
from voiceprobe.scenarios.catalog import get_scenario
from voiceprobe.verbalizers.deterministic import (
    DeterministicNaturalVerbalizer,
)


def test_call10_truncated_first_name_requires_no_ollama() -> None:
    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        raise AssertionError(
            "Truncated first-name request reached Ollama: "
            f"{request.method} {request.url}"
        )

    client = httpx.Client(
        transport=httpx.MockTransport(
            forbidden
        )
    )

    interpreter = OllamaConversationInterpreter(
        model="qwen3:14b",
        client=client,
    )

    session = PatientSession(
        scenario=get_scenario(
            "autonomous-phone-diagnostic"
        ),
        interpreter=interpreter,
        verbalizer=(
            DeterministicNaturalVerbalizer()
        ),
    )

    try:
        result = session.handle_agent_turn(
            "I just need your first"
        )
    finally:
        interpreter.close()
        client.close()

    assert calls == 0

    assert (
        result.meaning.requested_facts
        == ("first_name",)
    )

    assert result.decision.kind.value == "answer"
    assert result.patient_text == "Alex."


def test_unrelated_first_phrase_is_not_first_name() -> None:
    scenario = get_scenario(
        "autonomous-phone-diagnostic"
    )

    meaning = deterministic_turn_meaning(
        scenario=scenario,
        agent_turn=(
            "I just need your first appointment choice"
        ),
    )

    if meaning is not None:
        assert (
            meaning.requested_facts
            != ("first_name",)
        )
