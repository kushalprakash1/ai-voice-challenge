import httpx

from voiceprobe.autonomous_phone import (
    PRE_RENDERED_TTS_TEXTS,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.interpreters.ollama import (
    OllamaConversationInterpreter,
)
from voiceprobe.scenarios.catalog import get_scenario
from voiceprobe.verbalizers.deterministic import (
    DeterministicNaturalVerbalizer,
)


PROVIDER_RESPONSE = (
    "I don't have a preference. "
    "Any available provider is fine."
)


def run_without_ollama(agent_turn: str):
    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        raise AssertionError(
            "Known name/provider question reached Ollama: "
            f"{request.method} {request.url}"
        )

    client = httpx.Client(
        transport=httpx.MockTransport(forbidden)
    )

    scenario = get_scenario(
        "autonomous-phone-diagnostic"
    )

    interpreter = OllamaConversationInterpreter(
        model="qwen3:14b",
        client=client,
    )

    session = PatientSession(
        scenario=scenario,
        interpreter=interpreter,
        verbalizer=DeterministicNaturalVerbalizer(),
    )

    try:
        result = session.handle_agent_turn(
            agent_turn
        )
    finally:
        interpreter.close()
        client.close()

    assert calls == 0

    return result


def test_real_call_combined_name() -> None:
    result = run_without_ollama(
        "Great. Can you tell me your first and last name?"
    )

    assert result.meaning.requested_facts == (
        "first_name",
        "last_name",
    )

    assert result.patient_text == "Alex Morgan."


def test_direct_first_name_remains_alex() -> None:
    result = run_without_ollama(
        "What is your first name?"
    )

    assert result.meaning.requested_facts == (
        "first_name",
    )

    assert result.patient_text == "Alex."


def test_direct_last_name_remains_morgan() -> None:
    result = run_without_ollama(
        "What is your last name?"
    )

    assert result.meaning.requested_facts == (
        "last_name",
    )

    assert result.patient_text == "Morgan."


def test_real_call_which_provider() -> None:
    result = run_without_ollama(
        "Which provider would you like to see "
        "for your new patient consultation?"
    )

    assert result.meaning.requested_facts == (
        "provider_preference",
    )

    assert result.patient_text == PROVIDER_RESPONSE


def test_real_call_provider_name_does_not_leak_patient_name() -> None:
    result = run_without_ollama(
        "Could you tell me the name of the "
        "provider you prefer?"
    )

    assert result.meaning.requested_facts == (
        "provider_preference",
    )

    assert result.patient_text == PROVIDER_RESPONSE
    assert result.patient_text != "Alex Morgan."


def test_real_call_open_to_any_provider() -> None:
    result = run_without_ollama(
        "Do you have or are you open to "
        "any available provider?"
    )

    assert result.meaning.requested_facts == (
        "provider_preference",
    )

    assert result.patient_text == PROVIDER_RESPONSE


def test_provider_response_is_prerendered() -> None:
    assert (
        PROVIDER_RESPONSE
        in PRE_RENDERED_TTS_TEXTS
    )
