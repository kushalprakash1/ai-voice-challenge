import httpx

from voiceprobe.autonomous_phone import (
    PRE_RENDERED_TTS_TEXTS,
)
from voiceprobe.conversation.session import (
    PatientSession,
)
from voiceprobe.interpreters.ollama import (
    OllamaConversationInterpreter,
)
from voiceprobe.scenarios.catalog import (
    get_scenario,
)
from voiceprobe.verbalizers.deterministic import (
    DeterministicNaturalVerbalizer,
)


def run_without_ollama(text: str):
    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        raise AssertionError(
            "Known scheduling offer reached Ollama: "
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

    scenario = get_scenario(
        "autonomous-phone-diagnostic"
    )

    session = PatientSession(
        scenario=scenario,
        interpreter=interpreter,
        verbalizer=(
            DeterministicNaturalVerbalizer()
        ),
    )

    try:
        result = session.handle_agent_turn(
            text
        )
    finally:
        interpreter.close()
        client.close()

    assert calls == 0

    return result


def test_exact_live_morning_offer_is_rejected() -> None:
    result = run_without_ollama(
        "Would you like to book a morning slot "
        "at 9, 9.45 or 10.30 a.m.?"
    )

    offer = result.meaning.appointment_offer

    assert offer is not None
    assert offer.day is None
    assert offer.time == "morning"

    assert (
        result.decision.kind.value
        == "decline_offer"
    )

    assert (
        result.patient_text
        == "No, I need Friday afternoon."
    )


def test_matching_afternoon_offer_is_not_rejected() -> None:
    result = run_without_ollama(
        "Would you like to book "
        "an afternoon slot?"
    )

    offer = result.meaning.appointment_offer

    assert offer is not None
    assert offer.time == "afternoon"

    assert (
        result.decision.kind.value
        != "decline_offer"
    )


def test_non_scheduling_morning_permission_stays_non_offer() -> None:
    result = run_without_ollama(
        "Would you like to create "
        "a morning demo profile?"
    )

    assert (
        result.meaning.appointment_offer
        is None
    )


def test_exact_decline_is_prerendered() -> None:
    assert (
        "No, I need Friday afternoon."
        in PRE_RENDERED_TTS_TEXTS
    )


def test_checking_afternoon_availability_remains_permission() -> None:
    result = run_without_ollama(
        "Would you like me to check "
        "Friday afternoon appointments?"
    )

    assert result.meaning.appointment_offer is None

    assert (
        result.decision.kind.value
        == "agree"
    )

    assert (
        result.patient_text
        == "Yes, please."
    )
