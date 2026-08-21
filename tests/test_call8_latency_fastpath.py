
import httpx

from voiceprobe.autonomous_phone import (
    PRE_RENDERED_TTS_TEXTS,
)
from voiceprobe.conversation.meaning import (
    QuestionKind,
)
from voiceprobe.conversation.state import (
    build_initial_state,
)
from voiceprobe.interpreters.ollama import (
    OllamaConversationInterpreter,
)
from voiceprobe.scenarios.catalog import (
    get_scenario,
)


def interpret_without_ollama(text: str):
    calls = 0

    def forbidden(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        raise AssertionError(
            "Routine intake unexpectedly reached Ollama: "
            f"{request.method} {request.url}"
        )

    client = httpx.Client(
        transport=httpx.MockTransport(forbidden),
    )

    interpreter = OllamaConversationInterpreter(
        model="qwen3:14b",
        client=client,
    )

    scenario = get_scenario(
        "autonomous-phone-diagnostic"
    )

    try:
        meaning = interpreter.interpret(
            scenario=scenario,
            state=build_initial_state(scenario),
            agent_turn=text,
        )
    finally:
        interpreter.close()
        client.close()

    assert calls == 0

    return meaning


def test_call8_last_name_never_calls_ollama() -> None:
    meaning = interpret_without_ollama(
        "Thanks, Alex. And your last name?"
    )

    assert (
        meaning.question_kind
        is QuestionKind.PATIENT_ATTRIBUTE
    )
    assert meaning.requested_facts == (
        "last_name",
    )


def test_call8_appointment_type_never_calls_ollama() -> None:
    meaning = interpret_without_ollama(
        "What type of appointment do you need? "
        "For example, is this for a new patient "
        "consultation? a follow-up a general office "
        "visit or something else."
    )

    assert (
        meaning.question_kind
        is QuestionKind.PATIENT_ATTRIBUTE
    )
    assert meaning.requested_facts == (
        "appointment_type",
    )


def test_visit_category_never_calls_ollama() -> None:
    meaning = interpret_without_ollama(
        "Is this appointment for a routine checkup, "
        "a follow-up after a previous visit, "
        "an urgent concern, or a specific procedure?"
    )

    assert meaning.requested_facts == (
        "appointment_type",
    )


def test_patient_status_never_calls_ollama() -> None:
    meaning = interpret_without_ollama(
        "Are you a new patient, "
        "or have you visited us before?"
    )

    assert meaning.requested_facts == (
        "patient_status",
        "visited_before",
    )


def test_routine_answers_are_prerendered() -> None:
    required = {
        "Alex.",
        "Morgan.",
        "April 12, 1998.",
        "Blue Cross.",
        "Friday afternoon.",
        "I need a new patient consultation.",
        "I'm a new patient. No, I haven't visited before.",
    }

    assert required.issubset(
        set(PRE_RENDERED_TTS_TEXTS)
    )
