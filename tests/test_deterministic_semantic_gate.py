import httpx

from voiceprobe.agents.brain import CommunicationKind, PatientBrain
from voiceprobe.conversation.meaning import (
    QuestionKind,
    ResponseExpectation,
    WorkflowDirection,
    WorkflowRelation,
)
from voiceprobe.conversation.session import PatientSession
from voiceprobe.conversation.state import build_initial_state
from voiceprobe.interpreters.ollama import OllamaConversationInterpreter
from voiceprobe.interpreters.semantic_gate import deterministic_turn_meaning
from voiceprobe.scenarios.models import PatientFacts, PatientScenario
from voiceprobe.verbalizers.deterministic import (
    DeterministicNaturalVerbalizer,
)


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="semantic-gate",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            date_of_birth="April 12, 1998",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )


def test_optional_profile_workflow_is_declined_semantically() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn="Would you like to create a demo patient profile?",
    )

    assert meaning is not None
    assert meaning.response_expectation is ResponseExpectation.YES_NO
    assert meaning.question_kind is QuestionKind.WORKFLOW_PERMISSION
    assert meaning.workflow_direction is WorkflowDirection.CONTINUE
    assert meaning.workflow_relation is WorkflowRelation.NONE
    assert meaning.requested_facts == ()


def test_explicit_agent_profile_workflow_is_declined_semantically() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn="Would you like me to create a temporary patient account?",
    )

    assert meaning is not None
    assert meaning.question_kind is QuestionKind.WORKFLOW_PERMISSION
    assert meaning.workflow_relation is WorkflowRelation.NONE
    assert meaning.requested_facts == ()


def test_checking_appointment_availability_advances_objective() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn="Would you like me to check Friday afternoon appointments?",
    )

    assert meaning is not None
    assert meaning.response_expectation is ResponseExpectation.YES_NO
    assert meaning.question_kind is QuestionKind.WORKFLOW_PERMISSION
    assert meaning.workflow_direction is WorkflowDirection.CONTINUE
    assert meaning.workflow_relation is WorkflowRelation.ADVANCES_OBJECTIVE
    assert meaning.requested_facts == ()


def test_cancellation_opposes_objective() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn="Would you like me to cancel this scheduling request?",
    )

    assert meaning is not None
    assert meaning.question_kind is QuestionKind.WORKFLOW_PERMISSION
    assert meaning.workflow_direction is WorkflowDirection.STOP
    assert meaning.workflow_relation is WorkflowRelation.OPPOSES_OBJECTIVE


def test_generic_continue_remains_uncertain() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn="Would you like me to continue?",
    )

    assert meaning is not None
    assert meaning.workflow_relation is WorkflowRelation.UNCERTAIN


def test_required_date_of_birth_is_fact_request() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn=(
            "Before I can schedule you, I need to verify "
            "your date of birth."
        ),
    )

    assert meaning is not None
    assert meaning.response_expectation is ResponseExpectation.FACT
    assert meaning.question_kind is QuestionKind.PATIENT_ATTRIBUTE
    assert meaning.requested_facts == ("date_of_birth",)


def test_direct_insurance_question_is_fact_request() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn="All right, what's your insurance?",
    )

    assert meaning is not None
    assert meaning.requested_facts == ("insurance",)


def test_common_indirect_fact_questions_are_supported() -> None:
    scenario = build_scenario()

    cases = (
        ("Who am I speaking with?", ("name",)),
        ("What brought you in?", ("complaint",)),
        ("How long has this been going on?", ("duration",)),
        ("Who are you covered through?", ("insurance",)),
        ("Which day works best?", ("preferred_day",)),
        ("What time works best?", ("preferred_time",)),
    )

    for utterance, expected in cases:
        meaning = deterministic_turn_meaning(
            scenario=scenario,
            agent_turn=utterance,
        )

        assert meaning is not None, utterance
        assert meaning.requested_facts == expected, utterance


def test_concrete_slot_offer_is_extracted_deterministically() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn="Friday at 2:30 PM is available. Would that work?",
    )

    assert meaning is not None
    assert meaning.appointment_offer is not None
    assert meaning.appointment_offer.day == "Friday"
    assert meaning.appointment_offer.time == "2:30 PM"
    assert meaning.requested_facts == ()
    assert meaning.workflow_relation is WorkflowRelation.ADVANCES_OBJECTIVE


def test_dot_clock_slot_offer_is_normalized() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn="How about Friday at 2.30 p.m.?",
    )

    assert meaning is not None
    assert meaning.appointment_offer is not None
    assert meaning.appointment_offer.day == "Friday"
    assert meaning.appointment_offer.time == "2:30 PM"


def test_plain_goodbye_is_extracted_without_fake_fact_requests() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn="Okay, goodbye.",
    )

    assert meaning is not None
    assert meaning.conversation_end_requested
    assert meaning.requested_facts == ()
    assert meaning.appointment_offer is None
    assert not meaning.booking_confirmed


def test_booking_confirmation_and_goodbye_extracts_slot() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn=(
            "Yes, your Friday at 2:30 PM appointment "
            "is booked. Goodbye."
        ),
    )

    assert meaning is not None
    assert meaning.booking_confirmed
    assert meaning.conversation_end_requested
    assert meaning.appointment_offer is not None
    assert meaning.appointment_offer.day == "Friday"
    assert meaning.appointment_offer.time == "2:30 PM"
    assert meaning.requested_facts == ()


def test_deterministic_turn_never_calls_ollama() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(
            "Ollama must not be called for deterministic semantic turns."
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    scenario = build_scenario()
    interpreter = OllamaConversationInterpreter(client=client)

    utterances = (
        "Would you like me to check Friday afternoon appointments?",
        "Friday at 2:30 PM is available. Would that work?",
        "Okay, goodbye.",
        "Your Friday at 2:30 PM appointment is booked. Goodbye.",
    )

    try:
        for utterance in utterances:
            meaning = interpreter.interpret(
                scenario=scenario,
                state=build_initial_state(scenario),
                agent_turn=utterance,
            )

            assert meaning is not None
    finally:
        interpreter.close()
        client.close()

    assert calls == 0


def test_prefetch_skips_deterministic_turn() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(
            "Deterministic prefetch must not call Ollama."
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    scenario = build_scenario()
    interpreter = OllamaConversationInterpreter(client=client)

    try:
        started = interpreter.prefetch(
            scenario=scenario,
            state=build_initial_state(scenario),
            agent_turn="Friday at 2:30 PM is available. Would that work?",
        )
    finally:
        interpreter.close()
        client.close()

    assert started is False
    assert calls == 0


def test_full_previous_failure_sequence_requires_no_ollama() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(
            "The high-confidence scheduling flow must not call Ollama."
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    scenario = build_scenario()
    interpreter = OllamaConversationInterpreter(client=client)
    verbalizer = DeterministicNaturalVerbalizer()

    session = PatientSession(
        scenario=scenario,
        interpreter=interpreter,
        verbalizer=verbalizer,
        brain=PatientBrain(),
    )

    cases = (
        (
            "Would you like to create a demo patient profile?",
            CommunicationKind.DECLINE_WORKFLOW,
            "No, I need an appointment.",
        ),
        (
            "Before I can schedule you, I need to verify your date of birth.",
            CommunicationKind.ANSWER,
            "April 12, 1998.",
        ),
        (
            "All right, what's your insurance?",
            CommunicationKind.ANSWER,
            "Blue Cross.",
        ),
        (
            "Would you like me to check Friday afternoon appointments?",
            CommunicationKind.AGREE,
            "Yes, please.",
        ),
        (
            "Friday at 2:30 PM is available. Would that work?",
            CommunicationKind.ACCEPT_OFFER,
            "Yes, that works.",
        ),
        (
            "Okay, goodbye.",
            CommunicationKind.VERIFY_BOOKING,
            "Just to confirm, am I booked for Friday at 2:30 PM?",
        ),
        (
            "Yes, your Friday at 2:30 PM appointment is booked. Goodbye.",
            CommunicationKind.END_CONVERSATION,
            "Okay, thank you. Bye.",
        ),
    )

    try:
        for utterance, expected_kind, expected_text in cases:
            result = session.handle_agent_turn(utterance)

            assert result.decision.kind is expected_kind, utterance
            assert result.patient_text == expected_text, utterance

    finally:
        interpreter.close()
        client.close()

    assert calls == 0
    assert session.progress.offered_day == "Friday"
    assert session.progress.offered_time == "2:30 PM"
    assert session.progress.offer_accepted
    assert session.progress.booking_confirmed
    assert session.progress.objective_complete
    assert session.state.objective_complete


def test_real_call_four_great_day_is_premature_end_request() -> None:
    meaning = deterministic_turn_meaning(
        scenario=build_scenario(),
        agent_turn=(
            "You can scan the profile later if you'd like. "
            "Have a great day."
        ),
    )

    assert meaning is not None
    assert meaning.conversation_end_requested is True
    assert meaning.requested_facts == ()
    assert meaning.appointment_offer is None
    assert not meaning.booking_confirmed


def test_real_call_four_great_day_session_pushes_back() -> None:
    import httpx

    from voiceprobe.agents.brain import PatientBrain
    from voiceprobe.conversation.session import PatientSession
    from voiceprobe.interpreters.ollama import (
        OllamaConversationInterpreter,
    )
    from voiceprobe.scenarios.catalog import get_scenario
    from voiceprobe.verbalizers.deterministic import (
        DeterministicNaturalVerbalizer,
    )

    def forbidden_ollama(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"Ollama unexpectedly called: {request.method} {request.url}"
        )

    client = httpx.Client(
        transport=httpx.MockTransport(forbidden_ollama),
    )

    interpreter = OllamaConversationInterpreter(
        model="qwen3:1.7b",
        client=client,
    )

    session = PatientSession(
        scenario=get_scenario("autonomous-phone-diagnostic"),
        interpreter=interpreter,
        verbalizer=DeterministicNaturalVerbalizer(),
        brain=PatientBrain(),
    )

    try:
        first = session.handle_agent_turn(
            "This call may be recorded for quality and training purposes. "
            "Thank you for calling Pivot Point Orthopedics. "
            "Would you like to create a demo patient profile?"
        )

        assert first.decision.kind.value == "decline_workflow"
        assert first.patient_text == "No, I need an appointment."

        final = session.handle_agent_turn(
            "You can scan the profile later if you'd like. "
            "Have a great day."
        )

        assert final.meaning.conversation_end_requested is True
        assert final.decision.kind.value == "decline_workflow"
        assert final.patient_text == "No, I need an appointment."
        assert session.progress.objective_complete is False

    finally:
        interpreter.close()
        client.close()
