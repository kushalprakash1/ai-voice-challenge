import asyncio

from voiceprobe.v3.flow_state import FlowSnapshot, FlowStage
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.semantic_router import (
    SemanticClassification,
    SemanticIntent,
    V31SemanticRouter,
)


def snapshot(stage: FlowStage) -> FlowSnapshot:
    return FlowSnapshot(
        communicated=frozenset(),
        confirmed=frozenset(),
        current_stage=stage,
        complete=False,
    )


class ForbiddenClassifier:
    async def __call__(self, agent_turn, state):
        raise AssertionError(
            f"High-confidence prototype unexpectedly reached Qwen: {agent_turn!r}"
        )


def test_visit_reason_paraphrases_converge_without_model() -> None:
    async def scenario():
        router = V31SemanticRouter(classifier=ForbiddenClassifier())
        cases = (
            "What is the reason for your visit?",
            "What is the reason for your appointment?",
            "What brings you in?",
            "What are we seeing you for?",
            "What is this appointment for?",
            "What is the specific concern?",
        )

        for text in cases:
            result = await router.resolve(
                text,
                snapshot(FlowStage.VISIT_REASON),
            )
            assert result.kind == DecisionKind.ANSWER_COMPLAINT
            assert result.text == "I have right shoulder pain."

    asyncio.run(scenario())


def test_split_wrong_dob_is_corrected() -> None:
    async def fake_classifier(agent_turn, state):
        return SemanticClassification(
            SemanticIntent.DOB_ASSERTION,
            confidence=0.99,
            source="test",
        )

    async def scenario():
        router = V31SemanticRouter(classifier=fake_classifier)
        result = await router.resolve(
            (
                "Your patient profile is set up, and your date of birth is "
                "July fourth two thousand for demo purposes."
            ),
            snapshot(FlowStage.DOB),
        )

        assert result.kind == DecisionKind.CORRECT_FACT
        assert result.reason == "dob_correction"
        assert result.text == "Actually, my date of birth is April 12, 1998."

    asyncio.run(scenario())


def test_split_may_i_help_you_states_objective() -> None:
    async def fake_classifier(agent_turn, state):
        return SemanticClassification(
            SemanticIntent.OPEN_ENDED_HELP,
            confidence=0.98,
            source="test",
        )

    async def scenario():
        router = V31SemanticRouter(classifier=fake_classifier)
        result = await router.resolve(
            "May I help you?",
            snapshot(FlowStage.VISIT_REASON),
        )

        assert result.kind == DecisionKind.STATE_OBJECTIVE
        assert "Friday afternoon" in result.text

    asyncio.run(scenario())


def test_unknown_repeat_cannot_emit_identical_consecutive_clarifications() -> None:
    async def unknown(agent_turn, state):
        return SemanticClassification(
            SemanticIntent.UNKNOWN,
            confidence=0.0,
            source="test",
        )

    async def scenario():
        router = V31SemanticRouter(classifier=unknown)
        state = snapshot(FlowStage.VISIT_REASON)
        results = [
            await router.resolve(
                "Could you unpack the metaphysics of this appointment?",
                state,
            )
            for _ in range(4)
        ]

        assert all(result.kind == DecisionKind.CLARIFY for result in results)
        assert all(
            left.text != right.text
            for left, right in zip(results, results[1:])
        )

    asyncio.run(scenario())


def test_semantic_model_cannot_directly_accept_complex_slot() -> None:
    async def scheduling(agent_turn, state):
        return SemanticClassification(
            SemanticIntent.SCHEDULING_COMPLEX,
            confidence=0.99,
            source="test",
        )

    async def scenario():
        router = V31SemanticRouter(classifier=scheduling)
        result = await router.resolve(
            "I have Tuesday at 2:30 PM if that works.",
            snapshot(FlowStage.SLOT),
        )

        assert result.kind == DecisionKind.CLARIFY
        assert "book" not in result.text.casefold()
        assert "accept" not in result.text.casefold()

    asyncio.run(scenario())


def test_statistical_router_handles_nonliteral_visit_reason_wording() -> None:
    async def scenario():
        router = V31SemanticRouter(classifier=ForbiddenClassifier())

        for text in (
            "Why are you coming in?",
            "Can you tell me why you need the appointment?",
            "What specific concern are you being seen for?",
        ):
            classification = await router.classify(
                text,
                snapshot(FlowStage.VISIT_REASON),
            )
            assert classification.intent == SemanticIntent.VISIT_REASON_REQUEST
            assert classification.source == "statistical_v31"

    asyncio.run(scenario())


def test_statistical_router_handles_common_fact_requests_without_qwen() -> None:
    async def scenario():
        router = V31SemanticRouter(classifier=ForbiddenClassifier())

        cases = (
            (
                "What kind of visit are we scheduling?",
                FlowStage.APPOINTMENT_TYPE,
                SemanticIntent.APPOINTMENT_TYPE_REQUEST,
            ),
            (
                "Which insurance carrier do you use?",
                FlowStage.INSURANCE,
                SemanticIntent.INSURANCE_REQUEST,
            ),
            (
                "What's your DOB?",
                FlowStage.DOB,
                SemanticIntent.DOB_REQUEST,
            ),
            (
                "What is your surname?",
                FlowStage.IDENTITY,
                SemanticIntent.LAST_NAME_REQUEST,
            ),
        )

        for text, stage, expected in cases:
            classification = await router.classify(text, snapshot(stage))
            assert classification.intent == expected
            assert classification.source == "statistical_v31"

    asyncio.run(scenario())
