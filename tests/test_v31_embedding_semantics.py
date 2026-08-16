import asyncio

from voiceprobe.v3.embedding_semantics import (
    CompositionalEmbeddingClassifier,
    EmbeddingSemanticResult,
    split_clauses,
)
from voiceprobe.v3.flow_state import FlowStage
from voiceprobe.v3.models import DecisionKind
from voiceprobe.v3.semantic_router import V31SemanticRouter


def test_clause_split_removes_setup_context() -> None:
    assert split_clauses(
        "Once I know the reason, what type of appointment is this?"
    ) == ("what type of appointment is this?",)


def test_clause_split_preserves_compound_run3_question() -> None:
    assert split_clauses(
        "For -- What is the reason for your appointment? "
        "For example, is it for a routine visit, a follow-up, "
        "or a specific concern?"
    ) == (
        "For -- What is the reason for your appointment",
        (
            "For example, is it for a routine visit, a follow-up, "
            "or a specific concern?"
        ),
    )


def test_composition_requires_distinct_clauses_for_compound() -> None:
    intent, _, _ = CompositionalEmbeddingClassifier._compose(
        clause_fact_scores=[
            {
                "appointment_type_request": 0.914,
                "visit_reason_request": 0.880,
            }
        ],
        clause_speech_scores=[{}],
    )
    assert intent == "appointment_type_request"

    intent, _, _ = CompositionalEmbeddingClassifier._compose(
        clause_fact_scores=[
            {
                "visit_reason_request": 0.91,
                "appointment_type_request": 0.70,
            },
            {
                "appointment_type_request": 0.90,
                "visit_reason_request": 0.69,
            },
        ],
        clause_speech_scores=[{}, {}],
    )
    assert intent == "visit_reason_and_type_request"


def test_strong_profile_speech_act_beats_incidental_fact_similarity() -> None:
    intent, _, _ = CompositionalEmbeddingClassifier._compose(
        clause_fact_scores=[
            {
                "appointment_type_request": 0.753,
                "full_name_request": 0.693,
            }
        ],
        clause_speech_scores=[
            {
                "profile_create_request": 0.936,
                "unknown": 0.647,
            }
        ],
    )
    assert intent == "profile_create_request"


def test_strong_unknown_beats_supported_fact_false_positive() -> None:
    intent, _, _ = CompositionalEmbeddingClassifier._compose(
        clause_fact_scores=[
            {
                "provider_preference_request": 0.669,
                "visit_reason_request": 0.645,
            }
        ],
        clause_speech_scores=[
            {
                "unknown": 0.737,
                "scheduling_complex": 0.592,
            }
        ],
    )
    assert intent == "unknown"


def test_open_ended_help_beats_incidental_name_similarity() -> None:
    intent, _, _ = CompositionalEmbeddingClassifier._compose(
        clause_fact_scores=[
            {
                "full_name_request": 0.713,
                "first_name_request": 0.678,
                "visit_reason_request": 0.666,
            }
        ],
        clause_speech_scores=[
            {
                "open_ended_help": 0.892,
                "presence_check": 0.667,
                "unknown": 0.645,
            }
        ],
    )
    assert intent == "open_ended_help"


def test_open_ended_help_does_not_steal_real_name_request() -> None:
    intent, _, _ = CompositionalEmbeddingClassifier._compose(
        clause_fact_scores=[
            {
                "full_name_request": 0.921,
                "first_name_request": 0.857,
                "last_name_request": 0.788,
            }
        ],
        clause_speech_scores=[
            {
                "open_ended_help": 0.735,
                "profile_create_request": 0.685,
            }
        ],
    )
    assert intent == "full_name_request"


class FakeEmbeddingClassifier:
    async def classify(self, text: str) -> EmbeddingSemanticResult:
        assert "routine visit" in text
        return EmbeddingSemanticResult(
            intent="visit_reason_and_type_request",
            confidence=0.91,
            score=0.91,
            margin=0.12,
            source="embedding_v31:test",
            clauses=(
                "What is the reason for your appointment",
                "is it a routine visit or specific concern?",
            ),
        )


def test_v31_router_maps_embedding_compound_to_deterministic_patient_text() -> None:
    async def scenario() -> None:
        router = V31SemanticRouter(
            embedding_classifier=FakeEmbeddingClassifier(),
        )
        before = router.scorer  # proves old scorer remains available/fallback-safe
        assert before is not None

        from voiceprobe.v3.flow_state import SchedulingFlowTracker

        snapshot = SchedulingFlowTracker().snapshot()
        decision = await router.resolve(
            (
                "What is the reason for your appointment? "
                "For example, is it for a routine visit, a follow-up, "
                "or a specific concern?"
            ),
            snapshot,
        )

        assert decision.kind == DecisionKind.ANSWER_VISIT_DETAILS
        assert decision.text == (
            "I have right shoulder pain. "
            "This is for a new patient consultation."
        )
        assert decision.reason == "reason_and_visit_type_requested"
        assert decision.confidence == 0.91

    asyncio.run(scenario())
