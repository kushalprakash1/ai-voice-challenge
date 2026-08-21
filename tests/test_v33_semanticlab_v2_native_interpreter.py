from __future__ import annotations

from dataclasses import dataclass

import pytest

from voiceprobe.v33.semantic_corpus import SemanticLabCase
from voiceprobe.v33.semantic_frame import (
    AmbiguityKind,
    ConstraintAxis,
    SemanticAmbiguity,
    SemanticFrame,
    SemanticTopic,
    SpeechAct,
    TransactionOperation,
    TransactionSignal,
)
from voiceprobe.v33.semantic_frame_eval import evaluate_frame
from voiceprobe.v33.semantic_interpreter_v2 import OllamaSemanticFrameInterpreter


@dataclass
class FakeBackend:
    response: dict

    async def generate_json(self, *, system, prompt, schema):
        return dict(self.response)


def case(expected):
    return SemanticLabCase(
        case_id="x",
        category="test",
        utterance="x",
        context=(),
        expected=expected,
        tags=(),
    )


def test_native_schema_uses_independent_multi_label_axes():
    props = OllamaSemanticFrameInterpreter.schema()["properties"]
    assert props["failed_constraints"]["type"] == "array"
    assert props["proposed_changes"]["type"] == "array"
    assert props["retained_constraints"]["type"] == "array"
    assert "combination" not in props["failed_constraints"]["items"]["enum"]
    assert "kind" not in props
    assert "respond" not in props
    assert "fallback_target" not in props


@pytest.mark.asyncio
async def test_native_interpreter_preserves_compound_fact_and_fallback():
    interpreter = OllamaSemanticFrameInterpreter(
        backend=FakeBackend(
            {
                "speech_act": "question",
                "topic": "availability",
                "requested_fact": "insurance",
                "failed_constraints": [],
                "proposed_changes": ["time_of_day"],
                "retained_constraints": [],
                "offered_options": [],
                "selected_option": "",
                "record_claims": [],
                "transaction_operation": "none",
                "transaction_signal": "none",
                "reference": "none",
                "ambiguity": {"kind": "none", "candidates": [], "detail": ""},
            }
        )
    )
    frame, _ = await interpreter.interpret(
        remote_turn="What's your insurance, and would mornings work?"
    )
    assert frame.requested_fact == "insurance"
    assert frame.proposed_changes == (ConstraintAxis.TIME_OF_DAY,)


@pytest.mark.asyncio
async def test_native_interpreter_can_represent_multi_axis_failure_exactly():
    interpreter = OllamaSemanticFrameInterpreter(
        backend=FakeBackend(
            {
                "speech_act": "statement",
                "topic": "availability",
                "requested_fact": "none",
                "failed_constraints": ["day", "time_of_day"],
                "proposed_changes": [],
                "retained_constraints": [],
                "offered_options": [],
                "selected_option": "",
                "record_claims": [],
                "transaction_operation": "none",
                "transaction_signal": "none",
                "reference": "none",
                "ambiguity": {"kind": "none", "candidates": [], "detail": ""},
            }
        )
    )
    frame, _ = await interpreter.interpret(
        remote_turn="I don't have anything Friday afternoon."
    )
    assert set(frame.failed_constraints) == {
        ConstraintAxis.DAY,
        ConstraintAxis.TIME_OF_DAY,
    }


def test_evaluator_defaults_unspecified_fields_to_empty_or_none():
    frame = SemanticFrame(
        raw_text="Are you still there?",
        speech_act=SpeechAct.PRESENCE_CHECK,
        topic=SemanticTopic.PRESENCE,
    )
    failures = evaluate_frame(
        case({"speech_act": "presence_check", "topic": "presence"}),
        frame,
    )
    assert failures == ()


def test_evaluator_penalizes_false_positive_semantics():
    frame = SemanticFrame(
        raw_text="Friday is full.",
        speech_act=SpeechAct.STATEMENT,
        topic=SemanticTopic.AVAILABILITY,
        proposed_changes=(ConstraintAxis.DAY,),
    )
    failures = evaluate_frame(
        case({"speech_act": "statement", "topic": "availability"}),
        frame,
    )
    assert any(failure.field == "proposed_changes" for failure in failures)


def test_create_profile_is_first_class_operation():
    assert TransactionOperation.CREATE_PROFILE.value == "create_profile"


def test_search_is_not_itself_transaction_consent():
    frame = SemanticFrame(
        raw_text="Can I check Friday?",
        speech_act=SpeechAct.QUESTION,
        topic=SemanticTopic.AVAILABILITY,
        transaction_operation=TransactionOperation.SEARCH,
        transaction_signal=TransactionSignal.NONE,
    )
    assert frame.transaction_signal is TransactionSignal.NONE


def test_ambiguity_is_scored_instead_of_architecturally_unsupported():
    frame = SemanticFrame(
        raw_text="Could you move it later?",
        speech_act=SpeechAct.QUESTION,
        topic=SemanticTopic.OTHER,
        ambiguity=SemanticAmbiguity(
            kind=AmbiguityKind.TEMPORAL_REFERENCE,
            candidates=("time_of_day", "day"),
        ),
    )
    failures = evaluate_frame(
        case(
            {
                "speech_act": "question",
                "topic": "other",
                "ambiguity": {
                    "kind": "temporal_reference",
                    "candidates": ["time_of_day", "day"],
                },
            }
        ),
        frame,
    )
    assert failures == ()
