"""Exact evaluator for SemanticFrame-native SemanticLab v2 candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .semantic_corpus import SemanticLabCase
from .semantic_frame import SemanticFrame


@dataclass(frozen=True, slots=True)
class SemanticFrameFailure:
    field: str
    expected: Any
    actual: Any


def _norm_text(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _norm_set(values: object) -> set[str]:
    return {
        _norm_text(value)
        for value in tuple(values or ())
        if _norm_text(value)
    }


def expected_defaults(case: SemanticLabCase) -> dict[str, Any]:
    expected = dict(case.expected)
    expected.setdefault("requested_fact", "")
    expected.setdefault("failed_constraints", [])
    expected.setdefault("proposed_changes", [])
    expected.setdefault("retained_constraints", [])
    expected.setdefault("offered_options", [])
    expected.setdefault("selected_option", "")
    expected.setdefault("record_claims", [])
    expected.setdefault("transaction_operation", "none")
    expected.setdefault("transaction_signal", "none")
    expected.setdefault("reference", "none")
    expected.setdefault("ambiguity", {})
    return expected


def evaluate_frame(
    case: SemanticLabCase,
    frame: SemanticFrame,
) -> tuple[SemanticFrameFailure, ...]:
    exp = expected_defaults(case)
    failures: list[SemanticFrameFailure] = []

    scalar_pairs = {
        "speech_act": (exp.get("speech_act", ""), frame.speech_act.value),
        "topic": (exp.get("topic", ""), frame.topic.value),
        "requested_fact": (exp["requested_fact"], frame.requested_fact),
        "selected_option": (exp["selected_option"], frame.selected_option),
        "transaction_operation": (
            exp["transaction_operation"],
            frame.transaction_operation.value,
        ),
        "transaction_signal": (
            exp["transaction_signal"],
            frame.transaction_signal.value,
        ),
        "reference": (exp["reference"], frame.reference.value),
    }

    for field, (expected, actual) in scalar_pairs.items():
        if _norm_text(expected) != _norm_text(actual):
            failures.append(
                SemanticFrameFailure(
                    field=field,
                    expected=_norm_text(expected),
                    actual=_norm_text(actual),
                )
            )

    set_pairs = {
        "failed_constraints": (
            exp["failed_constraints"],
            [value.value for value in frame.failed_constraints],
        ),
        "proposed_changes": (
            exp["proposed_changes"],
            [value.value for value in frame.proposed_changes],
        ),
        "retained_constraints": (
            exp["retained_constraints"],
            [value.value for value in frame.retained_constraints],
        ),
        "offered_options": (exp["offered_options"], frame.offered_options),
        "record_claims": (
            exp["record_claims"],
            [value.value for value in frame.record_claims],
        ),
    }

    for field, (expected, actual) in set_pairs.items():
        exp_set = _norm_set(expected)
        act_set = _norm_set(actual)
        if exp_set != act_set:
            failures.append(
                SemanticFrameFailure(
                    field=field,
                    expected=sorted(exp_set),
                    actual=sorted(act_set),
                )
            )

    exp_ambiguity = exp["ambiguity"] or {}
    if not exp_ambiguity:
        expected_kind = "none"
        expected_candidates: set[str] = set()
    else:
        expected_kind = _norm_text(exp_ambiguity.get("kind", "none"))
        expected_candidates = _norm_set(exp_ambiguity.get("candidates", ()))

    actual_kind = frame.ambiguity.kind.value
    actual_candidates = _norm_set(frame.ambiguity.candidates)

    if expected_kind != actual_kind:
        failures.append(
            SemanticFrameFailure(
                field="ambiguity.kind",
                expected=expected_kind,
                actual=actual_kind,
            )
        )

    if expected_candidates != actual_candidates:
        failures.append(
            SemanticFrameFailure(
                field="ambiguity.candidates",
                expected=sorted(expected_candidates),
                actual=sorted(actual_candidates),
            )
        )

    return tuple(failures)
