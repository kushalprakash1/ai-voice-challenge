"""Evaluation adapter for the frozen VoiceProbe v3.3 v0.17 semantic baseline.

SemanticLab v2 intentionally has a richer contract than v0.17.  This module
scores only meaning that v0.17 can actually represent and reports architectural
coverage gaps separately.  It never turns an unsupported field into a model
failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .semantic_corpus import SemanticLabCase
from .world_model import ObservationKind, RemoteObservation


class EvalStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class FieldEvaluation:
    field: str
    status: EvalStatus
    expected: Any
    actual: Any
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    category: str
    fields: tuple[FieldEvaluation, ...]

    @property
    def scored_fields(self) -> tuple[FieldEvaluation, ...]:
        return tuple(
            item for item in self.fields
            if item.status is not EvalStatus.UNSUPPORTED
        )

    @property
    def failed_fields(self) -> tuple[FieldEvaluation, ...]:
        return tuple(
            item for item in self.fields
            if item.status is EvalStatus.FAIL
        )

    @property
    def unsupported_fields(self) -> tuple[FieldEvaluation, ...]:
        return tuple(
            item for item in self.fields
            if item.status is EvalStatus.UNSUPPORTED
        )

    @property
    def exact_on_supported_fields(self) -> bool | None:
        if not self.scored_fields:
            return None
        return not self.failed_fields


def _normalized_text(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _normalized_text_set(values: object) -> set[str]:
    return {
        _normalized_text(value)
        for value in tuple(values or ())
        if _normalized_text(value)
    }


def _normalized_axis_set(values: object) -> set[str]:
    return {str(value) for value in tuple(values or ())}


def _actual_transaction_signal(observation: RemoteObservation) -> str:
    if observation.kind is ObservationKind.TRANSACTION_PERMISSION_REQUEST:
        return "permission_request"
    if observation.kind is ObservationKind.TRANSACTION_CONFIRMED:
        return "confirmed"
    return "none"


def evaluate_v017_case(
    case: SemanticLabCase,
    observation: RemoteObservation,
) -> CaseEvaluation:
    """Compare one v0.17 observation with labels it can faithfully represent."""

    expected = case.expected
    fields: list[FieldEvaluation] = []

    if "requested_fact" in expected:
        exp = _normalized_text(expected["requested_fact"])
        act = _normalized_text(observation.requested_fact)
        fields.append(
            FieldEvaluation(
                "requested_fact",
                EvalStatus.PASS if exp == act else EvalStatus.FAIL,
                exp,
                act,
            )
        )

    if "failed_constraints" in expected:
        exp = _normalized_axis_set(expected["failed_constraints"])
        act = _normalized_axis_set(observation.unavailable_constraints)
        fields.append(
            FieldEvaluation(
                "failed_constraints",
                EvalStatus.PASS if exp == act else EvalStatus.FAIL,
                sorted(exp),
                sorted(act),
            )
        )

    if "proposed_changes" in expected:
        exp = _normalized_axis_set(expected["proposed_changes"])
        act = _normalized_axis_set(observation.search_constraints)
        fields.append(
            FieldEvaluation(
                "proposed_changes",
                EvalStatus.PASS if exp == act else EvalStatus.FAIL,
                sorted(exp),
                sorted(act),
            )
        )

    if "record_claims" in expected:
        exp = _normalized_text_set(expected["record_claims"])
        act = _normalized_text_set(observation.remote_claims)
        fields.append(
            FieldEvaluation(
                "record_claims",
                EvalStatus.PASS if exp == act else EvalStatus.FAIL,
                sorted(exp),
                sorted(act),
            )
        )

    if "offered_options" in expected:
        exp = _normalized_text_set(expected["offered_options"])
        act = _normalized_text_set(observation.offered_options)
        fields.append(
            FieldEvaluation(
                "offered_options",
                EvalStatus.PASS if exp == act else EvalStatus.FAIL,
                sorted(exp),
                sorted(act),
            )
        )

    if "selected_option" in expected:
        exp = _normalized_text(expected["selected_option"])
        act = _normalized_text(observation.selected_option)
        fields.append(
            FieldEvaluation(
                "selected_option",
                EvalStatus.PASS if exp == act else EvalStatus.FAIL,
                exp,
                act,
            )
        )

    if "transaction_operation" in expected:
        exp = _normalized_text(expected["transaction_operation"])
        act = _normalized_text(observation.transaction_operation)
        if act == "":
            act = "none"
        fields.append(
            FieldEvaluation(
                "transaction_operation",
                EvalStatus.PASS if exp == act else EvalStatus.FAIL,
                exp,
                act,
            )
        )

    if "transaction_signal" in expected:
        exp = _normalized_text(expected["transaction_signal"])
        if exp == "proposed":
            fields.append(
                FieldEvaluation(
                    "transaction_signal",
                    EvalStatus.UNSUPPORTED,
                    exp,
                    None,
                    (
                        "v0.17 has no independent transaction-signal field for "
                        "a non-permission proposal."
                    ),
                )
            )
        else:
            act = _actual_transaction_signal(observation)
            fields.append(
                FieldEvaluation(
                    "transaction_signal",
                    EvalStatus.PASS if exp == act else EvalStatus.FAIL,
                    exp,
                    act,
                )
            )

    retained = tuple(expected.get("retained_constraints", ()))
    if retained:
        fields.append(
            FieldEvaluation(
                "retained_constraints",
                EvalStatus.UNSUPPORTED,
                list(retained),
                None,
                "v0.17 does not expose retained constraints in RemoteObservation.",
            )
        )

    reference = _normalized_text(expected.get("reference", "none"))
    if reference not in {"", "none"}:
        fields.append(
            FieldEvaluation(
                "reference",
                EvalStatus.UNSUPPORTED,
                reference,
                None,
                "v0.17 has no explicit reference-resolution type.",
            )
        )

    ambiguity = expected.get("ambiguity") or {}
    if ambiguity:
        fields.append(
            FieldEvaluation(
                "ambiguity",
                EvalStatus.UNSUPPORTED,
                ambiguity,
                None,
                "v0.17 has no first-class unresolved-ambiguity representation.",
            )
        )

    return CaseEvaluation(
        case_id=case.case_id,
        category=case.category,
        fields=tuple(fields),
    )
