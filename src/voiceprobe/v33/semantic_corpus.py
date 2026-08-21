"""SemanticLab v2 corpus loading and schema validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .semantic_frame import (
    AmbiguityKind,
    ConstraintAxis,
    RecordClaim,
    ReferenceKind,
    SemanticTopic,
    SpeechAct,
    TransactionOperation,
    TransactionSignal,
)


@dataclass(frozen=True, slots=True)
class SemanticLabCase:
    case_id: str
    category: str
    utterance: str
    context: tuple[str, ...]
    expected: dict[str, Any]
    tags: tuple[str, ...]


def default_corpus_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tests" / "data" / "semanticlab_v2_cases.jsonl"


def load_semanticlab_cases(path: str | Path | None = None) -> tuple[SemanticLabCase, ...]:
    source = Path(path) if path is not None else default_corpus_path()
    cases: list[SemanticLabCase] = []

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            case = SemanticLabCase(
                case_id=str(raw["case_id"]),
                category=str(raw["category"]),
                utterance=str(raw["utterance"]),
                context=tuple(str(v) for v in raw.get("context", ())),
                expected=dict(raw.get("expected", {})),
                tags=tuple(str(v) for v in raw.get("tags", ())),
            )
            validate_case(case, location=f"{source}:{line_number}")
            cases.append(case)

    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("SemanticLab v2 corpus contains duplicate case_id values.")

    return tuple(cases)


def validate_case(case: SemanticLabCase, *, location: str = "") -> None:
    prefix = f"{location}: " if location else ""

    if not case.case_id.strip():
        raise ValueError(prefix + "case_id must not be empty")
    if not case.category.strip():
        raise ValueError(prefix + "category must not be empty")
    if not case.utterance.strip():
        raise ValueError(prefix + "utterance must not be empty")

    expected = case.expected

    _validate_enum(expected, "speech_act", SpeechAct, prefix)
    _validate_enum(expected, "topic", SemanticTopic, prefix)
    _validate_enum(expected, "reference", ReferenceKind, prefix)
    _validate_enum(expected, "transaction_operation", TransactionOperation, prefix)
    _validate_enum(expected, "transaction_signal", TransactionSignal, prefix)

    for key in ("failed_constraints", "proposed_changes", "retained_constraints"):
        values = tuple(expected.get(key, ()))
        if len(values) != len(set(values)):
            raise ValueError(prefix + f"{key} contains duplicate values")
        for value in values:
            ConstraintAxis(str(value))

    changed = set(expected.get("proposed_changes", ()))
    retained = set(expected.get("retained_constraints", ()))
    overlap = changed & retained
    if overlap:
        raise ValueError(
            prefix + "the same axis cannot be both proposed_changes and retained_constraints: "
            + ", ".join(sorted(str(v) for v in overlap))
        )

    claims = tuple(expected.get("record_claims", ()))
    if len(claims) != len(set(claims)):
        raise ValueError(prefix + "record_claims contains duplicate values")
    for value in claims:
        RecordClaim(str(value))

    ambiguity = expected.get("ambiguity") or {}
    if ambiguity:
        kind = AmbiguityKind(str(ambiguity["kind"]))
        candidates = tuple(str(v) for v in ambiguity.get("candidates", ()))
        if kind is AmbiguityKind.NONE:
            raise ValueError(prefix + "explicit ambiguity cannot use kind=none")
        if len(candidates) < 2:
            raise ValueError(prefix + "explicit ambiguity requires at least two candidates")


def _validate_enum(
    expected: dict[str, Any],
    key: str,
    enum_type: type,
    prefix: str,
) -> None:
    value = expected.get(key)
    if value is None:
        return
    try:
        enum_type(str(value))
    except ValueError as exc:
        raise ValueError(prefix + f"invalid {key}={value!r}") from exc


def cases_with_tag(
    cases: Iterable[SemanticLabCase],
    tag: str,
) -> tuple[SemanticLabCase, ...]:
    return tuple(case for case in cases if tag in case.tags)
