"""Ground semantic claims against authoritative patient facts."""

from __future__ import annotations

from dataclasses import dataclass

from voiceprobe.conversation.meaning import TurnMeaning
from voiceprobe.conversation.scheduling import time_matches_preference
from voiceprobe.conversation.state import FactKey
from voiceprobe.scenarios.models import PatientScenario


@dataclass(frozen=True, slots=True)
class FactConflict:
    """One tested-agent claim that conflicts with scenario truth."""

    fact: FactKey
    heard_value: str
    authoritative_value: str


@dataclass(frozen=True, slots=True)
class GroundedTurnMeaning:
    """Semantic meaning enriched with deterministic fact conflicts."""

    meaning: TurnMeaning
    conflicts: tuple[FactConflict, ...]


def normalize_comparison_text(value: str) -> str:
    """Normalize text for conservative semantic-value comparison."""
    return " ".join(value.lower().split())


def _normalize_fact_value(
    fact: FactKey,
    value: str,
) -> str:
    """Normalize one fact value for deterministic comparison."""
    normalized = normalize_comparison_text(value)

    if fact == "insurance":
        for suffix in (
            " health insurance",
            " insurance",
            " coverage",
            " plan",
        ):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()
                break

    return normalized


def ground_turn_meaning(
    *,
    scenario: PatientScenario,
    meaning: TurnMeaning,
) -> GroundedTurnMeaning:
    """Compare extracted tested-agent claims with scenario truth."""
    conflicts: list[FactConflict] = []

    for assertion in meaning.stated_facts:
        authoritative = getattr(
            scenario.facts,
            assertion.fact,
        )

        if authoritative is None:
            continue

        if assertion.fact == "preferred_time":
            forward_match = time_matches_preference(
                preferred=str(authoritative),
                offered=assertion.value,
            )
            reverse_match = time_matches_preference(
                preferred=assertion.value,
                offered=str(authoritative),
            )

            if forward_match or reverse_match:
                continue

        heard = _normalize_fact_value(
            assertion.fact,
            assertion.value,
        )
        truth = _normalize_fact_value(
            assertion.fact,
            str(authoritative),
        )

        if heard == truth:
            continue

        conflicts.append(
            FactConflict(
                fact=assertion.fact,
                heard_value=assertion.value,
                authoritative_value=str(authoritative),
            )
        )

    return GroundedTurnMeaning(
        meaning=meaning,
        conflicts=tuple(conflicts),
    )
