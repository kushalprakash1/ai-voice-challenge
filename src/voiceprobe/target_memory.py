"""Structured prior observations about the target voice agent.

This is target-agent knowledge, never caller-persona truth.

Entries are deliberately scoped and evidence-linked. A target behavior may
affect runtime strategy without being allowed to overwrite authoritative
PatientScenario facts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TargetMemoryEntry:
    """One evidence-backed behavior or bug associated with the target agent."""

    memory_id: str
    status: str
    summary: str
    evidence_call: int
    scope: str
    exploration_action: str
    match_all: tuple[str, ...] = ()


TARGET_MEMORY: tuple[TargetMemoryEntry, ...] = (
    TargetMemoryEntry(
        memory_id="BUG-001",
        status="verified_bug",
        summary=(
            "Declining the optional demo patient profile caused the remote "
            "call to terminate before scheduling."
        ),
        evidence_call=6,
        scope="demo_profile_offer",
        exploration_action="accept_to_preserve_coverage",
        match_all=("demo patient profile",),
    ),
    TargetMemoryEntry(
        memory_id="TB-002",
        status="observed_behavior",
        summary=(
            "During demo-profile creation the target assigned July 4, 2000 "
            "as a demo date of birth without requesting the caller's real DOB."
        ),
        evidence_call=7,
        scope="demo_profile_creation",
        exploration_action="tolerate_and_continue",
        match_all=("date of birth", "demo purposes"),
    ),
)


def target_memory_context() -> tuple[dict[str, object], ...]:
    """Return JSON-safe target memory for semantic context and observability."""
    return tuple(asdict(entry) for entry in TARGET_MEMORY)


def matching_target_memory(
    agent_turn: str,
) -> tuple[TargetMemoryEntry, ...]:
    """Retrieve scoped memories whose literal trigger evidence is present."""
    normalized = " ".join(agent_turn.casefold().split())

    return tuple(
        entry
        for entry in TARGET_MEMORY
        if entry.match_all
        and all(
            phrase.casefold() in normalized
            for phrase in entry.match_all
        )
    )


def should_tolerate_target_conflict(
    agent_turn: str,
) -> bool:
    """Return whether exploration should tolerate a known target behavior."""
    return any(
        entry.exploration_action == "tolerate_and_continue"
        for entry in matching_target_memory(agent_turn)
    )
