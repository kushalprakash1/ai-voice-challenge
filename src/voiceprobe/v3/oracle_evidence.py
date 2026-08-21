"""Small serializable evidence records for deterministic adversarial oracles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OracleEvidence:
    oracle_name: str
    status: str
    scenario: str
    field: str
    expected_value: object
    observed_value: object
    evidence_turns: tuple[str, ...]
    relevant_provenance: tuple[str, ...]
    reason: str
