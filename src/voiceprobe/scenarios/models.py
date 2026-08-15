"""Authoritative patient-scenario models for VoiceProbe.

Scenario data is ground truth. Language models may decide how to express
these facts conversationally, but they must not invent or overwrite them.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProbeKind(StrEnum):
    """Patient-driven experiment behavior that can be enacted deterministically."""

    REQUEST_AGENT_REPEAT_ONCE = "request_agent_repeat_once"
    VERIFY_BOOKING_BEFORE_END = "verify_booking_before_end"


class PatientFacts(BaseModel):
    """Facts the simulated patient must remain consistent with."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    complaint: str = Field(min_length=1)
    duration: str = Field(min_length=1)

    # Extensible caller-persona fields discovered from real target-agent
    # dialogue. These remain authoritative scenario truth, not model memory.
    first_name: str | None = None
    last_name: str | None = None
    patient_status: str | None = None
    visited_before: bool | None = None
    appointment_type: str | None = None
    provider_preference: str | None = None

    date_of_birth: str | None = None
    insurance: str | None = None
    preferred_day: str | None = None
    preferred_time: str | None = None

    @field_validator(
        "name",
        "complaint",
        "duration",
        "first_name",
        "last_name",
        "patient_status",
        "appointment_type",
        "provider_preference",
        "date_of_birth",
        "insurance",
        "preferred_day",
        "preferred_time",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        """Reject values that contain only whitespace."""
        if value is None:
            return None

        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("Scenario facts cannot be blank.")

        return normalized


class PatientScenario(BaseModel):
    """One autonomous call scenario."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    objective: str = Field(min_length=1)
    facts: PatientFacts
    test_targets: tuple[str, ...] = ()
    probes: tuple[ProbeKind, ...] = ()

    @field_validator("objective")
    @classmethod
    def strip_objective(cls, value: str) -> str:
        """Normalize the human-readable call objective."""
        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("Scenario objective cannot be blank.")

        return normalized

    @field_validator("probes")
    @classmethod
    def reject_duplicate_probes(
        cls,
        value: tuple[ProbeKind, ...],
    ) -> tuple[ProbeKind, ...]:
        """Keep experiment behavior deterministic and non-redundant."""
        if len(value) != len(set(value)):
            raise ValueError("Scenario probes cannot contain duplicates.")

        return value
