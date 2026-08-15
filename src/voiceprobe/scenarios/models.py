"""Authoritative patient-scenario models for VoiceProbe.

Scenario data is ground truth. Language models may decide how to express
these facts conversationally, but they must not invent or overwrite them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientFacts(BaseModel):
    """Facts the simulated patient must remain consistent with."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    complaint: str = Field(min_length=1)
    duration: str = Field(min_length=1)

    date_of_birth: str | None = None
    insurance: str | None = None
    preferred_day: str | None = None
    preferred_time: str | None = None

    @field_validator(
        "name",
        "complaint",
        "duration",
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

    @field_validator("objective")
    @classmethod
    def strip_objective(cls, value: str) -> str:
        """Normalize the human-readable call objective."""
        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("Scenario objective cannot be blank.")

        return normalized
