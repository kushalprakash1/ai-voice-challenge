"""Structured meaning extracted from tested-agent conversation turns.

The interpreter extracts what the tested agent communicated. Comparison
against authoritative patient truth happens separately in deterministic
Python code.
"""

from __future__ import annotations

from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from voiceprobe.conversation.state import FactKey


class FactAssertion(BaseModel):
    """One patient fact explicitly stated by the tested agent."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    fact: FactKey = Field(
        description=("Patient fact being stated by the tested voice agent.")
    )
    value: str = Field(
        description=(
            "Value actually stated by the tested voice agent. "
            "Do not replace it with patient ground truth."
        )
    )

    @field_validator("value")
    @classmethod
    def normalize_value(cls, value: str) -> str:
        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("Fact assertion value cannot be blank.")

        return normalized


class AppointmentOffer(BaseModel):
    """Concrete scheduling slot offered by the tested agent."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    day: str | None
    time: str | None

    @field_validator("day", "time")
    @classmethod
    def normalize_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = " ".join(value.split())

        return normalized or None

    @model_validator(mode="after")
    def require_slot_detail(self) -> Self:
        if self.day is None and self.time is None:
            raise ValueError("Appointment offer requires a day or time.")

        return self


class TurnMeaning(BaseModel):
    """Semantic interpretation of one tested-agent turn."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    requested_facts: tuple[FactKey, ...] = Field(
        default=(),
        description=(
            "Patient facts the tested voice agent asks the patient to "
            "provide, verify, confirm, or repeat. Fact ontology: "
            "name means the patient's name or identity; "
            "complaint means symptoms, body problem, reason for visit, "
            "reason for calling, or what brought the patient in; "
            "duration means how long the problem has existed or when it "
            "started; date_of_birth means DOB or birthday; "
            "insurance means insurance, coverage, carrier, or insurer; "
            "preferred_day means desired appointment day or date; "
            "preferred_time means desired appointment time or daypart."
        ),
    )
    stated_facts: tuple[FactAssertion, ...] = Field(
        default=(),
        description=(
            "Patient facts for which the tested voice agent itself "
            "states, assumes, summarizes, or proposes a specific value. "
            "Do not add a stated fact when the agent merely asks for a "
            "fact without supplying a candidate value."
        ),
    )

    appointment_offer: AppointmentOffer | None = Field(
        default=None,
        description=(
            "Appointment day or time being offered by the tested voice "
            "agent. Null when no slot is being offered."
        ),
    )

    booking_confirmed: bool = Field(
        default=False,
        description=(
            "True only when the tested voice agent explicitly says an "
            "appointment has been booked, scheduled, or confirmed."
        ),
    )
    requests_repetition: bool = Field(
        default=False,
        description=(
            "True when the tested voice agent asks the patient to repeat "
            "because something was not heard or understood."
        ),
    )
    unclear: bool = Field(
        default=False,
        description=(
            "True only when the voice agent's utterance itself cannot "
            "be interpreted reliably."
        ),
    )

    @field_validator("appointment_offer", mode="before")
    @classmethod
    def normalize_empty_appointment_offer(cls, value: object) -> object:
        """Treat an empty structured offer as no appointment offer."""
        if value is None:
            return None

        if isinstance(value, dict):
            day = value.get("day")
            time = value.get("time")

            day_empty = day is None or (isinstance(day, str) and not day.strip())
            time_empty = time is None or (isinstance(time, str) and not time.strip())

            if day_empty and time_empty:
                return None

        return value

    @model_validator(mode="after")
    def reject_duplicates(self) -> Self:
        if len(set(self.requested_facts)) != len(self.requested_facts):
            raise ValueError("requested_facts cannot contain duplicates.")

        asserted_keys = [assertion.fact for assertion in self.stated_facts]

        if len(set(asserted_keys)) != len(asserted_keys):
            raise ValueError("stated_facts cannot contain duplicate fact keys.")

        return self
