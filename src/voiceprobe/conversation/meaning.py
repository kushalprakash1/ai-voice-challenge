"""Structured meaning extracted from tested-agent conversation turns.

The interpreter extracts what the tested agent communicated. Comparison
against authoritative patient truth happens separately in deterministic
Python code.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from voiceprobe.conversation.state import FactKey


class ResponseExpectation(StrEnum):
    """General form of response the tested agent is soliciting."""

    NONE = "none"
    YES_NO = "yes_no"
    FACT = "fact"
    CHOICE = "choice"
    ACKNOWLEDGEMENT = "acknowledgement"
    FREEFORM = "freeform"


class WorkflowRelation(StrEnum):
    """How agreeing with a workflow request relates to the call objective."""

    NONE = "none"
    ADVANCES_OBJECTIVE = "advances_objective"
    OPPOSES_OBJECTIVE = "opposes_objective"
    UNCERTAIN = "uncertain"


class QuestionKind(StrEnum):
    """What sort of response-triggering question or request the agent made."""

    NONE = "none"
    WORKFLOW_PERMISSION = "workflow_permission"
    PATIENT_ATTRIBUTE = "patient_attribute"
    OTHER = "other"


class WorkflowDirection(StrEnum):
    """Direction of an agent workflow action, independent of patient facts."""

    NONE = "none"
    CONTINUE = "continue"
    STOP = "stop"
    UNKNOWN = "unknown"


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

    response_expectation: ResponseExpectation = Field(
        default=ResponseExpectation.NONE,
        description=(
            "General form of reply the tested voice agent is currently "
            "soliciting. This supplements, rather than replaces, the "
            "structured patient-fact and scheduling fields."
        ),
    )
    workflow_relation: WorkflowRelation = Field(
        default=WorkflowRelation.NONE,
        description=(
            "Whether agreeing with a workflow request would advance or oppose "
            "the supplied conversation objective. This is not itself the "
            "patient's answer."
        ),
    )
    question_kind: QuestionKind = Field(
        default=QuestionKind.NONE,
        description=(
            "General category of the current question. workflow_permission "
            "means the agent asks whether it may perform, continue, or stop "
            "a workflow action. patient_attribute means the agent asks whether "
            "the patient has or is something, including attributes outside the "
            "known patient-fact ontology."
        ),
    )
    workflow_direction: WorkflowDirection = Field(
        default=WorkflowDirection.NONE,
        description=(
            "For workflow_permission questions, whether the requested agent "
            "action continues/proceeds with the workflow or stops/cancels it. "
            "Use none for non-workflow questions."
        ),
    )
    topic: str | None = Field(
        default=None,
        description=(
            "Short neutral description of what the tested agent's current "
            "question or request is about. Do not infer patient facts."
        ),
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
    conversation_end_requested: bool = Field(
        default=False,
        description=(
            "True only when the tested voice agent explicitly ends or closes "
            "the conversation, such as saying goodbye, bye, or otherwise "
            "clearly indicating that the call is over."
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

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = " ".join(value.split())

        if not normalized:
            return None

        if normalized.casefold() in {
            "none",
            "null",
            "n/a",
            "not applicable",
        }:
            return None

        return normalized

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
