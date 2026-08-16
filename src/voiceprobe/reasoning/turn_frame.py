"""Typed semantic representation of one remote-agent turn.

This layer describes what the remote agent said.

It must not decide whether the simulated caller likes an option,
whether an offer satisfies the caller's preferences, or what the
caller should do next. Those responsibilities belong to the planner
and deterministic constraint validator.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class SpeechAct(StrEnum):
    """Primary communicative function of the remote agent's turn."""

    GREETING = "greeting"
    INFORMATION = "information"
    STATUS = "status"
    QUESTION = "question"
    REQUEST = "request"
    OFFER = "offer"
    CONFIRMATION = "confirmation"
    GOODBYE = "goodbye"
    OTHER = "other"


class WorkflowKind(StrEnum):
    """High-level workflow currently being discussed."""

    SCHEDULING = "scheduling"
    PATIENT_INTAKE = "patient_intake"
    PROFILE_SETUP = "profile_setup"
    INSURANCE = "insurance"
    IDENTITY = "identity"
    OTHER = "other"
    UNKNOWN = "unknown"


class RequestedAction(StrEnum):
    """What the remote agent expects from the caller next."""

    NONE = "none"

    # Remote side is still speaking, searching, loading, or working.
    WAIT = "wait"

    # Remote side requests one or more factual values.
    ANSWER_FACT = "answer_fact"

    # Remote side asks what the caller wants / how it can help.
    STATE_OBJECTIVE = "state_objective"

    # Permission to perform an action such as checking availability.
    GRANT_PERMISSION = "grant_permission"

    # Choose from concrete alternatives.
    CHOOSE_OPTION = "choose_option"

    # Confirm or reject a proposition.
    CONFIRM = "confirm"

    # Stop/cancel the current workflow.
    CANCEL = "cancel"

    # Meaning itself is genuinely uncertain.
    CLARIFY = "clarify"


class RequestedFact(StrEnum):
    """Canonical caller facts understood across VoiceProbe scenarios."""

    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    FULL_NAME = "full_name"
    DATE_OF_BIRTH = "date_of_birth"

    INSURANCE = "insurance"

    COMPLAINT = "complaint"
    SYMPTOM_DURATION = "symptom_duration"

    PREFERRED_DAY = "preferred_day"
    PREFERRED_TIME = "preferred_time"
    PROVIDER_PREFERENCE = "provider_preference"
    APPOINTMENT_TYPE = "appointment_type"

    PATIENT_STATUS = "patient_status"
    VISITED_BEFORE = "visited_before"

    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    ADDRESS = "address"


class SlotOption(BaseModel):
    """One appointment option explicitly communicated by the agent."""

    model_config = ConfigDict(
        extra="forbid",
    )

    # These values may come from the latest utterance or from clear
    # conversational inheritance in recent REMOTE-AGENT history.
    #
    # They must never be inferred from patient preferences.
    day: str | None = None
    date_text: str | None = None
    time: str | None = None
    daypart: str | None = None
    provider: str | None = None
    appointment_type: str | None = None


class TurnFrame(BaseModel):
    """Structured understanding of one complete remote-agent turn."""

    model_config = ConfigDict(
        extra="forbid",
    )

    speech_act: SpeechAct
    workflow: WorkflowKind
    requested_action: RequestedAction

    response_required: bool

    requested_facts: list[RequestedFact] = Field(
        default_factory=list,
    )

    # Facts outside our common ontology remain representable without
    # weakening requested_facts into arbitrary free-form strings.
    other_requested_facts: list[str] = Field(
        default_factory=list,
    )

    appointment_options: list[SlotOption] = Field(
        default_factory=list,
    )

    booking_confirmed: bool = False
    conversation_end_requested: bool = False
    agent_is_still_working: bool = False

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_semantic_consistency(
        self,
    ) -> Self:
        """Reject internally contradictory semantic frames."""

        if self.requested_action is RequestedAction.WAIT:
            if self.response_required:
                raise ValueError(
                    "WAIT cannot require an immediate caller response."
                )

            if self.requested_facts:
                raise ValueError(
                    "WAIT cannot simultaneously request caller facts."
                )

            if self.other_requested_facts:
                raise ValueError(
                    "WAIT cannot simultaneously request caller facts."
                )

        if (
            self.requested_action
            is RequestedAction.ANSWER_FACT
        ):
            if not self.response_required:
                raise ValueError(
                    "ANSWER_FACT must require a caller response."
                )

            if (
                not self.requested_facts
                and not self.other_requested_facts
            ):
                raise ValueError(
                    "ANSWER_FACT requires at least one requested fact."
                )

        if (
            self.requested_action
            is RequestedAction.CHOOSE_OPTION
        ):
            if not self.response_required:
                raise ValueError(
                    "CHOOSE_OPTION must require a caller response."
                )

            if not self.appointment_options:
                raise ValueError(
                    "CHOOSE_OPTION requires concrete options."
                )

        if self.requested_action in {
            RequestedAction.GRANT_PERMISSION,
            RequestedAction.CONFIRM,
            RequestedAction.CANCEL,
        }:
            if not self.response_required:
                raise ValueError(
                    f"{self.requested_action.value} must require a response."
                )

        return self
