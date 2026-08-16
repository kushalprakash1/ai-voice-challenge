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


class WorkflowRequirement(StrEnum):
    """How strongly the remote agent says a workflow step is required."""

    OPTIONAL = "optional"
    REQUIRED = "required"
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

    # Choose from concrete appointment-slot alternatives.
    CHOOSE_OPTION = "choose_option"

    # Choose among non-booking conversational/search alternatives.
    CHOOSE_PRESENTED_CHOICE = "choose_presented_choice"

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


class WorkflowProposal(BaseModel):
    """A workflow or sub-workflow proposed by the remote agent.

    This describes what the remote agent proposed.

    It does NOT decide whether the caller should accept it.
    Relevance to the caller's objective belongs to the planner.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    kind: WorkflowKind

    # Concise source-grounded description of the proposed operation.
    # Examples:
    #   "create a demo patient profile"
    #   "verify identity"
    #   "complete intake paperwork"
    description: str = Field(
        min_length=1,
    )

    requirement: WorkflowRequirement = (
        WorkflowRequirement.UNKNOWN
    )


class AgentFactAssertion(BaseModel):
    """One caller-related fact asserted by the remote agent.

    This records what the remote side claimed.

    It does NOT mean the claim is true.

    A later grounding/policy layer compares the assertion against
    authoritative PatientWorldModel truth.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    fact: RequestedFact

    # Preserve the remote agent's understood value.
    value: str = Field(
        min_length=1,
    )


class ChoiceKind(StrEnum):
    """Semantic category for a non-booking choice presented by the agent."""

    SEARCH_AVAILABILITY = "search_availability"
    WORKFLOW = "workflow"
    OTHER = "other"


class PresentedChoice(BaseModel):
    """One explicit non-booking alternative presented by the remote agent."""

    model_config = ConfigDict(
        extra="forbid",
    )

    label: str = Field(
        min_length=1,
    )
    kind: ChoiceKind

    day: str | None = None
    date_text: str | None = None
    time: str | None = None
    daypart: str | None = None
    provider: str | None = None
    appointment_type: str | None = None

    @model_validator(mode="after")
    def normalize_label(
        self,
    ) -> Self:
        label = " ".join(
            self.label.split()
        )

        if not label:
            raise ValueError(
                "PresentedChoice.label cannot be blank."
            )

        self.label = label
        return self


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

    # Caller-related facts stated or asserted by the REMOTE agent.
    #
    # These are observations, not trusted patient truth.
    stated_facts: list[AgentFactAssertion] = Field(
        default_factory=list,
    )

    # Optional workflow/sub-workflow the remote side is proposing.
    #
    # This remains separate from the primary workflow label because a
    # scheduling conversation can temporarily propose profile setup,
    # identity verification, paperwork, or another supporting workflow.
    proposed_workflow: WorkflowProposal | None = None

    appointment_options: list[SlotOption] = Field(
        default_factory=list,
    )

    # Explicit alternatives that are not concrete appointment slots.
    presented_choices: list[PresentedChoice] = Field(
        default_factory=list,
    )

    # A slot the remote side says is actually booked/confirmed.
    #
    # This is NOT an appointment offer and NOT a caller-profile fact.
    confirmed_appointment: SlotOption | None = None

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

        if (
            self.requested_action
            is RequestedAction.CHOOSE_PRESENTED_CHOICE
        ):
            if not self.response_required:
                raise ValueError(
                    "CHOOSE_PRESENTED_CHOICE must require a caller response."
                )

            if not self.presented_choices:
                raise ValueError(
                    "CHOOSE_PRESENTED_CHOICE requires presented choices."
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

        if (
            self.confirmed_appointment is not None
            and not self.booking_confirmed
        ):
            raise ValueError(
                "confirmed_appointment requires booking_confirmed=true."
            )

        if (
            self.booking_confirmed
            and self.confirmed_appointment is None
        ):
            raise ValueError(
                "booking_confirmed=true requires confirmed_appointment."
            )

        return self
