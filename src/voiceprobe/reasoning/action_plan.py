"""Typed actions available to the generic autonomous caller planner."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from voiceprobe.reasoning.turn_frame import (
    RequestedFact,
)


class PatientActionKind(StrEnum):
    """Actions the reasoning core may propose.

    There is intentionally NO context-free "AGREE" action.

    A slot-selection turn must result in an explicit option selection,
    alternative request, or clarification.
    """

    WAIT = "wait"

    ANSWER_FACT = "answer_fact"

    GRANT_PERMISSION = "grant_permission"
    DECLINE_PERMISSION = "decline_permission"

    SELECT_OPTION = "select_option"
    REQUEST_ALTERNATIVE = "request_alternative"

    CONFIRM = "confirm"
    REJECT_CONFIRMATION = "reject_confirmation"

    CLARIFY = "clarify"

    VERIFY_BOOKING = "verify_booking"

    END_CONVERSATION = "end_conversation"


class ActionPlan(BaseModel):
    """One proposed semantic caller action."""

    model_config = ConfigDict(
        extra="forbid",
    )

    action: PatientActionKind

    # Zero-based index into TurnFrame.appointment_options.
    selected_option_index: int | None = Field(
        default=None,
        ge=0,
    )

    facts_to_answer: list[RequestedFact] = Field(
        default_factory=list,
    )

    # Short machine-readable explanation useful for traces and evaluation.
    reason_code: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_action_shape(
        self,
    ) -> Self:

        if (
            self.action
            is PatientActionKind.SELECT_OPTION
        ):
            if self.selected_option_index is None:
                raise ValueError(
                    "SELECT_OPTION requires selected_option_index."
                )
        elif self.selected_option_index is not None:
            raise ValueError(
                "Only SELECT_OPTION may set selected_option_index."
            )

        if (
            self.action
            is PatientActionKind.ANSWER_FACT
        ):
            if not self.facts_to_answer:
                raise ValueError(
                    "ANSWER_FACT requires facts_to_answer."
                )
        elif self.facts_to_answer:
            raise ValueError(
                "Only ANSWER_FACT may set facts_to_answer."
            )

        normalized_reason = " ".join(
            self.reason_code.split()
        )

        if not normalized_reason:
            raise ValueError(
                "reason_code cannot be blank."
            )

        self.reason_code = normalized_reason

        return self
