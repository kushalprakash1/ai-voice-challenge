"""Strict model-boundary schemas for VoiceProbe v3.2."""

from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class RiskLevel(StrEnum):
    LOW = "low"
    AUTHORITATIVE_FACT = "authoritative_fact"
    TRANSACTION = "transaction"
    UNCERTAIN = "uncertain"


class DialogueAction(StrEnum):
    ANSWER = "answer"
    ANSWER_FACT = "answer_fact"
    STATE_OBJECTIVE = "state_objective"
    WAIT = "wait"
    CLARIFY = "clarify"


class Grounding(StrEnum):
    LOW_RISK_CONVERSATIONAL = "low_risk_conversational"
    AUTHORITATIVE_FACT = "authoritative_fact"
    CURRENT_GOAL = "current_goal"
    NONE = "none"


class FactKey(StrEnum):
    NONE = "none"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    DOB = "dob"
    INSURANCE = "insurance"
    COMPLAINT = "complaint"
    SYMPTOM_DURATION = "symptom_duration"
    PREFERRED_DAY = "preferred_day"
    PREFERRED_TIME = "preferred_time"
    APPOINTMENT_TYPE = "appointment_type"
    PROVIDER_PREFERENCE = "provider_preference"


class ContextualProposal(BaseModel):
    """Only information the model is permitted to propose."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    meaning: str = Field(
        min_length=1,
        max_length=80,
    )

    risk: RiskLevel
    action: DialogueAction
    grounding: Grounding
    fact_key: FactKey = FactKey.NONE

    # This is used only for genuinely low-risk conversational answers.
    # Authoritative facts are always rendered by Python.
    response_text: str = Field(
        default="",
        max_length=160,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    @field_validator("meaning", "response_text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())


PROPOSAL_JSON_SCHEMA = ContextualProposal.model_json_schema()
