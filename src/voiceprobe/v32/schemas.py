"""Typed v3.2 semantic rewrite and planning schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TurnKind(StrEnum):
    QUESTION = "question"
    INSTRUCTION = "instruction"
    ACKNOWLEDGEMENT = "acknowledgement"
    STATUS = "status"
    TRANSACTION = "transaction"
    OTHER = "other"


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


REWRITE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "meaning",
        "turn_kind",
        "subject",
        "risk",
        "requires_response",
        "confidence",
    ],
    "properties": {
        "meaning": {"type": "string"},
        "turn_kind": {
            "type": "string",
            "enum": [x.value for x in TurnKind],
        },
        "subject": {"type": "string"},
        "risk": {
            "type": "string",
            "enum": [x.value for x in RiskLevel],
        },
        "requires_response": {"type": "boolean"},
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
}


ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action",
        "grounding",
        "fact_key",
        "answer_text",
        "proposed_state_change",
        "confidence",
    ],
    "properties": {
        "action": {
            "type": "string",
            "enum": [x.value for x in DialogueAction],
        },
        "grounding": {
            "type": "string",
            "enum": [x.value for x in Grounding],
        },
        "fact_key": {"type": "string"},
        "answer_text": {"type": "string"},
        "proposed_state_change": {"type": "boolean"},
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
}


@dataclass(frozen=True, slots=True)
class IntentRewrite:
    meaning: str
    turn_kind: TurnKind
    subject: str
    risk: RiskLevel
    requires_response: bool
    confidence: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IntentRewrite":
        return cls(
            meaning=str(payload["meaning"]).strip(),
            turn_kind=TurnKind(payload["turn_kind"]),
            subject=str(payload["subject"]).strip(),
            risk=RiskLevel(payload["risk"]),
            requires_response=bool(payload["requires_response"]),
            confidence=float(payload["confidence"]),
        )


@dataclass(frozen=True, slots=True)
class ActionProposal:
    action: DialogueAction
    grounding: Grounding
    fact_key: str
    answer_text: str
    proposed_state_change: bool
    confidence: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionProposal":
        return cls(
            action=DialogueAction(payload["action"]),
            grounding=Grounding(payload["grounding"]),
            fact_key=str(payload["fact_key"]).strip(),
            answer_text=str(payload["answer_text"]).strip(),
            proposed_state_change=bool(payload["proposed_state_change"]),
            confidence=float(payload["confidence"]),
        )
