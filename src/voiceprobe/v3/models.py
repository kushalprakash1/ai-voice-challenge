"""Core typed models for VoiceProbe v3 dialogue control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DecisionKind(StrEnum):
    WAIT = "wait"
    HOLD = "hold"
    ANSWER_FACT = "answer_fact"
    CREATE_PROFILE = "create_profile"
    ANSWER_APPOINTMENT_TYPE = "answer_appointment_type"
    ANSWER_VISIT_DETAILS = "answer_visit_details"
    ANSWER_COMPLAINT = "answer_complaint"
    ANSWER_PROVIDER_PREFERENCE = "answer_provider_preference"
    CONTEXTUAL_ANSWER = "contextual_answer"
    CORRECT_AND_STATE_OBJECTIVE = "correct_and_state_objective"
    GRANT_PERMISSION = "grant_permission"
    CHOOSE_SEARCH_BRANCH = "choose_search_branch"
    SEARCH_ALTERNATE_DAY_AFTERNOON = "search_alternate_day_afternoon"
    DECLINE_INCOMPATIBLE_OFFER = "decline_incompatible_offer"
    STATE_OBJECTIVE = "state_objective"
    CORRECT_FACT = "correct_fact"
    CLARIFY = "clarify"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class PatientFacts:
    first_name: str = "Chitragupta"
    last_name: str = "Subramnian Singh"
    dob: str = "April 12, 1998"
    insurance: str = "Blue Cross"
    complaint: str = "right shoulder pain"
    symptom_duration: str = "five days"
    preferred_day: str = "Friday"
    preferred_time: str = "afternoon"
    appointment_type: str = "new patient consultation"
    provider_preference: str = "first available"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    kind: DecisionKind
    text: str = ""
    reason: str = ""
    confidence: float = 1.0

    @property
    def requires_response(self) -> bool:
        return self.kind not in {
            DecisionKind.WAIT,
            DecisionKind.HOLD,
        }
