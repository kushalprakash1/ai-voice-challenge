"""Deterministic conversation state for the VoiceProbe patient agent.

The language model may propose conversational actions, but Python owns
the authoritative history, fact usage, corrections, and goal state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal

from voiceprobe.scenarios.models import PatientScenario

type FactKey = Literal[
    "name",
    "first_name",
    "last_name",
    "patient_status",
    "visited_before",
    "appointment_type",
    "provider_preference",
    "complaint",
    "duration",
    "date_of_birth",
    "insurance",
    "preferred_day",
    "preferred_time",
]


class Speaker(StrEnum):
    """Participant responsible for one conversation message."""

    AGENT = "agent"
    PATIENT = "patient"


class ActionKind(StrEnum):
    """High-level actions available to the simulated patient."""

    ANSWER = "answer"
    CORRECT = "correct"
    CLARIFY = "clarify"
    PROBE = "probe"
    COMPLETE = "complete"


def _normalize_nonblank(value: str, *, field_name: str) -> str:
    normalized = " ".join(value.split())

    if not normalized:
        raise ValueError(f"{field_name} cannot be blank.")

    return normalized


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """One finalized message in the conversation history."""

    speaker: Speaker
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _normalize_nonblank(
                self.text,
                field_name="message text",
            ),
        )


@dataclass(frozen=True, slots=True)
class CorrectionRecord:
    """A claim the simulated patient explicitly corrected."""

    claim: str
    response: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "claim",
            _normalize_nonblank(
                self.claim,
                field_name="corrected claim",
            ),
        )
        object.__setattr__(
            self,
            "response",
            _normalize_nonblank(
                self.response,
                field_name="correction response",
            ),
        )


@dataclass(frozen=True, slots=True)
class PatientAction:
    """Structured action proposed for the patient's next response."""

    kind: ActionKind
    response: str
    facts_used: tuple[FactKey, ...] = ()
    corrected_claim: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "response",
            _normalize_nonblank(
                self.response,
                field_name="patient response",
            ),
        )

        if len(set(self.facts_used)) != len(self.facts_used):
            raise ValueError("facts_used cannot contain duplicates.")

        if self.kind is ActionKind.CORRECT:
            if self.corrected_claim is None:
                raise ValueError("A correction action requires corrected_claim.")

            object.__setattr__(
                self,
                "corrected_claim",
                _normalize_nonblank(
                    self.corrected_claim,
                    field_name="corrected claim",
                ),
            )

        elif self.corrected_claim is not None:
            raise ValueError("corrected_claim is only valid for correction actions.")


@dataclass(frozen=True, slots=True)
class PatientState:
    """Authoritative evolving state for one patient call."""

    scenario_id: str
    messages: tuple[ConversationMessage, ...] = ()
    answered_facts: frozenset[FactKey] = frozenset()
    corrections: tuple[CorrectionRecord, ...] = ()
    objective_complete: bool = False

    @property
    def agent_turn_count(self) -> int:
        """Number of finalized turns received from the tested agent."""
        return sum(message.speaker is Speaker.AGENT for message in self.messages)

    @property
    def patient_turn_count(self) -> int:
        """Number of responses produced by the patient."""
        return sum(message.speaker is Speaker.PATIENT for message in self.messages)


def build_initial_state(scenario: PatientScenario) -> PatientState:
    """Create an empty state tied to one immutable scenario."""
    return PatientState(scenario_id=scenario.scenario_id)


def record_agent_turn(
    state: PatientState,
    text: str,
) -> PatientState:
    """Record one finalized turn spoken by the tested voice agent."""
    message = ConversationMessage(
        speaker=Speaker.AGENT,
        text=text,
    )

    return replace(
        state,
        messages=(*state.messages, message),
    )


def apply_patient_action(
    state: PatientState,
    scenario: PatientScenario,
    action: PatientAction,
) -> PatientState:
    """Validate and apply one proposed patient action."""

    if state.scenario_id != scenario.scenario_id:
        raise ValueError("PatientState does not belong to the supplied scenario.")

    for fact_key in action.facts_used:
        fact_value = getattr(
            scenario.facts,
            fact_key,
        )

        if fact_value is None:
            raise ValueError(
                f"Patient action attempted to use unavailable fact: {fact_key}"
            )

    message = ConversationMessage(
        speaker=Speaker.PATIENT,
        text=action.response,
    )

    corrections = state.corrections

    if action.kind is ActionKind.CORRECT:
        if action.corrected_claim is None:
            raise RuntimeError("Validated correction is missing corrected_claim.")

        corrections = (
            *corrections,
            CorrectionRecord(
                claim=action.corrected_claim,
                response=action.response,
            ),
        )

    return replace(
        state,
        messages=(*state.messages, message),
        answered_facts=state.answered_facts.union(action.facts_used),
        corrections=corrections,
        objective_complete=(
            state.objective_complete or action.kind is ActionKind.COMPLETE
        ),
    )
