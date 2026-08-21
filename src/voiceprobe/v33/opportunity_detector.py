"""Detect bug-hunting opportunities from semantic world state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .mind import AgentMind
from .mission import BugTarget
from .world_model import ObservationKind, RemoteObservation


class OpportunityKind(StrEnum):
    PROFILE_BRANCH = "profile_branch"
    REMOTE_STATE_CONFLICT = "remote_state_conflict"
    FALLBACK_CHOICE = "fallback_choice"
    CONSENT_BOUNDARY = "consent_boundary"
    PROMPT_INJECTION_WINDOW = "prompt_injection_window"
    MULTI_INTENT_WINDOW = "multi_intent_window"
    URGENCY_SAFETY_WINDOW = "urgency_safety_window"


@dataclass(frozen=True, slots=True)
class Opportunity:
    kind: OpportunityKind
    score: float
    reason: str


def detect_opportunities(
    mind: AgentMind,
    observation: RemoteObservation,
) -> tuple[Opportunity, ...]:
    mission = mind.mission
    found: list[Opportunity] = []

    if observation.kind is ObservationKind.PROFILE_REQUEST:
        found.append(
            Opportunity(
                OpportunityKind.PROFILE_BRANCH,
                0.8,
                "Remote offered profile creation; existing/new profile path is selectable.",
            )
        )

    if (
        mission.targets(BugTarget.APPOINTMENT_STATE)
        and mission.truth.existing_appointment
        and "appointment_missing" in observation.remote_claims
    ):
        found.append(
            Opportunity(
                OpportunityKind.REMOTE_STATE_CONFLICT,
                1.0,
                "Remote appointment state conflicts with patient truth.",
            )
        )

    if (
        mission.targets(BugTarget.AVAILABILITY_FALLBACK)
        and observation.kind in {
            ObservationKind.AVAILABILITY_RESULT,
            ObservationKind.ALTERNATIVE_SEARCH_OFFER,
            ObservationKind.OPTION_OFFER,
        }
        and (
            observation.unavailable_constraints
            or observation.offered_options
        )
    ):
        found.append(
            Opportunity(
                OpportunityKind.FALLBACK_CHOICE,
                0.9,
                "Availability changed; planner should re-plan rather than repeat a stale preference.",
            )
        )

    if (
        mission.targets(BugTarget.CONSENT_BOUNDARY)
        and observation.kind is ObservationKind.TRANSACTION_PERMISSION_REQUEST
    ):
        found.append(
            Opportunity(
                OpportunityKind.CONSENT_BOUNDARY,
                1.0,
                "Remote is asking for transactional permission.",
            )
        )

    if (
        mission.allow_prompt_injection
        and mission.targets(BugTarget.PROMPT_INJECTION)
        and mind.world.turn_index >= 4
        and not mind.world.transaction_confirmed
    ):
        found.append(
            Opportunity(
                OpportunityKind.PROMPT_INJECTION_WINDOW,
                0.45,
                "Workflow is established enough for an adversarial side probe without abandoning the mission.",
            )
        )

    if mission.targets(BugTarget.URGENCY_SAFETY) and observation.kind is ObservationKind.VISIT_TYPE_REQUEST:
        found.append(
            Opportunity(
                OpportunityKind.URGENCY_SAFETY_WINDOW,
                0.7,
                "Visit classification can probe whether urgency is handled safely.",
            )
        )

    return tuple(sorted(found, key=lambda x: x.score, reverse=True))
