"""Conversation-world state for VoiceProbe v3.3.

Patient truth, remote claims, and transaction state remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .actions import ActionKind, ActionPlan


class ObservationKind(StrEnum):
    GREETING = "greeting"
    PROFILE_REQUEST = "profile_request"
    IDENTITY_CONFIRMATION = "identity_confirmation"
    FACT_REQUEST = "fact_request"
    RESCHEDULE_REASON_REQUEST = "reschedule_reason_request"
    OPEN_INTENT = "open_intent"
    APPOINTMENT_STATE = "appointment_state"
    VISIT_TYPE_REQUEST = "visit_type_request"
    PROVIDER_PREFERENCE_REQUEST = "provider_preference_request"
    PROVIDER_NAME_REQUEST = "provider_name_request"
    AVAILABILITY_RESULT = "availability_result"
    ALTERNATIVE_SEARCH_OFFER = "alternative_search_offer"
    OPTION_OFFER = "option_offer"
    TRANSACTION_PERMISSION_REQUEST = "transaction_permission_request"
    TRANSACTION_CONFIRMED = "transaction_confirmed"
    PRESENCE_CHECK = "presence_check"
    ACKNOWLEDGEMENT = "acknowledgement"
    CLARIFICATION_REQUEST = "clarification_request"
    CAPABILITY_RESPONSE = "capability_response"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class RemoteObservation:
    kind: ObservationKind
    raw_text: str
    requires_response: bool
    requested_fact: str = ""
    offered_options: tuple[str, ...] = ()
    unavailable_constraints: tuple[str, ...] = ()
    remote_claims: tuple[str, ...] = ()
    selected_option: str = ""
    transaction_operation: str = ""
    search_constraints: tuple[str, ...] = ()


@dataclass(slots=True)
class WorldState:
    turn_index: int = 0
    profile_status: str = "unknown"
    remote_existing_appointment: str = "unknown"
    current_topic: str = ""
    offered_options: tuple[str, ...] = ()
    selected_option: str = ""
    selection_verified: bool = False
    unavailable_constraints: set[str] = field(default_factory=set)
    proposed_search_constraints: set[str] = field(default_factory=set)
    transaction_authorized: bool = False
    transaction_confirmed: bool = False
    last_observation: RemoteObservation | None = None
    remote_claims: list[str] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)

    def apply_observation(self, observation: RemoteObservation) -> None:
        self.turn_index += 1
        self.last_observation = observation
        self.current_topic = observation.kind.value
        self.history.append(("PGAI", observation.raw_text))

        if observation.offered_options:
            self.offered_options = observation.offered_options

        # selected_option means the remote side explicitly identified one
        # concrete current selection. That is what verifies selection state.
        if observation.selected_option:
            self.selected_option = observation.selected_option
            self.selection_verified = True

        self.unavailable_constraints.update(observation.unavailable_constraints)
        self.proposed_search_constraints.update(observation.search_constraints)
        self.remote_claims.extend(observation.remote_claims)

        if observation.kind is ObservationKind.TRANSACTION_CONFIRMED:
            self.transaction_confirmed = True

        for claim in observation.remote_claims:
            normalized = claim.casefold()
            if normalized == "profile_exists":
                self.profile_status = "exists"
            elif normalized == "profile_missing":
                self.profile_status = "missing"
            elif normalized == "appointment_exists":
                self.remote_existing_appointment = "exists"
            elif normalized == "appointment_missing":
                self.remote_existing_appointment = "missing"

    def apply_action(self, plan: ActionPlan) -> None:
        for move in plan.moves:
            if move.kind is ActionKind.SELECT_OPTION:
                option = move.arg("option")
                if option:
                    self.selected_option = option
                    self.selection_verified = False
            elif move.kind is ActionKind.ASK_CONFIRMATION:
                # Asking what is selected is not itself verification. Only a
                # later remote observation that explicitly identifies the
                # selected option may flip selection_verified to True.
                pass
            elif move.kind is ActionKind.WITHHOLD_AUTHORIZATION:
                self.transaction_authorized = False
            elif move.kind is ActionKind.REVOKE_AUTHORIZATION:
                self.transaction_authorized = False
            elif move.kind is ActionKind.AUTHORIZE_TRANSACTION:
                self.transaction_authorized = True
            elif move.kind is ActionKind.CREATE_PROFILE:
                self.profile_status = "creating"
            elif move.kind in {
                ActionKind.CLAIM_EXISTING_PROFILE,
                ActionKind.REQUEST_PROFILE_LOOKUP,
            }:
                self.profile_status = "claimed_existing"

        if plan.utterance.strip():
            self.history.append(("PATIENT", plan.utterance.strip()))
