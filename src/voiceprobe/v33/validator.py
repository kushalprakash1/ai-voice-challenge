"""Deterministic safety and consistency validator for v3.3 plans."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import ActionKind, ActionPlan
from .mind import AgentMind
from .mission import BugTarget
from .world_model import RemoteObservation


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...] = ()


class PlanValidator:
    def validate(
        self,
        *,
        mind: AgentMind,
        observation: RemoteObservation,
        plan: ActionPlan,
    ) -> ValidationResult:
        reasons: list[str] = []
        truth = mind.mission.truth
        world = mind.world

        if observation.requires_response and plan.has(ActionKind.WAIT):
            reasons.append("actionable_remote_turn_cannot_wait")

        for move in plan.moves:
            kind = move.kind

            if kind is ActionKind.PROVIDE_FACT:
                fact_key = move.arg("fact_key")
                if not fact_key or truth.fact(fact_key) is None:
                    reasons.append("unknown_or_ungrounded_fact_key")
                if "value" in move.arguments:
                    reasons.append("model_must_not_supply_authoritative_fact_value")

            elif kind is ActionKind.CREATE_PROFILE:
                if (
                    truth.existing_profile
                    and not mind.mission.targets(BugTarget.PROFILE_DUPLICATION)
                ):
                    reasons.append("cannot_create_new_profile_for_known_existing_patient")

            elif kind in {
                ActionKind.CLAIM_EXISTING_PROFILE,
                ActionKind.REQUEST_PROFILE_LOOKUP,
            }:
                if not truth.existing_profile:
                    reasons.append("cannot_claim_existing_profile_when_patient_is_new")

            elif kind is ActionKind.SELECT_OPTION:
                option = move.arg("option")
                if not option:
                    reasons.append("select_option_requires_option")
                elif world.offered_options and option not in world.offered_options:
                    reasons.append("selected_option_not_in_remote_offer")

            elif kind in {
                ActionKind.RELAX_PREFERENCE,
                ActionKind.CHANGE_PREFERENCE,
            }:
                key = move.arg("key")
                pref = mind.mission.preference(key)
                if pref is None:
                    reasons.append("unknown_preference_key")
                elif kind is ActionKind.RELAX_PREFERENCE and not pref.relaxable:
                    reasons.append("preference_is_not_relaxable")

            elif kind is ActionKind.AUTHORIZE_TRANSACTION:
                if not world.selected_option:
                    reasons.append("transaction_authorization_requires_selected_option")
                if mind.mission.require_explicit_transaction_authorization and not world.selection_verified:
                    reasons.append("transaction_authorization_requires_verified_selection")
                if world.transaction_confirmed:
                    reasons.append("transaction_already_confirmed")

            elif kind is ActionKind.PROMPT_INJECTION_PROBE:
                if not mind.mission.allow_prompt_injection:
                    reasons.append("prompt_injection_not_allowed_by_mission")

            elif kind is ActionKind.CORRECT_REMOTE_STATE:
                if not observation.remote_claims:
                    reasons.append("no_remote_claim_available_to_correct")

        return ValidationResult(
            valid=not reasons,
            reasons=tuple(reasons),
        )
