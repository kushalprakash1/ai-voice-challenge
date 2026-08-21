"""Generate legal strategic actions from semantic world state.

This module is intentionally language-agnostic. It never matches whole clinic
sentences. The LLM interprets the clinic turn into RemoteObservation; Python
then derives what moves are logically available from patient truth, mission,
preferences, and transaction state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .actions import ActionKind, ActionMove, ActionPlan
from .mind import AgentMind
from .world_model import ObservationKind, RemoteObservation


_DAY_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
}
_TIME_BUCKETS = {"morning", "afternoon", "evening"}
_TIME_PATTERN = re.compile(r"\b(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?)\b", re.I)


@dataclass(slots=True)
class StrategicActionGenerator:
    """Derive candidate actions without asking the model to author plans."""

    def generate(
        self,
        *,
        mind: AgentMind,
        observation: RemoteObservation,
    ) -> tuple[ActionPlan, ...]:
        plans: list[ActionPlan] = []

        def add(*moves: ActionMove, rationale: str) -> None:
            plans.append(ActionPlan(moves=tuple(moves), rationale=rationale))

        if not observation.requires_response:
            add(ActionMove(ActionKind.WAIT), rationale="semantic:no_response_needed")
            return tuple(plans)

        truth = mind.mission.truth
        world = mind.world

        # Grounded information requests.
        if observation.requested_fact and truth.fact(observation.requested_fact) is not None:
            add(
                ActionMove(
                    ActionKind.PROVIDE_FACT,
                    {"fact_key": observation.requested_fact},
                ),
                rationale="semantic:grounded_fact",
            )

        if observation.kind is ObservationKind.RESCHEDULE_REASON_REQUEST:
            add(
                ActionMove(
                    ActionKind.PROVIDE_FACT,
                    {"fact_key": "reschedule_reason"},
                ),
                rationale="semantic:reschedule_reason",
            )

        if observation.kind is ObservationKind.PROFILE_REQUEST:
            if truth.existing_profile:
                add(
                    ActionMove(ActionKind.REQUEST_PROFILE_LOOKUP),
                    rationale="strategy:existing_profile_lookup",
                )
                add(
                    ActionMove(ActionKind.CLAIM_EXISTING_PROFILE),
                    rationale="strategy:existing_profile_claim",
                )
            else:
                add(
                    ActionMove(ActionKind.CREATE_PROFILE),
                    rationale="strategy:new_profile",
                )

        if observation.kind is ObservationKind.OPEN_INTENT:
            add(ActionMove(ActionKind.STATE_GOAL), rationale="strategy:state_goal")

        if observation.kind is ObservationKind.VISIT_TYPE_REQUEST:
            add(
                ActionMove(ActionKind.PROVIDE_FACT, {"fact_key": "visit_type"}),
                rationale="semantic:visit_type",
            )

        if observation.kind is ObservationKind.PROVIDER_PREFERENCE_REQUEST:
            if mind.mission.preference("provider") is not None:
                add(
                    ActionMove(ActionKind.SET_PREFERENCE, {"key": "provider"}),
                    rationale="strategy:provider_preference",
                )

        if observation.kind is ObservationKind.PROVIDER_NAME_REQUEST:
            # The default mission has no authoritative named-provider fact.
            # Asking for options is safer than inventing a clinician.
            add(
                ActionMove(ActionKind.ASK_ALTERNATIVES),
                rationale="strategy:no_grounded_provider_name",
            )

        if observation.kind in {
            ObservationKind.AVAILABILITY_RESULT,
            ObservationKind.ALTERNATIVE_SEARCH_OFFER,
            ObservationKind.OPTION_OFFER,
        }:
            self._add_availability_candidates(
                mind=mind,
                observation=observation,
                add=add,
            )

        # Remote state contradiction is treated as an opportunity, not as a
        # reason to overwrite Python-owned patient truth.
        if observation.remote_claims:
            if (
                truth.existing_appointment
                and "appointment_missing" in observation.remote_claims
            ):
                add(
                    ActionMove(ActionKind.CHALLENGE_REMOTE_STATE),
                    rationale="strategy:appointment_state_conflict",
                )
            if truth.existing_profile and "profile_missing" in observation.remote_claims:
                add(
                    ActionMove(ActionKind.REQUEST_PROFILE_LOOKUP),
                    rationale="strategy:profile_state_conflict",
                )

        if observation.kind is ObservationKind.TRANSACTION_PERMISSION_REQUEST:
            # Consent is a strategic choice, not a language-model free-for-all.
            add(
                ActionMove(ActionKind.WITHHOLD_AUTHORIZATION),
                rationale="strategy:withhold_consent_probe",
            )
            if world.selected_option and world.selection_verified:
                add(
                    ActionMove(ActionKind.AUTHORIZE_TRANSACTION),
                    rationale="strategy:verified_transaction_can_close",
                )
            else:
                add(
                    ActionMove(ActionKind.WITHHOLD_AUTHORIZATION),
                    ActionMove(ActionKind.ASK_CONFIRMATION),
                    rationale="strategy:verify_before_commit",
                )

        if observation.kind is ObservationKind.PRESENCE_CHECK:
            add(
                ActionMove(ActionKind.RESUME_WORKFLOW),
                rationale="strategy:presence_recovery",
            )
            add(
                ActionMove(ActionKind.STATE_GOAL),
                rationale="strategy:presence_restate_goal",
            )

        if observation.kind is ObservationKind.CLARIFICATION_REQUEST:
            add(
                ActionMove(ActionKind.STATE_GOAL),
                rationale="strategy:clarify_by_restate",
            )

        # Controlled narrative-driving probe. This is generated from mission
        # state/opportunity coverage, never from an exact clinic phrase.
        if (
            mind.mission.allow_prompt_injection
            and world.turn_index >= 4
            and not world.transaction_confirmed
            and not mind.has_attempted(ActionKind.PROMPT_INJECTION_PROBE)
        ):
            add(
                ActionMove(ActionKind.PROMPT_INJECTION_PROBE),
                ActionMove(ActionKind.RESUME_WORKFLOW),
                rationale="strategy:mission_adversarial_probe",
            )

        # Generic non-silent escape hatch. This is intentionally low value in
        # NarrativeDirector and should lose whenever a specific semantic move
        # exists.
        add(
            ActionMove(ActionKind.ASK_QUESTION),
            rationale="strategy:generic_clarification",
        )

        return self._dedupe(plans)

    def _add_availability_candidates(self, *, mind, observation, add) -> None:
        offered = observation.offered_options

        # Concrete appointment slots may be considered without granting consent.
        if observation.kind is ObservationKind.OPTION_OFFER:
            for option in offered[:4]:
                if self._looks_like_concrete_slot(option):
                    add(
                        ActionMove(ActionKind.SELECT_OPTION, {"option": option}),
                        rationale="strategy:consider_offered_slot",
                    )

        # Proposed fallback axes and failed axes are different semantics.
        # For an alternative-search question, only what the clinic proposes to
        # vary may directly create a relaxation candidate. Failed constraints
        # update world state but must not hijack the next action. For a bare
        # availability result, failed constraints are the only evidence we have
        # for proactive replanning. Concrete option offers are interpreted from
        # the offered slots themselves.
        if observation.kind is ObservationKind.ALTERNATIVE_SEARCH_OFFER:
            raw_relevant = set(observation.search_constraints)
        elif observation.kind is ObservationKind.AVAILABILITY_RESULT:
            raw_relevant = set(observation.unavailable_constraints)
        else:
            raw_relevant = set(observation.search_constraints)

        raw_relevant.discard("none")
        has_combination = "combination" in raw_relevant
        raw_relevant.discard("combination")

        # Backward-compatible ontology alias only; no raw clinic text involved.
        if "date_range" in raw_relevant:
            raw_relevant.discard("date_range")
            if mind.mission.preference("day") is not None:
                raw_relevant.add("day")

        relevant = {
            key
            for key in raw_relevant
            if mind.mission.preference(key) is not None
        }
        relevant.update(self._infer_relaxations_from_offers(mind, offered))

        if observation.kind is ObservationKind.AVAILABILITY_RESULT and not relevant and not has_combination:
            # A completed no-availability report should still drive the
            # narrative forward even if the model cannot isolate one failed axis.
            has_combination = True

        if has_combination and not relevant:
            relevant.update(
                pref.key for pref in mind.mission.preferences if pref.relaxable
            )

        for pref in mind.mission.preferences:
            if pref.relaxable and pref.key in relevant:
                add(
                    ActionMove(ActionKind.RELAX_PREFERENCE, {"key": pref.key}),
                    rationale="strategy:relevant_constraint_relaxation",
                )

        add(
            ActionMove(ActionKind.ASK_ALTERNATIVES),
            rationale="strategy:request_closest_alternatives",
        )

    def _infer_relaxations_from_offers(
        self,
        mind: AgentMind,
        offered: tuple[str, ...],
    ) -> set[str]:
        relevant: set[str] = set()
        day_pref = mind.mission.preference("day")
        time_pref = mind.mission.preference("time_of_day")

        for option in offered:
            normalized = " ".join(option.casefold().split())

            if day_pref is not None:
                desired_day = day_pref.value.casefold()
                option_days = {d for d in _DAY_WORDS if d in normalized}
                if option_days and desired_day not in option_days:
                    relevant.add("day")

            if time_pref is not None:
                desired_time = time_pref.value.casefold()
                option_buckets = {t for t in _TIME_BUCKETS if t in normalized}
                if option_buckets and desired_time not in option_buckets:
                    relevant.add("time_of_day")

        return relevant

    @staticmethod
    def _looks_like_concrete_slot(option: str) -> bool:
        normalized = option.casefold()
        return bool(
            _TIME_PATTERN.search(option)
            or any(day in normalized for day in _DAY_WORDS)
        )

    @staticmethod
    def _dedupe(plans: list[ActionPlan]) -> tuple[ActionPlan, ...]:
        seen: set[tuple[tuple[str, tuple[tuple[str, str], ...]], ...]] = set()
        result: list[ActionPlan] = []
        for plan in plans:
            signature = tuple(
                (
                    move.kind.value,
                    tuple(sorted(move.arguments.items())),
                )
                for move in plan.moves
            )
            if signature in seen:
                continue
            seen.add(signature)
            result.append(plan)
        return tuple(result)
