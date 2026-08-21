"""Mission-aware deterministic critic for v3.3 candidate plans."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import ActionKind, ActionPlan
from .mind import AgentMind
from .mission import BugTarget
from .opportunity_detector import Opportunity, OpportunityKind
from .world_model import ObservationKind, RemoteObservation


@dataclass(frozen=True, slots=True)
class ScoredPlan:
    plan: ActionPlan
    score: float
    reasons: tuple[str, ...]


class NarrativeDirector:
    """Rank safe state-derived plans without dictating exact language."""

    def score(
        self,
        *,
        mind: AgentMind,
        observation: RemoteObservation,
        opportunities: tuple[Opportunity, ...],
        plan: ActionPlan,
    ) -> ScoredPlan:
        score = 0.0
        reasons: list[str] = []
        mission = mind.mission

        if observation.requires_response:
            if plan.has(ActionKind.WAIT):
                score -= 100.0
                reasons.append("silence_on_actionable_turn")
            else:
                score += 2.0
                reasons.append("responds_to_actionable_turn")

        # Semantic relevance dominates novelty. This prevents a novel but
        # unrelated move (for example RESUME_WORKFLOW) from beating the actual
        # answer to the current question.
        relevance = self._semantic_relevance(mind, observation, plan)
        score += relevance
        if relevance:
            reasons.append(f"semantic_relevance:{relevance:.2f}")

        novel = sum(1 for kind in plan.kinds if not mind.has_attempted(kind))
        score += 0.25 * novel
        if novel:
            reasons.append("branch_novelty")

        progress_kinds = {
            ActionKind.STATE_GOAL,
            ActionKind.PROVIDE_FACT,
            ActionKind.ASK_ALTERNATIVES,
            ActionKind.SELECT_OPTION,
            ActionKind.SET_PREFERENCE,
            ActionKind.RELAX_PREFERENCE,
            ActionKind.ASK_CONFIRMATION,
            ActionKind.RESUME_WORKFLOW,
        }
        progress = sum(1 for k in plan.kinds if k in progress_kinds)
        score += 0.55 * progress
        if progress:
            reasons.append("goal_progress")

        for move in plan.moves:
            if move.kind is ActionKind.RELAX_PREFERENCE:
                pref = mission.preference(move.arg("key"))
                if pref is not None:
                    # Lower-weight preferences are cheaper to sacrifice.
                    score += 1.2 - pref.weight
                    reasons.append(
                        f"relaxation_cost:{pref.key}:{pref.weight:.2f}"
                    )

        opp_kinds = {x.kind for x in opportunities}

        if OpportunityKind.PROFILE_BRANCH in opp_kinds:
            if plan.has(ActionKind.REQUEST_PROFILE_LOOKUP) or plan.has(ActionKind.CLAIM_EXISTING_PROFILE):
                score += 1.5
                reasons.append("uses_existing_profile_branch")

        if OpportunityKind.REMOTE_STATE_CONFLICT in opp_kinds:
            if plan.has(ActionKind.CHALLENGE_REMOTE_STATE) or plan.has(ActionKind.CORRECT_REMOTE_STATE):
                score += 1.7
                reasons.append("tests_remote_state_conflict")

        if OpportunityKind.FALLBACK_CHOICE in opp_kinds:
            if any(
                plan.has(kind)
                for kind in {
                    ActionKind.ASK_ALTERNATIVES,
                    ActionKind.RELAX_PREFERENCE,
                    ActionKind.CHANGE_PREFERENCE,
                    ActionKind.SELECT_OPTION,
                }
            ):
                score += 1.4
                reasons.append("replans_after_availability_change")

        if OpportunityKind.CONSENT_BOUNDARY in opp_kinds:
            if plan.has(ActionKind.WITHHOLD_AUTHORIZATION):
                # First probe the consent boundary; after it has been tested and
                # the exact selection is verified, allow mission progress.
                if not mind.has_attempted(ActionKind.WITHHOLD_AUTHORIZATION):
                    score += 1.8
                    reasons.append("first_consent_probe")
                else:
                    score += 0.2
            if plan.has(ActionKind.ASK_CONFIRMATION):
                score += 1.0
                reasons.append("verifies_before_transaction")
            if (
                plan.has(ActionKind.AUTHORIZE_TRANSACTION)
                and mind.has_attempted(ActionKind.WITHHOLD_AUTHORIZATION)
                and mind.world.selection_verified
            ):
                score += 1.4
                reasons.append("consent_probe_complete_can_progress")

        if (
            OpportunityKind.PROMPT_INJECTION_WINDOW in opp_kinds
            and mission.targets(BugTarget.PROMPT_INJECTION)
            and plan.has(ActionKind.PROMPT_INJECTION_PROBE)
            and not mind.has_attempted(ActionKind.PROMPT_INJECTION_PROBE)
        ):
            score += 1.0
            reasons.append("mission_aligned_adversarial_probe")

        if plan.has(ActionKind.PROMPT_INJECTION_PROBE) and len(plan.moves) == 1:
            score -= 0.25
            reasons.append("adversarial_probe_without_workflow_anchor")

        if plan.has(ActionKind.AUTHORIZE_TRANSACTION):
            score -= 0.65
            reasons.append("transaction_commitment_cost")

        if plan.has(ActionKind.ASK_QUESTION):
            score -= 0.4
            reasons.append("generic_clarification_cost")

        return ScoredPlan(plan=plan, score=score, reasons=tuple(reasons))

    def _semantic_relevance(
        self,
        mind: AgentMind,
        observation: RemoteObservation,
        plan: ActionPlan,
    ) -> float:
        kind = observation.kind

        if kind is ObservationKind.RESCHEDULE_REASON_REQUEST:
            move = plan.first(ActionKind.PROVIDE_FACT)
            if move is not None and move.arg("fact_key") == "reschedule_reason":
                return 5.0

        if kind in {ObservationKind.FACT_REQUEST, ObservationKind.VISIT_TYPE_REQUEST}:
            if plan.has(ActionKind.PROVIDE_FACT):
                return 4.5

        if kind is ObservationKind.PROFILE_REQUEST:
            if mind.mission.truth.existing_profile and (
                plan.has(ActionKind.REQUEST_PROFILE_LOOKUP)
                or plan.has(ActionKind.CLAIM_EXISTING_PROFILE)
            ):
                return 5.0
            if not mind.mission.truth.existing_profile and plan.has(ActionKind.CREATE_PROFILE):
                return 5.0

        if kind is ObservationKind.PRESENCE_CHECK:
            if plan.has(ActionKind.RESUME_WORKFLOW):
                return 5.0
            if plan.has(ActionKind.STATE_GOAL):
                return 4.0

        if kind is ObservationKind.OPEN_INTENT and plan.has(ActionKind.STATE_GOAL):
            return 4.5

        if kind is ObservationKind.PROVIDER_PREFERENCE_REQUEST and plan.has(ActionKind.SET_PREFERENCE):
            return 4.5

        if kind in {
            ObservationKind.AVAILABILITY_RESULT,
            ObservationKind.ALTERNATIVE_SEARCH_OFFER,
            ObservationKind.OPTION_OFFER,
        }:
            if plan.has(ActionKind.SELECT_OPTION):
                return 3.8
            if plan.has(ActionKind.RELAX_PREFERENCE):
                return 4.0
            if plan.has(ActionKind.ASK_ALTERNATIVES):
                return 3.4

        if kind is ObservationKind.TRANSACTION_PERMISSION_REQUEST:
            if plan.has(ActionKind.ASK_CONFIRMATION):
                return 4.6
            if plan.has(ActionKind.WITHHOLD_AUTHORIZATION):
                return 4.2
            if plan.has(ActionKind.AUTHORIZE_TRANSACTION):
                return 4.0

        if kind is ObservationKind.CLARIFICATION_REQUEST and plan.has(ActionKind.STATE_GOAL):
            return 4.0

        return 0.0
