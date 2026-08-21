"""Autonomous v3.3 patient planning loop."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import ActionPlan
from .mind import AgentMind
from .narrative_director import NarrativeDirector, ScoredPlan
from .opportunity_detector import Opportunity, detect_opportunities
from .reasoner import Reasoner
from .validator import PlanValidator, ValidationResult
from .verbalizer import GroundedVerbalizer
from .world_model import RemoteObservation


@dataclass(frozen=True, slots=True)
class RejectedPlan:
    plan: ActionPlan
    validation: ValidationResult


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    observation: RemoteObservation
    opportunities: tuple[Opportunity, ...]
    selected: ScoredPlan
    rejected: tuple[RejectedPlan, ...]
    scored_candidates: tuple[ScoredPlan, ...]
    spoken_text: str


class V33Planner:
    def __init__(
        self,
        *,
        mind: AgentMind,
        reasoner: Reasoner,
        validator: PlanValidator | None = None,
        director: NarrativeDirector | None = None,
        verbalizer: GroundedVerbalizer | None = None,
    ) -> None:
        self.mind = mind
        self.reasoner = reasoner
        self.validator = validator or PlanValidator()
        self.director = director or NarrativeDirector()
        self.verbalizer = verbalizer or GroundedVerbalizer()

    async def decide(self, remote_turn: str) -> PlannerDecision:
        observation, candidates = await self.reasoner.propose(
            mind=self.mind,
            remote_turn=remote_turn,
        )

        # Remote observation updates the world before plans are evaluated.
        self.mind.world.apply_observation(observation)

        opportunities = detect_opportunities(self.mind, observation)
        for opportunity in opportunities:
            self.mind.opportunity_history.append(opportunity.kind.value)

        safe: list[ScoredPlan] = []
        rejected: list[RejectedPlan] = []

        for plan in candidates:
            validation = self.validator.validate(
                mind=self.mind,
                observation=observation,
                plan=plan,
            )
            if not validation.valid:
                rejected.append(RejectedPlan(plan, validation))
                continue

            safe.append(
                self.director.score(
                    mind=self.mind,
                    observation=observation,
                    opportunities=opportunities,
                    plan=plan,
                )
            )

        if not safe:
            reasons = [
                reason
                for item in rejected
                for reason in item.validation.reasons
            ]
            raise RuntimeError(
                "v3.3 reasoner produced no safe candidate plans: "
                + ", ".join(reasons)
            )

        safe.sort(key=lambda item: item.score, reverse=True)
        selected = safe[0]
        spoken_text = self.verbalizer.render(
            mind=self.mind,
            plan=selected.plan,
        )

        # Only validated/selected actions become durable patient behavior.
        committed = ActionPlan(
            moves=selected.plan.moves,
            rationale=selected.plan.rationale,
            utterance=spoken_text,
        )
        self.mind.record_action(committed)

        return PlannerDecision(
            observation=observation,
            opportunities=opportunities,
            selected=ScoredPlan(
                plan=committed,
                score=selected.score,
                reasons=selected.reasons,
            ),
            rejected=tuple(rejected),
            scored_candidates=tuple(safe),
            spoken_text=spoken_text,
        )
