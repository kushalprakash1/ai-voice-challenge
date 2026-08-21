"""Persistent reasoning state for VoiceProbe v3.3."""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import ActionKind, ActionPlan
from .mission import TestMission
from .world_model import WorldState


@dataclass(slots=True)
class AgentMind:
    mission: TestMission
    world: WorldState = field(default_factory=WorldState)
    attempted_action_kinds: list[ActionKind] = field(default_factory=list)
    opportunity_history: list[str] = field(default_factory=list)
    strategic_notes: list[str] = field(default_factory=list)

    def record_action(self, plan: ActionPlan) -> None:
        self.attempted_action_kinds.extend(plan.kinds)
        self.world.apply_action(plan)

    def has_attempted(self, kind: ActionKind) -> bool:
        return kind in self.attempted_action_kinds
