"""Conversation-quality invariants for StageLab."""

from __future__ import annotations

from dataclasses import dataclass, field

from voiceprobe.v33.actions import ActionKind
from voiceprobe.v33.planner import PlannerDecision

from .simulated_clock import RealtimeBudget


@dataclass(slots=True)
class StageViolations:
    items: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.items.append(message)

    @property
    def ok(self) -> bool:
        return not self.items

    def raise_if_any(self) -> None:
        if self.items:
            raise AssertionError("StageLab violations:\n- " + "\n- ".join(self.items))


def check_decision(
    decision: PlannerDecision,
    *,
    estimated_audio_start_seconds: float,
    budget: RealtimeBudget,
    violations: StageViolations,
) -> None:
    observation = decision.observation
    plan = decision.selected.plan

    if observation.requires_response and plan.has(ActionKind.WAIT):
        violations.add("actionable question produced WAIT")

    if observation.requires_response and not decision.spoken_text.strip():
        violations.add("actionable question produced empty patient speech")

    if estimated_audio_start_seconds > budget.first_audio_deadline_seconds:
        violations.add(
            f"first audio estimated at {estimated_audio_start_seconds:.3f}s "
            f"> {budget.first_audio_deadline_seconds:.3f}s deadline"
        )
