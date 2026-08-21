"""Conversation-burst coalescing for VoiceProbe v3.

The old implementation queued every finalized remote utterance. That preserved
speech but allowed acknowledgements/status messages to become stale work. V3
keeps the audio/transcript evidence while selecting the latest actionable turn
from a burst for dialogue policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .fast_policy import RoutineSchedulingPolicy
from .models import DecisionKind, PolicyDecision


_SUBORDINATE_EXAMPLE_PREFIXES = (
    "for example",
    "for instance",
    "such as",
    "e.g.",
    "e.g ",
)


def _is_subordinate_example(turn: str) -> bool:
    normalized = " ".join(turn.casefold().split())
    return normalized.startswith(_SUBORDINATE_EXAMPLE_PREFIXES)


@dataclass(frozen=True, slots=True)
class CoalescedTurn:
    source_turns: tuple[str, ...]
    actionable_turn: str | None
    decision: PolicyDecision
    discarded_non_actionable: tuple[str, ...]


class ConversationBurstCoalescer:
    def __init__(
        self,
        policy: RoutineSchedulingPolicy | None = None,
    ) -> None:
        self.policy = policy or RoutineSchedulingPolicy()

    def coalesce(self, turns: Iterable[str]) -> CoalescedTurn:
        source = tuple(turn for turn in turns if turn.strip())

        if not source:
            return CoalescedTurn(
                source_turns=(),
                actionable_turn=None,
                decision=PolicyDecision(
                    DecisionKind.WAIT,
                    reason="empty_burst",
                ),
                discarded_non_actionable=(),
            )

        evaluations = [
            (turn, self.policy.decide(turn))
            for turn in source
        ]

        actionable = [
            (turn, decision)
            for turn, decision in evaluations
            if decision.kind not in {
                DecisionKind.WAIT,
                DecisionKind.HOLD,
                DecisionKind.FALLBACK,
            }
        ]

        if actionable:
            turn, decision = actionable[-1]

            # A separately finalized illustrative tail elaborates the direct
            # question before it; it does not open a new workflow step. Keep
            # the earlier question as owner while still evaluating every turn
            # in order so independent state transitions remain intact.
            if _is_subordinate_example(turn) and len(actionable) >= 2:
                turn, decision = actionable[-2]
            discarded = tuple(
                prior_turn
                for prior_turn, prior_decision in evaluations
                if prior_turn != turn
                and prior_decision.kind == DecisionKind.WAIT
            )
            return CoalescedTurn(
                source_turns=source,
                actionable_turn=turn,
                decision=decision,
                discarded_non_actionable=discarded,
            )

        # HOLD wins over WAIT/FALLBACK because an incomplete remote clause must
        # not be passed to a model merely because there is no other action yet.
        for turn, decision in reversed(evaluations):
            if decision.kind == DecisionKind.HOLD:
                return CoalescedTurn(
                    source_turns=source,
                    actionable_turn=None,
                    decision=decision,
                    discarded_non_actionable=tuple(
                        item for item in source if item != turn
                    ),
                )

        # If nothing actionable exists, prefer a deterministic WAIT. A novel
        # complete turn remains available for the future LLM fallback layer.
        for turn, decision in reversed(evaluations):
            if decision.kind == DecisionKind.FALLBACK:
                return CoalescedTurn(
                    source_turns=source,
                    actionable_turn=turn,
                    decision=decision,
                    discarded_non_actionable=tuple(
                        item for item in source if item != turn
                    ),
                )

        return CoalescedTurn(
            source_turns=source,
            actionable_turn=None,
            decision=PolicyDecision(
                DecisionKind.WAIT,
                reason="burst_contains_only_non_actionable_turns",
            ),
            discarded_non_actionable=source,
        )
