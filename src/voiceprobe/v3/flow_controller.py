"""Flow-aware decision coordination for VoiceProbe v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .coalescer import ConversationBurstCoalescer
from .flow_state import (
    FlowSnapshot,
    SchedulingFlowTracker,
    extract_concrete_slot,
)
from .models import DecisionKind, PolicyDecision


@dataclass(frozen=True, slots=True)
class FlowDecision:
    """One conversational decision plus before/after flow snapshots."""

    source_turns: tuple[str, ...]
    actionable_turn: str | None
    decision: PolicyDecision
    before: FlowSnapshot
    after: FlowSnapshot


class SchedulingFlowController:
    """Combine burst coalescing, routine policy, and progress tracking."""

    def __init__(
        self,
        *,
        coalescer: ConversationBurstCoalescer | None = None,
        tracker: SchedulingFlowTracker | None = None,
    ) -> None:
        self._coalescer = coalescer or ConversationBurstCoalescer()
        self._tracker = tracker or SchedulingFlowTracker()

    @property
    def tracker(self) -> SchedulingFlowTracker:
        return self._tracker

    def decide_burst(
        self,
        turns: Iterable[str],
    ) -> FlowDecision:
        source = tuple(turn for turn in turns if turn.strip())
        before = self._tracker.snapshot()

        # Keep the deterministic policy synchronized with durable flow state.
        # The policy remains Friday-first until the remote scheduler explicitly
        # offers an alternate-day afternoon branch.
        if before.allow_earlier_week_afternoons:
            self._coalescer.policy.relax_day_constraint_for_afternoon()

        relaxation_prompt_seen = any(
            self._coalescer.policy.should_relax_day_constraint_for_afternoon(
                turn
            )
            for turn in source
        )

        if relaxation_prompt_seen:
            # Set policy state before coalescing so a concrete Mon-Thu PM slot
            # arriving later in the same Flux burst is evaluated correctly.
            self._coalescer.policy.relax_day_constraint_for_afternoon()

        # Remote confirmations are evidence even when the utterance itself does
        # not require a patient response.
        for turn in source:
            self._tracker.observe_remote_turn(turn)

        observed = self._tracker.snapshot()

        # An authoritative booking confirmation completes the mission.
        # Trailing small-talk or intake wording in the same stabilized
        # Flux burst must not generate another patient response.
        if observed.complete:
            return FlowDecision(
                source_turns=source,
                actionable_turn=None,
                decision=PolicyDecision(
                    DecisionKind.WAIT,
                    reason="booking_confirmation",
                ),
                before=before,
                after=observed,
            )

        coalesced = self._coalescer.coalesce(source)

        if relaxation_prompt_seen:
            self._tracker.relax_day_constraint_for_afternoon()

        self._tracker.apply_decision(coalesced.decision)

        if (
            coalesced.decision.reason
            == "compatible_concrete_slot_offered"
        ):
            slot_source = (
                coalesced.actionable_turn
                or " ".join(coalesced.source_turns)
            )
            slot_text = extract_concrete_slot(slot_source)

            if slot_text is None:
                raise ValueError(
                    "Compatible concrete-slot decision had no extractable slot"
                )

            self._tracker.record_slot_acceptance(slot_text)

        after = self._tracker.snapshot()

        return FlowDecision(
            source_turns=coalesced.source_turns,
            actionable_turn=coalesced.actionable_turn,
            decision=coalesced.decision,
            before=before,
            after=after,
        )
