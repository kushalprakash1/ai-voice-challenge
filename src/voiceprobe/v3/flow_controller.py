"""Flow-aware decision coordination for VoiceProbe v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .coalescer import CoalescedTurn, ConversationBurstCoalescer
from .flow_state import (
    FlowSnapshot,
    SchedulingFlowTracker,
    extract_concrete_slot,
)
from .models import DecisionKind, PolicyDecision
from dataclasses import replace


DecisionOverlay = Callable[
    [
        tuple[str, ...],
        str | None,
        PolicyDecision,
        FlowSnapshot,
    ],
    PolicyDecision,
]


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
        decision_overlay: DecisionOverlay | None = None,
        semantic_only: bool = False,
    ) -> None:
        self._coalescer = coalescer or ConversationBurstCoalescer()
        self._tracker = tracker or SchedulingFlowTracker()
        self._decision_overlay = decision_overlay
        self._semantic_only = semantic_only

    @property
    def tracker(self) -> SchedulingFlowTracker:
        return self._tracker

    def decide_burst(
        self,
        turns: Iterable[str],
    ) -> FlowDecision:
        source = tuple(turn for turn in turns if turn.strip())
        before = self._tracker.snapshot()

        if self._semantic_only:
            decision = PolicyDecision(
                DecisionKind.FALLBACK,
                reason="scenario_requires_whole_turn_semantics",
                confidence=0.0,
            )
            return FlowDecision(
                source_turns=source,
                actionable_turn=" ".join(source),
                decision=decision,
                before=before,
                after=before,
            )

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
            completion_decision = PolicyDecision(
                DecisionKind.WAIT,
                reason="booking_confirmation",
            )

            # Allow an active adversarial persona to observe that PGAI
            # completed a transaction early. A well-behaved persona overlay
            # leaves this authoritative WAIT unchanged.
            if self._decision_overlay is not None:
                completion_decision = self._decision_overlay(
                    source,
                    None,
                    completion_decision,
                    observed,
                )

            return FlowDecision(
                source_turns=source,
                actionable_turn=None,
                decision=completion_decision,
                before=before,
                after=observed,
            )

        pre_coalesced_decision = None
        if self._decision_overlay is not None:
            pre_shared = getattr(
                self._decision_overlay,
                "decide_before_shared_policy",
                None,
            )
            if callable(pre_shared):
                pre_coalesced_decision = pre_shared(source, observed)

        if pre_coalesced_decision is not None:
            coalesced = CoalescedTurn(
                source_turns=source,
                actionable_turn=" ".join(source),
                decision=pre_coalesced_decision,
                discarded_non_actionable=(),
            )
        else:
            coalesced = self._coalescer.coalesce(source)

        if relaxation_prompt_seen:
            self._tracker.relax_day_constraint_for_afternoon()

        effective_decision = coalesced.decision

        # The deterministic coalescer remains first authority. But a complete
        # clinic question must not disappear as WAIT before semantic fallback.
        if (
            effective_decision.kind == DecisionKind.WAIT
            and effective_decision.reason
            == "burst_contains_only_non_actionable_turns"
            and "?" in " ".join(coalesced.source_turns)
        ):
            effective_decision = PolicyDecision(
                DecisionKind.FALLBACK,
                reason="complete_burst_requires_semantic_interpretation",
                confidence=0.0,
            )
            coalesced = replace(
                coalesced,
                actionable_turn=" ".join(coalesced.source_turns),
                decision=effective_decision,
            )

        # Persona testing happens before the durable flow tracker records what
        # the patient communicated. Default None preserves baseline behavior.
        if self._decision_overlay is not None and pre_coalesced_decision is None:
            effective_decision = self._decision_overlay(
                coalesced.source_turns,
                coalesced.actionable_turn,
                effective_decision,
                self._tracker.snapshot(),
            )

        # A concrete slot has exactly one owner. Extract it once from the
        # remote offer, render the patient response from that exact value, and
        # persist the same value into flow state.
        slot_text: str | None = None

        if (
            effective_decision.reason
            == "compatible_concrete_slot_offered"
        ):
            slot_source = (
                coalesced.actionable_turn
                or " ".join(coalesced.source_turns)
            )
            slot_text = extract_concrete_slot(slot_source)

            # A policy overlay may accept a previously grounded offer after a
            # verification turn. Its response still enters the same stable
            # offered-slot observation and acceptance machinery below.
            if slot_text is None:
                slot_text = extract_concrete_slot(effective_decision.text)

            if slot_text is None and self._decision_overlay is not None:
                retained_slot = getattr(
                    self._decision_overlay,
                    "grounded_slot_for_acceptance",
                    None,
                )
                if callable(retained_slot):
                    slot_text = retained_slot()

            if slot_text is None:
                raise ValueError(
                    "Compatible concrete-slot decision had no extractable slot"
                )

            grounded_text = effective_decision.text
            if grounded_text != "Yes, please book that appointment.":
                grounded_text = (
                    f"Yes, you can book it—the {slot_text} slot."
                    if "you can book it" in grounded_text.casefold()
                    else f"Yes, please book the {slot_text} slot."
                )
            effective_decision = replace(
                effective_decision,
                text=grounded_text,
            )
            self._tracker.observe_target_value(
                "offered_slot",
                slot_text,
                evidence=slot_source,
            )

        self._tracker.apply_decision(effective_decision)

        if slot_text is not None:
            self._tracker.record_slot_acceptance(slot_text)

        after = self._tracker.snapshot()

        return FlowDecision(
            source_turns=coalesced.source_turns,
            actionable_turn=coalesced.actionable_turn,
            decision=effective_decision,
            before=before,
            after=after,
        )
