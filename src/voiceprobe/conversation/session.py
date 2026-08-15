"""Stateful orchestration for one simulated-patient conversation.

The session connects semantic interpretation, deterministic grounding,
PatientBrain reasoning, natural verbalization, conversation state, and
appointment-objective state.

No language model is allowed to mutate scenario truth or directly mark
the scheduling objective complete.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
    PatientBrain,
)
from voiceprobe.agents.probes import ProbeProgress, apply_probe_policy
from voiceprobe.conversation.exploration_policy import (
    apply_exploration_policy,
)
from voiceprobe.conversation.grounding import (
    GroundedTurnMeaning,
    ground_turn_meaning,
)
from voiceprobe.conversation.goal_policy import (
    GoalContext,
    apply_goal_policy,
)
from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    TurnMeaning,
)
from voiceprobe.conversation.normalization import (
    normalize_turn_meaning,
    recover_asr_booking_confirmation,
)
from voiceprobe.conversation.objective import (
    AppointmentProgress,
    record_booking_confirmed,
    record_offer_accepted,
    record_preferences_shared,
    record_slot_offer,
)
from voiceprobe.conversation.scheduling import time_matches_preference
from voiceprobe.conversation.state import (
    ActionKind,
    PatientAction,
    PatientState,
    apply_patient_action,
    build_initial_state,
    record_agent_turn,
)
from voiceprobe.scenarios.models import PatientScenario


_EXPLORATION_MODE_ENV = "VOICEPROBE_EXPLORATION_MODE"


def _exploration_mode_from_environment() -> bool:
    """Read an explicit one-call exploration feature flag.

    Only 0 and 1 are accepted so a typo cannot silently switch production
    conversation behavior.
    """
    value = os.environ.get(_EXPLORATION_MODE_ENV)

    if value is None or value == "" or value == "0":
        return False

    if value == "1":
        return True

    raise ValueError(
        f"{_EXPLORATION_MODE_ENV} must be exactly '0' or '1'."
    )


class ConversationInterpreter(Protocol):
    """Semantic interpreter required by PatientSession."""

    def interpret(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> TurnMeaning:
        """Extract semantic meaning from one tested-agent turn."""
        ...


@runtime_checkable
class PrefetchingConversationInterpreter(Protocol):
    """Optional early semantic-computation capability."""

    def prefetch(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> bool:
        """Start interpreting a provisional complete turn."""
        ...

    def invalidate_prefetch(self) -> None:
        """Prevent a provisional result from being consumed."""
        ...


class NaturalVerbalizer(Protocol):
    """Speech realization component required by PatientSession."""

    def verbalize(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        decision: CommunicationDecision,
    ) -> str:
        """Realize one PatientBrain decision as patient speech."""
        ...


@dataclass(frozen=True, slots=True)
class SessionTurnTimings:
    """Measured latency of the reasoning pipeline."""

    interpreter_seconds: float
    decision_seconds: float
    verbalizer_seconds: float
    state_update_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            self.interpreter_seconds
            + self.decision_seconds
            + self.verbalizer_seconds
            + self.state_update_seconds
        )


@dataclass(frozen=True, slots=True)
class SessionTurnResult:
    """Complete observable result of one conversation turn."""

    agent_turn: str
    meaning: TurnMeaning
    grounded: GroundedTurnMeaning
    decision: CommunicationDecision
    patient_text: str
    state: PatientState
    progress: AppointmentProgress
    timings: SessionTurnTimings


class PatientSession:
    """Own persistent state for one simulated patient call."""

    def __init__(
        self,
        *,
        scenario: PatientScenario,
        interpreter: ConversationInterpreter,
        verbalizer: NaturalVerbalizer,
        brain: PatientBrain | None = None,
    ) -> None:
        self._scenario = scenario
        self._interpreter = interpreter
        self._verbalizer = verbalizer
        self._brain = brain or PatientBrain()

        # Normal mode remains the default. Exploration must be explicitly
        # enabled for a single process with VOICEPROBE_EXPLORATION_MODE=1.
        self._exploration_mode = _exploration_mode_from_environment()

        self._state = build_initial_state(scenario)
        self._progress = AppointmentProgress()
        self._probe_progress = ProbeProgress()

        # Persistent workflow focus for elliptical references.
        self._goal_context = GoalContext()

        # Narrow conversational context for elliptical scheduling replies.
        # This is populated only after PatientBrain has determined that a
        # partial offer is compatible with patient truth.
        self._pending_offer: AppointmentOffer | None = None

    @property
    def state(self) -> PatientState:
        """Current immutable conversation state."""
        return self._state

    @property
    def progress(self) -> AppointmentProgress:
        """Current immutable appointment progress."""
        return self._progress

    @property
    def goal_context(self) -> GoalContext:
        """Persistent workflow focus for the scheduling mission."""
        return self._goal_context

    @property
    def exploration_mode(self) -> bool:
        """Whether this call explicitly enabled cooperative exploration."""
        return self._exploration_mode

    def prefetch_agent_turn(
        self,
        agent_turn: str,
    ) -> bool:
        """Begin optional interpretation before endpoint confirmation."""
        if not agent_turn.strip():
            return False

        interpreter = self._interpreter

        if not isinstance(
            interpreter,
            PrefetchingConversationInterpreter,
        ):
            return False

        return interpreter.prefetch(
            scenario=self._scenario,
            state=self._state,
            agent_turn=agent_turn,
        )

    def invalidate_prefetch(self) -> None:
        """Invalidate any provisional semantic interpretation."""
        interpreter = self._interpreter

        if isinstance(
            interpreter,
            PrefetchingConversationInterpreter,
        ):
            interpreter.invalidate_prefetch()

    def handle_agent_turn(
        self,
        agent_turn: str,
    ) -> SessionTurnResult:
        """Process one tested-agent utterance atomically."""
        if not agent_turn.strip():
            raise ValueError("Agent turn cannot be blank.")

        # Interpretation deliberately sees the state *before* the current
        # receptionist utterance is recorded. The interpreter receives the
        # latest turn separately, so recording first would duplicate it in
        # recent-history context.
        pre_turn_state = self._state
        pre_turn_progress = self._progress
        pre_turn_probe_progress = self._probe_progress
        pre_turn_goal_context = self._goal_context

        interpreter_started = perf_counter()

        raw_meaning = self._interpreter.interpret(
            scenario=self._scenario,
            state=pre_turn_state,
            agent_turn=agent_turn,
        )

        interpreter_seconds = perf_counter() - interpreter_started

        decision_started = perf_counter()

        meaning = normalize_turn_meaning(
            raw_meaning,
            agent_turn=agent_turn,
            pending_offer=self._pending_offer,
        )

        offer = meaning.appointment_offer

        accepted_offer_matches = (
            pre_turn_progress.offer_accepted
            and offer is not None
            and self._is_same_recorded_slot(
                progress=pre_turn_progress,
                day=offer.day,
                time=offer.time,
            )
        )

        meaning = recover_asr_booking_confirmation(
            meaning,
            agent_turn=agent_turn,
            accepted_offer_matches=accepted_offer_matches,
        )

        grounded = ground_turn_meaning(
            scenario=self._scenario,
            meaning=meaning,
        )

        decision = self._brain.decide(
            scenario=self._scenario,
            grounded=grounded,
            progress=pre_turn_progress,
        )

        prior_agent_turn_count = sum(
            message.speaker.value == "agent" for message in pre_turn_state.messages
        )

        decision, next_probe_progress = apply_probe_policy(
            scenario=self._scenario,
            appointment=pre_turn_progress,
            probe_progress=pre_turn_probe_progress,
            prior_agent_turn_count=prior_agent_turn_count,
            base_decision=decision,
            booking_confirmed_this_turn=meaning.booking_confirmed,
        )

        # Final deterministic mission guard.
        #
        # Semantic interpretation may describe what was said and the
        # brain may propose a response, but neither may redirect the
        # patient into an unrelated workflow.
        if self._exploration_mode:
            # Exploration intentionally bypasses only the final scheduling
            # mission guard. Interpreter, grounding, PatientBrain, probes,
            # patient truth, appointment progress, and state validation all
            # remain active.
            decision = apply_exploration_policy(
                scenario=self._scenario,
                grounded=grounded,
                progress=pre_turn_progress,
                agent_turn=agent_turn,
                base_decision=decision,
            )

            # GoalContext belongs to normal scheduling mode. Preserve it
            # unchanged rather than manufacturing side-workflow state.
            next_goal_context = pre_turn_goal_context
        else:
            decision, next_goal_context = apply_goal_policy(
                scenario=self._scenario,
                grounded=grounded,
                progress=pre_turn_progress,
                agent_turn=agent_turn,
                context=pre_turn_goal_context,
                base_decision=decision,
            )

        state_with_agent = record_agent_turn(
            pre_turn_state,
            agent_turn,
        )

        decision_seconds = perf_counter() - decision_started

        if decision.kind is CommunicationKind.WAIT:
            state_update_started = perf_counter()

            # A non-actionable agent turn still belongs in conversation
            # history, but the simulated patient intentionally says nothing.
            # Appointment, probe, and pending-offer state remain unchanged.
            self._state = state_with_agent
            self._progress = pre_turn_progress
            self._probe_progress = next_probe_progress
            self._goal_context = next_goal_context

            return SessionTurnResult(
                agent_turn=agent_turn,
                meaning=meaning,
                grounded=grounded,
                decision=decision,
                patient_text="",
                state=state_with_agent,
                progress=pre_turn_progress,
                timings=SessionTurnTimings(
                    interpreter_seconds=interpreter_seconds,
                    decision_seconds=decision_seconds,
                    verbalizer_seconds=0.0,
                    state_update_seconds=(perf_counter() - state_update_started),
                ),
            )

        verbalizer_started = perf_counter()

        patient_text = self._verbalizer.verbalize(
            scenario=self._scenario,
            state=state_with_agent,
            decision=decision,
        )

        verbalizer_seconds = perf_counter() - verbalizer_started

        state_update_started = perf_counter()

        next_progress = self._advance_progress(
            progress=pre_turn_progress,
            meaning=meaning,
            decision=decision,
        )

        action = self._build_patient_action(
            decision=decision,
            grounded=grounded,
            patient_text=patient_text,
            progress=next_progress,
        )

        next_state = apply_patient_action(
            state_with_agent,
            self._scenario,
            action,
        )

        next_pending_offer = self._next_pending_offer(
            current=self._pending_offer,
            meaning=meaning,
            decision=decision,
        )

        # Commit only after the entire turn succeeds. An interpreter,
        # verbalizer, or validation failure therefore cannot leave the
        # session in a partially advanced state.
        self._state = next_state
        self._progress = next_progress
        self._probe_progress = next_probe_progress
        self._pending_offer = next_pending_offer
        self._goal_context = next_goal_context

        return SessionTurnResult(
            agent_turn=agent_turn,
            meaning=meaning,
            grounded=grounded,
            decision=decision,
            patient_text=patient_text,
            state=next_state,
            progress=next_progress,
            timings=SessionTurnTimings(
                interpreter_seconds=interpreter_seconds,
                decision_seconds=decision_seconds,
                verbalizer_seconds=verbalizer_seconds,
                state_update_seconds=(perf_counter() - state_update_started),
            ),
        )

    @staticmethod
    def _next_pending_offer(
        *,
        current: AppointmentOffer | None,
        meaning: TurnMeaning,
        decision: CommunicationDecision,
    ) -> AppointmentOffer | None:
        """Advance trusted elliptical scheduling context deterministically."""
        if decision.kind is CommunicationKind.ACCEPT_PARTIAL_OFFER:
            offer = meaning.appointment_offer

            if offer is None:
                raise RuntimeError(
                    "Partial-offer decision requires an appointment offer."
                )

            if (offer.day is None) == (offer.time is None):
                raise RuntimeError(
                    "Partial-offer decision requires exactly one slot detail."
                )

            return offer

        if decision.kind in {
            CommunicationKind.ACCEPT_OFFER,
            CommunicationKind.DECLINE_OFFER,
            CommunicationKind.ACKNOWLEDGE_COMPLETE,
            CommunicationKind.END_CONVERSATION,
        }:
            return None

        return current

    def _advance_progress(
        self,
        *,
        progress: AppointmentProgress,
        meaning: TurnMeaning,
        decision: CommunicationDecision,
    ) -> AppointmentProgress:
        next_progress = progress

        offer = meaning.appointment_offer

        offer_matches_recorded = offer is not None and self._is_same_recorded_slot(
            progress=next_progress,
            day=offer.day,
            time=offer.time,
        )

        # A contradictory *confirmation* must not erase the slot the
        # patient already accepted. Keep the accepted slot authoritative
        # until the receptionist either confirms it or makes a genuine
        # new offer that the patient accepts.
        conflicting_confirmation = (
            meaning.booking_confirmed
            and next_progress.offer_accepted
            and offer is not None
            and not offer_matches_recorded
        )

        confirmed_accepted_offer = (
            meaning.booking_confirmed
            and next_progress.offer_accepted
            and (offer is None or offer_matches_recorded)
        )

        if (
            offer is not None
            and not offer_matches_recorded
            and not conflicting_confirmation
        ):
            next_progress = record_slot_offer(
                next_progress,
                day=offer.day,
                time=offer.time,
            )

        communicated_facts = set(decision.facts_to_communicate)

        day_shared = "preferred_day" in communicated_facts
        time_shared = "preferred_time" in communicated_facts

        if day_shared or time_shared:
            next_progress = record_preferences_shared(
                next_progress,
                day_shared=day_shared,
                time_shared=time_shared,
            )

        if decision.kind is CommunicationKind.ACCEPT_OFFER:
            next_progress = record_offer_accepted(
                next_progress,
            )

        if confirmed_accepted_offer:
            next_progress = record_booking_confirmed(
                next_progress,
            )

        return next_progress

    @staticmethod
    def _is_same_recorded_slot(
        *,
        progress: AppointmentProgress,
        day: str | None,
        time: str | None,
    ) -> bool:
        """Avoid resetting acceptance when an agent repeats the same slot."""
        if not progress.has_offer:
            return False

        if day is not None:
            if progress.offered_day is None:
                return False

            normalized_existing_day = " ".join(progress.offered_day.casefold().split())
            normalized_new_day = " ".join(day.casefold().split())

            if normalized_existing_day != normalized_new_day:
                return False

        if time is not None:
            if progress.offered_time is None:
                return False

            forward_match = time_matches_preference(
                preferred=progress.offered_time,
                offered=time,
            )
            reverse_match = time_matches_preference(
                preferred=time,
                offered=progress.offered_time,
            )

            if not forward_match and not reverse_match:
                return False

        return True

    @staticmethod
    def _build_patient_action(
        *,
        decision: CommunicationDecision,
        grounded: GroundedTurnMeaning,
        patient_text: str,
        progress: AppointmentProgress,
    ) -> PatientAction:
        if decision.kind is CommunicationKind.CORRECT:
            if not grounded.conflicts:
                raise RuntimeError(
                    "PatientBrain requested a correction without "
                    "a grounded conflicting claim."
                )

            corrected_claim = "; ".join(
                (f"{conflict.fact}={conflict.heard_value!r}")
                for conflict in grounded.conflicts
            )

            return PatientAction(
                kind=ActionKind.CORRECT,
                response=patient_text,
                facts_used=decision.facts_to_communicate,
                corrected_claim=corrected_claim,
            )

        if decision.kind is CommunicationKind.CLARIFY:
            return PatientAction(
                kind=ActionKind.CLARIFY,
                response=patient_text,
                facts_used=decision.facts_to_communicate,
            )

        if decision.kind is CommunicationKind.ACKNOWLEDGE_COMPLETE:
            if not progress.objective_complete:
                raise RuntimeError(
                    "Completion acknowledgement was generated before "
                    "the deterministic appointment objective completed."
                )

            return PatientAction(
                kind=ActionKind.COMPLETE,
                response=patient_text,
                facts_used=decision.facts_to_communicate,
            )

        if (
            decision.kind is CommunicationKind.END_CONVERSATION
            and progress.objective_complete
        ):
            return PatientAction(
                kind=ActionKind.COMPLETE,
                response=patient_text,
                facts_used=decision.facts_to_communicate,
            )

        return PatientAction(
            kind=ActionKind.ANSWER,
            response=patient_text,
            facts_used=decision.facts_to_communicate,
        )
