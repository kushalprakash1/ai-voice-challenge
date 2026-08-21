"""Production-session adapter for Reasoning Core v2.

This module deliberately sits at the PatientSession boundary.

It does NOT modify telephony, AudioSocket, ASR, TTS, or the legacy
PatientSession implementation.

Reasoning v2 owns:

    remote speech
        -> structured TurnFrame
        -> deterministic fact grounding
        -> validated patient planning
        -> deterministic verbalization

The adapter exposes the small compatibility surface required by
autonomous_phone.py and the Asterisk assessment adapter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

import httpx

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.objective import (
    AppointmentProgress,
    record_booking_confirmed,
    record_offer_accepted,
    record_preferences_shared,
    record_slot_offer,
)
from voiceprobe.conversation.scheduling import (
    time_matches_preference,
)
from voiceprobe.conversation.session import (
    SessionTurnTimings,
)
from voiceprobe.conversation.state import (
    FactKey,
)
from voiceprobe.reasoning.action_plan import (
    ActionPlan,
    PatientActionKind,
)
from voiceprobe.reasoning.action_verbalizer import (
    GenericActionVerbalizer,
)
from voiceprobe.reasoning.fact_grounding import (
    ground_fact_assertions,
)
from voiceprobe.reasoning.planner import (
    QwenPatientPlanner,
)
from voiceprobe.reasoning.semantic_reasoner import (
    StructuredTurnReasoner,
)
from voiceprobe.reasoning.semanticlab_turn_bridge import (
    build_reasoning_v2_semantic_interpreter,
    reasoning_v2_edge_model_from_environment,
)
from voiceprobe.reasoning.turn_frame import (
    SlotOption,
    TurnFrame,
)
from voiceprobe.reasoning.world_model import (
    PatientWorldModel,
    build_world_model,
)
from voiceprobe.scenarios.models import (
    PatientScenario,
)


_REASONING_V2_ENV = "VOICEPROBE_REASONING_V2"


def reasoning_v2_enabled_from_environment() -> bool:
    """Read the production reasoning feature flag.

    Default is deliberately OFF.

    Only exact 0/1 values are accepted so a typo cannot silently switch
    the reasoning implementation used by an authorized call.
    """

    value = os.environ.get(
        _REASONING_V2_ENV
    )

    if (
        value is None
        or value == ""
        or value == "0"
    ):
        return False

    if value == "1":
        return True

    raise ValueError(
        f"{_REASONING_V2_ENV} must be exactly '0' or '1'."
    )


@dataclass(
    frozen=True,
    slots=True,
)
class ReasoningV2SessionTurnResult:
    """Compatibility result consumed by autonomous_phone.process_turns."""

    agent_turn: str

    # This is intentionally JSON-safe diagnostic evidence rather than
    # pretending TurnFrame is the legacy TurnMeaning representation.
    meaning: dict[str, object]

    # Not currently consumed by the media loop, but retained as useful
    # structured evidence for callers/tests.
    grounded: object

    decision: CommunicationDecision
    patient_text: str
    progress: AppointmentProgress
    timings: SessionTurnTimings


_LEGACY_FACT_KEYS = {
    "name",
    "first_name",
    "last_name",
    "patient_status",
    "visited_before",
    "appointment_type",
    "complaint",
    "duration",
    "date_of_birth",
    "insurance",
    "preferred_day",
    "preferred_time",
}


_FACT_ALIASES = {
    "full_name": "name",
    "symptom_duration": "duration",
}


_ACTION_TO_COMMUNICATION = {
    "wait": CommunicationKind.WAIT,

    "answer_fact": CommunicationKind.ANSWER,
    "state_objective": CommunicationKind.ANSWER,

    "grant_permission": CommunicationKind.AGREE,
    "confirm": CommunicationKind.AGREE,

    "decline_permission": CommunicationKind.DECLINE_WORKFLOW,

    "select_option": CommunicationKind.ACCEPT_OFFER,

    # General/search branch selection is not a concrete slot acceptance.
    "select_presented_choice": CommunicationKind.ANSWER,

    "request_alternative": CommunicationKind.DECLINE_OFFER,
    "reject_confirmation": CommunicationKind.DECLINE_OFFER,

    "clarify": CommunicationKind.CLARIFY,
    "verify_booking": CommunicationKind.VERIFY_BOOKING,

    "end_conversation": CommunicationKind.END_CONVERSATION,
}


def _normalize_text(
    value: str,
) -> str:
    return " ".join(
        value.casefold().split()
    )


def _days_match(
    left: str | None,
    right: str | None,
) -> bool:
    if left is None:
        return True

    if right is None:
        return False

    return (
        _normalize_text(left)
        == _normalize_text(right)
    )


def _times_match(
    left: str | None,
    right: str | None,
) -> bool:
    if left is None:
        return True

    if right is None:
        return False

    return (
        time_matches_preference(
            preferred=left,
            offered=right,
        )
        or time_matches_preference(
            preferred=right,
            offered=left,
        )
    )


def _confirmation_matches_progress(
    *,
    progress: AppointmentProgress,
    confirmed: SlotOption,
) -> bool:
    """Require a confirmation to agree with the accepted slot."""

    if not (
        progress.has_offer
        and progress.offer_accepted
    ):
        return False

    return (
        _days_match(
            progress.offered_day,
            confirmed.day,
        )
        and _times_match(
            progress.offered_time,
            confirmed.time,
        )
    )


def _legacy_fact_key(
    value: str,
) -> FactKey | None:
    normalized = _FACT_ALIASES.get(
        value,
        value,
    )

    if normalized not in _LEGACY_FACT_KEYS:
        return None

    return cast(
        FactKey,
        normalized,
    )


def _facts_communicated(
    *,
    world: PatientWorldModel,
    turn: TurnFrame,
    plan: ActionPlan,
    grounding: object,
) -> tuple[FactKey, ...]:
    """Build diagnostic legacy fact metadata from v2 output."""

    values: list[str] = [
        item.value
        for item in plan.facts_to_answer
    ]

    conflicts = getattr(
        grounding,
        "conflicts",
        (),
    )

    for conflict in conflicts:
        fact = getattr(
            conflict,
            "fact",
            None,
        )

        fact_value = getattr(
            fact,
            "value",
            None,
        )

        if isinstance(
            fact_value,
            str,
        ):
            values.append(
                fact_value
            )

    action = plan.action.value

    # The deterministic objective response communicates the scheduling
    # constraints when they exist.
    if action in {
        "state_objective",
        "request_alternative",
        "select_presented_choice",
    }:
        if world.facts.get(
            "preferred_day"
        ) is not None:
            values.append(
                "preferred_day"
            )

        if world.facts.get(
            "preferred_time"
        ) is not None:
            values.append(
                "preferred_time"
            )

    resolved: list[FactKey] = []

    for value in values:
        fact = _legacy_fact_key(
            value
        )

        if (
            fact is not None
            and fact not in resolved
        ):
            resolved.append(
                fact
            )

    return tuple(
        resolved
    )


def _selected_option(
    *,
    turn: TurnFrame,
    plan: ActionPlan,
) -> SlotOption | None:
    if (
        plan.action.value
        != "select_option"
    ):
        return None

    index = plan.selected_option_index

    if index is None:
        return None

    if not (
        0
        <= index
        < len(
            turn.appointment_options
        )
    ):
        raise RuntimeError(
            "Validated SELECT_OPTION points outside appointment_options."
        )

    return turn.appointment_options[
        index
    ]


def _advance_progress(
    *,
    world: PatientWorldModel,
    progress: AppointmentProgress,
    turn: TurnFrame,
    plan: ActionPlan,
) -> AppointmentProgress:
    """Update legacy-compatible evidence state from validated v2 meaning."""

    next_progress = progress
    action = plan.action.value

    # Preserve evidence for a single concrete/partial offered slot.
    #
    # Multiple alternatives cannot faithfully fit into the legacy
    # single-slot AppointmentProgress structure, so only the selected
    # option is persisted in that case.
    offered: SlotOption | None = None

    selected = _selected_option(
        turn=turn,
        plan=plan,
    )

    if selected is not None:
        offered = selected

    elif (
        action == "request_alternative"
        and len(
            turn.appointment_options
        ) == 1
    ):
        offered = (
            turn.appointment_options[
                0
            ]
        )

    if (
        offered is not None
        and (
            offered.day is not None
            or offered.time is not None
        )
    ):
        next_progress = record_slot_offer(
            next_progress,
            day=offered.day,
            time=offered.time,
        )

        if action == "select_option":
            next_progress = (
                record_offer_accepted(
                    next_progress
                )
            )

    day_shared = False
    time_shared = False

    communicated = {
        item.value
        for item in plan.facts_to_answer
    }

    if action in {
        "state_objective",
        "request_alternative",
        "select_presented_choice",
    }:
        day_shared = (
            world.facts.get(
                "preferred_day"
            )
            is not None
        )

        time_shared = (
            world.facts.get(
                "preferred_time"
            )
            is not None
        )

    if (
        "preferred_day"
        in communicated
    ):
        day_shared = True

    if (
        "preferred_time"
        in communicated
    ):
        time_shared = True

    if (
        day_shared
        or time_shared
    ):
        next_progress = (
            record_preferences_shared(
                next_progress,
                day_shared=day_shared,
                time_shared=time_shared,
            )
        )

    confirmed = (
        turn.confirmed_appointment
    )

    if (
        turn.booking_confirmed
        and confirmed is not None
        and _confirmation_matches_progress(
            progress=next_progress,
            confirmed=confirmed,
        )
    ):
        next_progress = (
            record_booking_confirmed(
                next_progress
            )
        )

    return next_progress


def _safety_guard_plan(
    *,
    progress: AppointmentProgress,
    turn: TurnFrame,
    plan: ActionPlan,
) -> tuple[
    ActionPlan,
    str | None,
]:
    """Prevent an unverified booking from triggering production hangup.

    The semantic/planning core remains authoritative for normal turns.

    This compatibility guard exists because autonomous_phone uses
    CommunicationKind.END_CONVERSATION as an actual telephony side effect.
    """

    if not turn.booking_confirmed:
        return plan, None

    confirmed = (
        turn.confirmed_appointment
    )

    if confirmed is None:
        return (
            ActionPlan(
                action=PatientActionKind(
                    "verify_booking"
                ),
                reason_code=(
                    "production_booking_confirmation_missing_slot"
                ),
                confidence=1.0,
            ),
            (
                "Just to confirm, "
                "is my appointment booked?"
            ),
        )

    if not (
        progress.has_offer
        and progress.offer_accepted
    ):
        return (
            ActionPlan(
                action=PatientActionKind(
                    "verify_booking"
                ),
                reason_code=(
                    "production_booking_confirmation_without_accepted_offer"
                ),
                confidence=1.0,
            ),
            (
                "Just to confirm, "
                "is my appointment booked?"
            ),
        )

    if not _confirmation_matches_progress(
        progress=progress,
        confirmed=confirmed,
    ):
        return (
            ActionPlan(
                action=PatientActionKind(
                    "reject_confirmation"
                ),
                reason_code=(
                    "production_confirmed_slot_conflicts_with_accepted_slot"
                ),
                confidence=1.0,
            ),
            (
                "That doesn't match the appointment I accepted. "
                "Could you verify the booking?"
            ),
        )

    return plan, None


def _compatibility_decision(
    *,
    world: PatientWorldModel,
    turn: TurnFrame,
    plan: ActionPlan,
    grounding: object,
    progress: AppointmentProgress,
) -> CommunicationDecision:
    action = plan.action.value

    kind = _ACTION_TO_COMMUNICATION.get(
        action
    )

    if kind is None:
        raise RuntimeError(
            "Reasoning v2 produced an action without a telephony "
            f"compatibility mapping: {action!r}."
        )

    # Keep semantic perception separate from patient-owned state.
    #
    # autonomous_phone interprets END_CONVERSATION as permission to send
    # an AudioSocket terminate packet. Never expose that side effect until
    # deterministic appointment progress says the objective is complete.
    if (
        kind
        is CommunicationKind.END_CONVERSATION
        and not progress.objective_complete
    ):
        kind = CommunicationKind.WAIT

    selected = _selected_option(
        turn=turn,
        plan=plan,
    )

    offered_day: str | None = None
    offered_time: str | None = None

    if selected is not None:
        offered_day = selected.day
        offered_time = selected.time

    elif (
        plan.action.value
        in {
            "reject_confirmation",
            "verify_booking",
        }
        and turn.confirmed_appointment
        is not None
    ):
        offered_day = (
            turn.confirmed_appointment.day
        )

        offered_time = (
            turn.confirmed_appointment.time
        )

    return CommunicationDecision(
        kind=kind,
        facts_to_communicate=(
            _facts_communicated(
                world=world,
                turn=turn,
                plan=plan,
                grounding=grounding,
            )
        ),
        offered_day=offered_day,
        offered_time=offered_time,
    )


class ReasoningV2PatientSession:
    """Stateful production adapter for the validated v2 reasoning core."""

    def __init__(
        self,
        *,
        scenario: PatientScenario,
        model: str,
        url: str,
        client: httpx.Client | None = None,
        semantic: Any | None = None,
        planner: Any | None = None,
        verbalizer: Any | None = None,
    ) -> None:
        self._scenario = scenario

        self._world = build_world_model(
            scenario
        )

        self._semantic = (
            semantic
            if semantic is not None
            else build_reasoning_v2_semantic_interpreter(
                model=model,
                url=url,
                client=client,
            )
        )

        self._planner = (
            planner
            if planner is not None
            else QwenPatientPlanner(
                model=reasoning_v2_edge_model_from_environment(model),
                url=url,
                client=client,
            )
        )

        self._verbalizer = (
            verbalizer
            if verbalizer is not None
            else GenericActionVerbalizer()
        )

        self._progress = (
            AppointmentProgress()
        )

        self._recent_agent_history: list[
            str
        ] = []

        self._recent_actions: list[
            ActionPlan
        ] = []

    @property
    def progress(
        self,
    ) -> AppointmentProgress:
        """Legacy-compatible appointment evidence used by the adapter."""

        return self._progress

    def prefetch_agent_turn(
        self,
        agent_turn: str,
    ) -> bool:
        """Prefetch is deliberately disabled for the first v2 integration.

        This prevents provisional ASR candidates from causing duplicate Qwen
        calls until committed-turn production behavior is validated.
        """

        del agent_turn
        return False

    def invalidate_prefetch(
        self,
    ) -> None:
        """No-op while v2 prefetch remains disabled."""

    def close(
        self,
    ) -> None:
        """Close owned reasoning components when applicable."""

        close_semantic = getattr(
            self._semantic,
            "close",
            None,
        )

        if callable(
            close_semantic
        ):
            close_semantic()

        close_planner = getattr(
            self._planner,
            "close",
            None,
        )

        if callable(
            close_planner
        ):
            close_planner()

    def handle_agent_turn(
        self,
        agent_turn: str,
    ) -> ReasoningV2SessionTurnResult:
        """Process one committed remote turn atomically."""

        normalized_turn = " ".join(
            agent_turn.split()
        )

        if not normalized_turn:
            raise ValueError(
                "Agent turn cannot be blank."
            )

        # Snapshot all mutable session state.
        #
        # No session state is committed until every stage below succeeds.
        pre_progress = self._progress

        prior_history = tuple(
            self._recent_agent_history
        )

        prior_actions = tuple(
            self._recent_actions
        )

        interpreter_started = (
            perf_counter()
        )

        frame = self._semantic.interpret(
            agent_turn=normalized_turn,
            recent_history=prior_history,
        )

        interpreter_seconds = (
            perf_counter()
            - interpreter_started
        )

        decision_started = (
            perf_counter()
        )

        grounding = (
            ground_fact_assertions(
                world=self._world,
                turn=frame,
            )
        )

        plan, repaired_from = (
            self._planner.plan(
                world=self._world,
                turn=frame,
                recent_actions=prior_actions,
            )
        )

        effective_plan, forced_text = (
            _safety_guard_plan(
                progress=pre_progress,
                turn=frame,
                plan=plan,
            )
        )

        next_progress = (
            _advance_progress(
                world=self._world,
                progress=pre_progress,
                turn=frame,
                plan=effective_plan,
            )
        )

        decision = (
            _compatibility_decision(
                world=self._world,
                turn=frame,
                plan=effective_plan,
                grounding=grounding,
                progress=next_progress,
            )
        )

        decision_seconds = (
            perf_counter()
            - decision_started
        )

        verbalizer_started = (
            perf_counter()
        )

        if (
            decision.kind
            is CommunicationKind.WAIT
        ):
            patient_text = ""
            verbalizer_seconds = 0.0

        elif forced_text is not None:
            patient_text = forced_text
            verbalizer_seconds = (
                perf_counter()
                - verbalizer_started
            )

        else:
            patient_text = (
                self._verbalizer.verbalize(
                    world=self._world,
                    turn=frame,
                    plan=effective_plan,
                    corrections=(
                        grounding.conflicts
                    ),
                )
            )

            verbalizer_seconds = (
                perf_counter()
                - verbalizer_started
            )

            if not patient_text.strip():
                raise RuntimeError(
                    "Non-WAIT v2 action produced blank patient speech."
                )

        state_update_started = (
            perf_counter()
        )

        # Build durable JSON-safe diagnostic evidence before commit.
        meaning: dict[str, object] = {
            "reasoning_version": "v2",
            "turn_frame": (
                frame.model_dump(
                    mode="json",
                )
            ),
            "action_plan": (
                effective_plan.model_dump(
                    mode="json",
                )
            ),
            "planner_repaired_from": [
                {
                    "code": item.code,
                    "detail": item.detail,
                }
                for item
                in repaired_from
            ],
            "fact_conflicts": [
                {
                    "fact": (
                        conflict.fact.value
                    ),
                    "asserted": (
                        conflict.asserted_value
                    ),
                    "authoritative": (
                        conflict.authoritative_value
                    ),
                }
                for conflict
                in grounding.conflicts
            ],
        }

        # Atomic commit happens last.
        self._progress = (
            next_progress
        )

        self._recent_agent_history.append(
            normalized_turn
        )

        self._recent_actions.append(
            effective_plan
        )

        state_update_seconds = (
            perf_counter()
            - state_update_started
        )

        return ReasoningV2SessionTurnResult(
            agent_turn=normalized_turn,
            meaning=meaning,
            grounded=grounding,
            decision=decision,
            patient_text=patient_text,
            progress=next_progress,
            timings=SessionTurnTimings(
                interpreter_seconds=(
                    interpreter_seconds
                ),
                decision_seconds=(
                    decision_seconds
                ),
                verbalizer_seconds=(
                    verbalizer_seconds
                ),
                state_update_seconds=(
                    state_update_seconds
                ),
            ),
        )
