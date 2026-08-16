"""Structured scheduling-flow state for VoiceProbe v3.

The flow tracker is deliberately separate from language understanding. Patient
facts remain authoritative elsewhere. This module records only what has been
communicated to the remote scheduling agent and what the remote agent has
explicitly confirmed.

The target agent is allowed to skip, repeat, combine, or revisit stages. The
tracker therefore does not force a linear script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from .models import DecisionKind, PolicyDecision


class FlowStage(StrEnum):
    PROFILE = "profile"
    IDENTITY = "identity"
    DOB = "dob"
    VISIT_REASON = "visit_reason"
    APPOINTMENT_TYPE = "appointment_type"
    INSURANCE = "insurance"
    DATE_TIME = "date_time"
    PROVIDER = "provider"
    SLOT = "slot"
    CONFIRMATION = "confirmation"
    COMPLETE = "complete"


ORDERED_STAGES: tuple[FlowStage, ...] = (
    FlowStage.PROFILE,
    FlowStage.IDENTITY,
    FlowStage.DOB,
    FlowStage.VISIT_REASON,
    FlowStage.APPOINTMENT_TYPE,
    FlowStage.INSURANCE,
    FlowStage.DATE_TIME,
    FlowStage.PROVIDER,
    FlowStage.SLOT,
    FlowStage.CONFIRMATION,
)


@dataclass(frozen=True, slots=True)
class FlowSnapshot:
    """Immutable public view of flow progress."""

    communicated: frozenset[FlowStage]
    confirmed: frozenset[FlowStage]
    current_stage: FlowStage
    complete: bool
    accepted_slot_text: str | None = None
    booking_confirmation_text: str | None = None
    allow_earlier_week_afternoons: bool = False


@dataclass(slots=True)
class _MutableFlowState:
    communicated: set[FlowStage] = field(default_factory=set)
    confirmed: set[FlowStage] = field(default_factory=set)
    accepted_slot_text: str | None = None
    booking_confirmation_text: str | None = None
    allow_earlier_week_afternoons: bool = False


class SchedulingFlowTracker:
    """Track scheduling progress without mutating authoritative patient facts."""

    def __init__(self) -> None:
        self._state = _MutableFlowState()

    def snapshot(self) -> FlowSnapshot:
        current = self._current_stage()
        complete = FlowStage.CONFIRMATION in self._state.confirmed

        return FlowSnapshot(
            communicated=frozenset(self._state.communicated),
            confirmed=frozenset(self._state.confirmed),
            current_stage=(
                FlowStage.COMPLETE
                if complete
                else current
            ),
            complete=complete,
            accepted_slot_text=self._state.accepted_slot_text,
            booking_confirmation_text=self._state.booking_confirmation_text,
            allow_earlier_week_afternoons=(
                self._state.allow_earlier_week_afternoons
            ),
        )

    def observe_remote_turn(self, agent_turn: str) -> FlowSnapshot:
        """Record explicit confirmations/assertions from the remote agent."""

        text = " ".join(agent_turn.casefold().split())
        raw = " ".join(agent_turn.split())

        if _contains_any(
            text,
            (
                "profile is set up",
                "profile is created",
                "created your demo patient profile",
                "demo patient profile is set up",
            ),
        ):
            self._confirm(
                FlowStage.PROFILE,
                FlowStage.IDENTITY,
            )

        if (
            "date of birth" in text
            and _contains_any(
                text,
                (
                    "april 12, 1998",
                    "april 12 1998",
                    "april 12th, 1998",
                    "april 12th 1998",
                    "april twelfth nineteen ninety eight",
                ),
            )
        ):
            self._confirm(FlowStage.DOB)

        if _contains_any(
            text,
            (
                "friday afternoon",
                "following friday",
                "friday, august 28",
            ),
        ) and _contains_any(
            text,
            (
                "check",
                "openings",
                "appointments",
                "availability",
                "available",
            ),
        ):
            self._confirm(FlowStage.DATE_TIME)

        # "first available" being offered is not confirmation of the patient's
        # preference. Confirmation is only recorded when the remote agent later
        # acknowledges that preference.
        if _contains_any(
            text,
            (
                "first available is fine",
                "first available provider",
                "any available provider",
                "no provider preference",
            ),
        ):
            self._confirm(FlowStage.PROVIDER)

        slot_text = _extract_concrete_slot(raw)

        # Confirmation is transaction-relative, not merely lexical.
        #
        # Persistent test profiles may contain appointments created during an
        # earlier call. Therefore "you already have ... booked for 2:15 PM"
        # must never complete the current call unless VoiceProbe has already
        # accepted a concrete slot during this runtime.
        if (
            slot_text is not None
            and FlowStage.SLOT in self._state.communicated
            and _contains_any(
                text,
                (
                    "scheduled",
                    "booked",
                    "appointment is confirmed",
                    "appointment has been confirmed",
                    "you're confirmed",
                    "you are confirmed",
                    "reserved",
                ),
            )
        ):
            self._state.accepted_slot_text = slot_text
            self._state.booking_confirmation_text = raw
            self._confirm(
                FlowStage.SLOT,
                FlowStage.CONFIRMATION,
            )

        return self.snapshot()

    def apply_decision(
        self,
        decision: PolicyDecision,
    ) -> FlowSnapshot:
        """Record only the facts/preferences that VoiceProbe actually sent."""

        kind = decision.kind
        reason = decision.reason

        if kind == DecisionKind.CREATE_PROFILE:
            self._communicate(
                FlowStage.PROFILE,
                FlowStage.IDENTITY,
            )

        elif kind == DecisionKind.ANSWER_FACT:
            if reason in {
                "first_name_requested",
                "last_name_requested",
                "full_name_requested",
            }:
                self._communicate(FlowStage.IDENTITY)
            elif reason == "dob_requested":
                self._communicate(FlowStage.DOB)
            elif reason == "insurance_requested":
                self._communicate(FlowStage.INSURANCE)

        elif kind == DecisionKind.CORRECT_FACT:
            if reason == "dob_correction":
                self._communicate(FlowStage.DOB)

        elif kind == DecisionKind.CORRECT_AND_STATE_OBJECTIVE:
            self._communicate(
                FlowStage.DOB,
                FlowStage.DATE_TIME,
            )

        elif kind == DecisionKind.STATE_OBJECTIVE:
            self._communicate(FlowStage.DATE_TIME)

        elif kind == DecisionKind.ANSWER_COMPLAINT:
            self._communicate(FlowStage.VISIT_REASON)

        elif kind == DecisionKind.ANSWER_APPOINTMENT_TYPE:
            self._communicate(FlowStage.APPOINTMENT_TYPE)

        elif kind == DecisionKind.ANSWER_VISIT_DETAILS:
            self._communicate(
                FlowStage.VISIT_REASON,
                FlowStage.APPOINTMENT_TYPE,
            )

        elif kind == DecisionKind.ANSWER_PROVIDER_PREFERENCE:
            self._communicate(FlowStage.PROVIDER)

        elif kind == DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON:
            # Live run 4: the remote side explicitly reported that the
            # requested Friday afternoon was unavailable and offered an
            # alternate-day search. Relax only the day constraint.
            # Afternoon remains mandatory.
            self._state.allow_earlier_week_afternoons = True
            self._communicate(FlowStage.DATE_TIME)

        elif kind in {
            DecisionKind.DECLINE_INCOMPATIBLE_OFFER,
            DecisionKind.GRANT_PERMISSION,
            DecisionKind.CHOOSE_SEARCH_BRANCH,
        }:
            self._communicate(FlowStage.DATE_TIME)

        return self.snapshot()

    def record_slot_acceptance(
        self,
        slot_text: str,
    ) -> FlowSnapshot:
        """Record a concrete slot only after VoiceProbe explicitly accepts it."""

        normalized = " ".join(slot_text.split())

        if not normalized:
            raise ValueError("slot_text must not be empty")

        self._state.accepted_slot_text = normalized
        self._communicate(FlowStage.SLOT)
        return self.snapshot()

    def relax_day_constraint_for_afternoon(self) -> FlowSnapshot:
        """Preserve PM while allowing Monday-Thursday after explicit fallback."""

        self._state.allow_earlier_week_afternoons = True
        self._communicate(FlowStage.DATE_TIME)
        return self.snapshot()

    def _communicate(
        self,
        *stages: FlowStage,
    ) -> None:
        self._state.communicated.update(stages)

    def _confirm(
        self,
        *stages: FlowStage,
    ) -> None:
        self._state.communicated.update(stages)
        self._state.confirmed.update(stages)

    def _current_stage(self) -> FlowStage:
        for stage in ORDERED_STAGES:
            if stage not in self._state.communicated:
                return stage

        # All requirements have been communicated, but the booking itself may
        # still be awaiting remote confirmation.
        return FlowStage.CONFIRMATION


def _contains_any(
    text: str,
    phrases: tuple[str, ...],
) -> bool:
    return any(phrase in text for phrase in phrases)


_SLOT_TIME_RE = re.compile(
    r"\b(?:1[0-2]|0?[1-9])"
    r"(?::[0-5]\d|\.[0-5]\d)?"
    r"\s*(?:a\.?m\.?|p\.?m\.?|am|pm)\b",
    flags=re.IGNORECASE,
)

_SPOKEN_SLOT_TIME_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"(?:\s+(?:fifteen|thirty|forty[\s-]?five))?"
    r"\s+(?:a\.?m\.?|p\.?m\.?|am|pm)\b",
    flags=re.IGNORECASE,
)


def extract_concrete_slot(text: str) -> str | None:
    """Extract a concrete digit or common spoken appointment time."""

    match = _SLOT_TIME_RE.search(text)

    if match is not None:
        return match.group(0)

    spoken = _SPOKEN_SLOT_TIME_RE.search(text)

    if spoken is not None:
        return spoken.group(0)

    return None


def _extract_concrete_slot(text: str) -> str | None:
    # Compatibility alias for the existing internal call site.
    return extract_concrete_slot(text)
