"""Deterministic patient reasoning over grounded conversation meaning.

PatientBrain decides what the simulated patient should communicate.
It does not generate final spoken wording. Scenario truth, objective
progress, and completion remain under deterministic Python control.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from voiceprobe.conversation.grounding import GroundedTurnMeaning
from voiceprobe.conversation.objective import AppointmentProgress
from voiceprobe.conversation.scheduling import time_matches_preference
from voiceprobe.conversation.state import FactKey
from voiceprobe.scenarios.models import PatientScenario


class CommunicationKind(StrEnum):
    """High-level conversational behavior chosen by PatientBrain."""

    ANSWER = "answer"
    CORRECT = "correct"
    ACCEPT_OFFER = "accept_offer"
    DECLINE_OFFER = "decline_offer"
    REPEAT = "repeat"
    ACKNOWLEDGE_COMPLETE = "acknowledge_complete"
    CLARIFY = "clarify"


@dataclass(frozen=True, slots=True)
class CommunicationDecision:
    """Semantic instructions for the future natural-language verbalizer."""

    kind: CommunicationKind
    facts_to_communicate: tuple[FactKey, ...] = ()
    offered_day: str | None = None
    offered_time: str | None = None


class PatientBrain:
    """Choose patient behavior from grounded semantic meaning."""

    def decide(
        self,
        *,
        scenario: PatientScenario,
        grounded: GroundedTurnMeaning,
        progress: AppointmentProgress,
    ) -> CommunicationDecision:
        """Determine the patient's next semantic communication."""

        meaning = grounded.meaning

        if grounded.conflicts:
            conflict_facts = tuple(conflict.fact for conflict in grounded.conflicts)

            return CommunicationDecision(
                kind=CommunicationKind.CORRECT,
                facts_to_communicate=conflict_facts,
            )

        if meaning.requests_repetition:
            return CommunicationDecision(
                kind=CommunicationKind.REPEAT,
            )

        if meaning.booking_confirmed:
            confirmation_offer = meaning.appointment_offer

            confirmation_matches = self._confirmation_matches_offer(
                progress=progress,
                day=(
                    confirmation_offer.day if confirmation_offer is not None else None
                ),
                time=(
                    confirmation_offer.time if confirmation_offer is not None else None
                ),
            )

            if progress.offer_accepted and confirmation_matches:
                return CommunicationDecision(
                    kind=CommunicationKind.ACKNOWLEDGE_COMPLETE,
                )

            # If the patient already accepted a concrete slot and the
            # receptionist explicitly confirms different details, the
            # statement was understood. It is a booking inconsistency,
            # not a request for repetition.
            if progress.offer_accepted and confirmation_offer is not None:
                return CommunicationDecision(
                    kind=CommunicationKind.DECLINE_OFFER,
                    facts_to_communicate=self._preference_facts(scenario),
                    offered_day=confirmation_offer.day,
                    offered_time=confirmation_offer.time,
                )

            return CommunicationDecision(
                kind=CommunicationKind.CLARIFY,
            )

        if meaning.appointment_offer is not None:
            offer = meaning.appointment_offer

            if self._offer_matches_preferences(
                scenario=scenario,
                day=offer.day,
                time=offer.time,
            ):
                return CommunicationDecision(
                    kind=CommunicationKind.ACCEPT_OFFER,
                    offered_day=offer.day,
                    offered_time=offer.time,
                )

            return CommunicationDecision(
                kind=CommunicationKind.DECLINE_OFFER,
                facts_to_communicate=self._preference_facts(scenario),
                offered_day=offer.day,
                offered_time=offer.time,
            )

        if meaning.requested_facts:
            return CommunicationDecision(
                kind=CommunicationKind.ANSWER,
                facts_to_communicate=meaning.requested_facts,
            )

        if meaning.unclear:
            return CommunicationDecision(
                kind=CommunicationKind.CLARIFY,
            )

        return CommunicationDecision(
            kind=CommunicationKind.CLARIFY,
        )

    @staticmethod
    def _confirmation_matches_offer(
        *,
        progress: AppointmentProgress,
        day: str | None,
        time: str | None,
    ) -> bool:
        """Check that explicit confirmation details match the accepted slot."""
        if day is not None:
            if progress.offered_day is None:
                return False

            if " ".join(day.casefold().split()) != " ".join(
                progress.offered_day.casefold().split()
            ):
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
    def _preference_facts(
        scenario: PatientScenario,
    ) -> tuple[FactKey, ...]:
        facts: list[FactKey] = []

        if scenario.facts.preferred_day is not None:
            facts.append("preferred_day")

        if scenario.facts.preferred_time is not None:
            facts.append("preferred_time")

        return tuple(facts)

    @staticmethod
    def _offer_matches_preferences(
        *,
        scenario: PatientScenario,
        day: str | None,
        time: str | None,
    ) -> bool:
        preferred_day = scenario.facts.preferred_day
        preferred_time = scenario.facts.preferred_time

        if (
            day is not None
            and preferred_day is not None
            and day.casefold() != preferred_day.casefold()
        ):
            return False

        return time_matches_preference(
            preferred=preferred_time,
            offered=time,
        )
