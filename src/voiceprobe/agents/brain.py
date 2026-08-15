"""Deterministic patient reasoning over grounded conversation meaning.

PatientBrain decides what the simulated patient should communicate.
It does not generate final spoken wording. Scenario truth, objective
progress, and completion remain under deterministic Python control.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from voiceprobe.conversation.grounding import GroundedTurnMeaning
from voiceprobe.conversation.meaning import (
    QuestionKind,
    ResponseExpectation,
    TurnMeaning,
    WorkflowDirection,
    WorkflowRelation,
)
from voiceprobe.conversation.objective import AppointmentProgress
from voiceprobe.conversation.scheduling import time_matches_preference
from voiceprobe.conversation.state import FactKey
from voiceprobe.scenarios.models import PatientScenario, ProbeKind


class CommunicationKind(StrEnum):
    """High-level conversational behavior chosen by PatientBrain."""

    ANSWER = "answer"
    CORRECT = "correct"
    ACCEPT_OFFER = "accept_offer"
    ACCEPT_PARTIAL_OFFER = "accept_partial_offer"
    DECLINE_OFFER = "decline_offer"
    REPEAT = "repeat"
    ACKNOWLEDGE_COMPLETE = "acknowledge_complete"
    END_CONVERSATION = "end_conversation"
    ASK_AGENT_TO_REPEAT = "ask_agent_to_repeat"
    VERIFY_BOOKING = "verify_booking"
    AGREE = "agree"
    DECLINE_WORKFLOW = "decline_workflow"
    WAIT = "wait"
    CLARIFY = "clarify"


@dataclass(frozen=True, slots=True)
class CommunicationDecision:
    """Semantic instructions for the future natural-language verbalizer."""

    kind: CommunicationKind
    facts_to_communicate: tuple[FactKey, ...] = ()
    offered_day: str | None = None
    offered_time: str | None = None
    probe: ProbeKind | None = None
    state_objective: bool = False


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
                    kind=(
                        CommunicationKind.END_CONVERSATION
                        if meaning.conversation_end_requested
                        else CommunicationKind.ACKNOWLEDGE_COMPLETE
                    ),
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

            if self._offer_conflicts_with_preferences(
                scenario=scenario,
                day=offer.day,
                time=offer.time,
            ):
                return CommunicationDecision(
                    kind=CommunicationKind.DECLINE_OFFER,
                    facts_to_communicate=self._preference_facts(scenario),
                    offered_day=offer.day,
                    offered_time=offer.time,
                )

            if self._offer_missing_required_detail(
                scenario=scenario,
                day=offer.day,
                time=offer.time,
            ):
                return CommunicationDecision(
                    kind=CommunicationKind.ACCEPT_PARTIAL_OFFER,
                    offered_day=offer.day,
                    offered_time=offer.time,
                )

            return CommunicationDecision(
                kind=CommunicationKind.ACCEPT_OFFER,
                offered_day=offer.day,
                offered_time=offer.time,
            )

        if meaning.requested_facts:
            return CommunicationDecision(
                kind=CommunicationKind.ANSWER,
                facts_to_communicate=meaning.requested_facts,
            )

        if meaning.conversation_end_requested:
            # A goodbye is not proof that the scheduling objective has
            # completed. Never voluntarily cooperate with premature call
            # termination while the appointment remains unfinished.
            #
            # A correctly matched booking confirmation is handled earlier
            # in this method and may legitimately return END_CONVERSATION.
            if progress.objective_complete:
                return CommunicationDecision(
                    kind=CommunicationKind.END_CONVERSATION,
                )

            # If we already accepted a concrete slot but have not received
            # authoritative confirmation that it was booked, keep the call
            # alive long enough to request that confirmation.
            if progress.offer_accepted:
                return CommunicationDecision(
                    kind=CommunicationKind.VERIFY_BOOKING,
                    offered_day=progress.offered_day,
                    offered_time=progress.offered_time,
                )

            # No valid slot has even been accepted yet. Decline the attempt
            # to terminate and keep pursuing the scheduling objective.
            return CommunicationDecision(
                kind=CommunicationKind.DECLINE_WORKFLOW,
            )

        if meaning.unclear:
            return CommunicationDecision(
                kind=CommunicationKind.CLARIFY,
            )

        if self._should_wait(meaning):
            return CommunicationDecision(
                kind=CommunicationKind.WAIT,
            )

        if meaning.response_expectation is ResponseExpectation.YES_NO:
            if meaning.question_kind is QuestionKind.WORKFLOW_PERMISSION:
                if meaning.workflow_direction is WorkflowDirection.STOP:
                    return CommunicationDecision(
                        kind=CommunicationKind.DECLINE_WORKFLOW,
                    )

                if meaning.workflow_direction is WorkflowDirection.CONTINUE:
                    if (
                        meaning.workflow_relation
                        is WorkflowRelation.ADVANCES_OBJECTIVE
                    ):
                        return CommunicationDecision(
                            kind=CommunicationKind.AGREE,
                        )

                    if meaning.workflow_relation in {
                        WorkflowRelation.NONE,
                        WorkflowRelation.OPPOSES_OBJECTIVE,
                    }:
                        return CommunicationDecision(
                            kind=CommunicationKind.DECLINE_WORKFLOW,
                        )

                    return CommunicationDecision(
                        kind=CommunicationKind.CLARIFY,
                    )

            # A yes/no question about an unsupported patient attribute must
            # never become an automatic yes merely because answering it might
            # help the external agent's workflow.
            return CommunicationDecision(
                kind=CommunicationKind.CLARIFY,
            )

        return CommunicationDecision(
            kind=CommunicationKind.CLARIFY,
        )

    @staticmethod
    def _should_wait(meaning: TurnMeaning) -> bool:
        """Return whether the agent turn requires no patient response.

        WAIT is deliberately conservative. It is used only when semantic
        interpretation found no question, patient fact, scheduling offer,
        booking action, repetition request, ending, ambiguity, or substantive
        topic.
        """
        return bool(
            not meaning.unclear
            and meaning.response_expectation is ResponseExpectation.NONE
            and meaning.question_kind is QuestionKind.NONE
            and meaning.workflow_direction is WorkflowDirection.NONE
            and meaning.topic is None
            and not meaning.requested_facts
            and not meaning.stated_facts
            and meaning.appointment_offer is None
            and not meaning.booking_confirmed
            and not meaning.conversation_end_requested
            and not meaning.requests_repetition
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
    def _offer_conflicts_with_preferences(
        *,
        scenario: PatientScenario,
        day: str | None,
        time: str | None,
    ) -> bool:
        """Return whether any supplied slot detail conflicts with patient truth."""
        preferred_day = scenario.facts.preferred_day
        preferred_time = scenario.facts.preferred_time

        if (
            day is not None
            and preferred_day is not None
            and " ".join(day.casefold().split())
            != " ".join(preferred_day.casefold().split())
        ):
            return True

        return bool(
            time is not None
            and preferred_time is not None
            and not time_matches_preference(
                preferred=preferred_time,
                offered=time,
            )
        )

    @staticmethod
    def _offer_missing_required_detail(
        *,
        scenario: PatientScenario,
        day: str | None,
        time: str | None,
    ) -> bool:
        """Return whether a compatible offer is still missing required detail."""
        preferred_day = scenario.facts.preferred_day
        preferred_time = scenario.facts.preferred_time

        if preferred_day is not None and day is None:
            return True

        return bool(preferred_time is not None and time is None)
