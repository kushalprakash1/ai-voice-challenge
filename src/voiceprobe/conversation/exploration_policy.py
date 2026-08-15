"""Cooperative conversation policy used only for explicit exploration calls.

Exploration mode is intentionally separate from the normal scheduling mission.

Its purpose is to maximize benign dialogue coverage of the tested voice agent:
accept optional workflows, provide known scenario facts, continue through
workflow permissions, and expose as much of the remote dialogue tree as
possible.

It must never fabricate patient facts that are absent from the scenario.
"""

from __future__ import annotations

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.grounding import GroundedTurnMeaning
from voiceprobe.conversation.meaning import (
    QuestionKind,
    WorkflowDirection,
)
from voiceprobe.conversation.objective import AppointmentProgress
from voiceprobe.conversation.state import FactKey
from voiceprobe.scenarios.models import PatientScenario
from voiceprobe.target_memory import should_tolerate_target_conflict


def _objective_facts(
    scenario: PatientScenario,
) -> tuple[FactKey, ...]:
    """Facts useful when naturally stating the scheduling objective."""
    facts: list[FactKey] = []

    if scenario.facts.preferred_day is not None:
        facts.append("preferred_day")

    if scenario.facts.preferred_time is not None:
        facts.append("preferred_time")

    return tuple(facts)


def apply_exploration_policy(
    *,
    scenario: PatientScenario,
    grounded: GroundedTurnMeaning,
    progress: AppointmentProgress,
    agent_turn: str,
    base_decision: CommunicationDecision,
) -> CommunicationDecision:
    """Cooperate with benign workflows while preserving scenario truth.

    This function deliberately does not rewrite appointment offers,
    corrections, booking confirmations, repetition requests, or known fact
    answers. PatientBrain remains authoritative for those behaviors.

    Unknown patient information is also left to PatientBrain rather than being
    invented merely to keep a remote workflow moving.
    """
    meaning = grounded.meaning

    # Never interfere with authoritative scheduling events.
    if meaning.booking_confirmed or meaning.appointment_offer is not None:
        return base_decision

    # Once the actual scheduling objective has completed, retain normal
    # completion behavior rather than artificially prolonging the call.
    if progress.objective_complete:
        return base_decision

    # PRESERVE AUTHORITATIVE BOOKING VERIFICATION
    #
    # PatientBrain emits VERIFY_BOOKING when a concrete slot has
    # already been accepted but the remote agent tries to end the
    # call before confirming that the booking actually completed.
    #
    # Exploration mode must not replace that scheduling safeguard
    # with a generic restatement of the original objective.
    if base_decision.kind is CommunicationKind.VERIFY_BOOKING:
        return base_decision

    # Active recall: a previously observed target behavior may deliberately
    # conflict with caller truth during a demo workflow. Exploration can tolerate
    # that behavior without ever allowing it to overwrite scenario truth.
    if (
        grounded.conflicts
        and should_tolerate_target_conflict(agent_turn)
    ):
        # If the same turn also asks an actionable caller question, answer that
        # question instead of wasting the turn correcting known demo behavior.
        if meaning.requested_facts:
            return CommunicationDecision(
                kind=CommunicationKind.ANSWER,
                facts_to_communicate=meaning.requested_facts,
            )

        if meaning.topic == "call purpose":
            return CommunicationDecision(
                kind=CommunicationKind.ANSWER,
                facts_to_communicate=_objective_facts(scenario),
                state_objective=True,
            )

        if (
            meaning.question_kind is QuestionKind.WORKFLOW_PERMISSION
            and meaning.workflow_direction is not WorkflowDirection.STOP
        ):
            return CommunicationDecision(
                kind=CommunicationKind.AGREE,
            )

        # A standalone known demo assertion requires no argument from the
        # exploratory caller. Preserve truth internally and keep listening.
        return CommunicationDecision(
            kind=CommunicationKind.WAIT,
        )

    # Preserve genuine corrections, known fact disclosure, and repetition.
    if (
        grounded.conflicts
        or meaning.requested_facts
        or meaning.requests_repetition
    ):
        return base_decision

    # If the remote agent asks the purpose of the call, answer it rather than
    # producing a generic clarification.
    if meaning.topic == "call purpose":
        return CommunicationDecision(
            kind=CommunicationKind.ANSWER,
            facts_to_communicate=_objective_facts(scenario),
            state_objective=True,
        )

    # The core exploration behavior: say yes to benign workflow permissions,
    # including optional demo/profile flows that normal mode deliberately
    # rejects. Explicit STOP/cancel requests are not accepted.
    if (
        meaning.question_kind is QuestionKind.WORKFLOW_PERMISSION
        and meaning.workflow_direction is not WorkflowDirection.STOP
    ):
        return CommunicationDecision(
            kind=CommunicationKind.AGREE,
        )

    # If the remote agent tries to end before the scheduling objective has
    # completed, make one natural attempt to keep the interaction moving.
    if meaning.conversation_end_requested:
        return CommunicationDecision(
            kind=CommunicationKind.ANSWER,
            facts_to_communicate=_objective_facts(scenario),
            state_objective=True,
        )

    # Unsupported attributes remain unsupported. Do not invent email
    # addresses, phone numbers, addresses, medical history, or other facts.
    return base_decision
