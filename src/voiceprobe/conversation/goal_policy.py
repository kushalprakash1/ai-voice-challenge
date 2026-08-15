"""Goal-directed conversation policy for simulated patient calls.

Semantic interpretation describes what the external agent said. This module
decides whether the resulting action is compatible with the patient's
persistent scheduling mission.

The policy deliberately remembers workflow focus across turns so elliptical
follow-ups such as "Would you like to do that?" cannot escape the objective
guard merely because the current sentence omits the words "demo profile".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.grounding import GroundedTurnMeaning
from voiceprobe.conversation.meaning import (
    QuestionKind,
    WorkflowRelation,
)
from voiceprobe.conversation.objective import AppointmentProgress
from voiceprobe.conversation.state import FactKey
from voiceprobe.scenarios.models import PatientScenario


class WorkflowFocus(StrEnum):
    """Workflow currently controlling conversational references."""

    NONE = "none"
    SCHEDULING = "scheduling"
    SIDE_WORKFLOW = "side_workflow"


@dataclass(frozen=True, slots=True)
class GoalContext:
    """Persistent mission context carried across agent turns."""

    focus: WorkflowFocus = WorkflowFocus.NONE


_SIDE_WORKFLOW_RE = re.compile(
    r"\b(?:"
    r"profile|"
    r"patient profile|"
    r"demo patient|"
    r"demo profile|"
    r"account|"
    r"registration|"
    r"register|"
    r"enroll|"
    r"enrollment|"
    r"demo setup|"
    r"temporary setup"
    r")\b"
)

_SCHEDULING_RE = re.compile(
    r"\b(?:"
    r"schedule|"
    r"scheduling|"
    r"appointment|"
    r"appointments|"
    r"book|"
    r"booking|"
    r"availability|"
    r"available slot|"
    r"available time|"
    r"time slot"
    r")\b"
)

_REQUIRED_RE = re.compile(
    r"\b(?:"
    r"before i can|"
    r"before we can|"
    r"need to|"
    r"have to|"
    r"must|"
    r"required|"
    r"necessary|"
    r"in order to|"
    r"so i can|"
    r"so we can"
    r")\b"
)

_ELLIPTICAL_PERMISSION_RE = re.compile(
    r"\b(?:"
    r"would you like to|"
    r"do you want to|"
    r"would you like me to|"
    r"do you want me to|"
    r"should i|"
    r"shall i|"
    r"may i|"
    r"can i"
    r").{0,45}\b(?:that|this|it)\b"
)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _objective_facts(
    scenario: PatientScenario,
) -> tuple[FactKey, ...]:
    """Facts that are safe and useful when restating the mission."""
    facts: list[FactKey] = []

    if scenario.facts.preferred_day is not None:
        facts.append("preferred_day")

    if scenario.facts.preferred_time is not None:
        facts.append("preferred_time")

    return tuple(facts)


def apply_goal_policy(
    *,
    scenario: PatientScenario,
    grounded: GroundedTurnMeaning,
    progress: AppointmentProgress,
    agent_turn: str,
    context: GoalContext,
    base_decision: CommunicationDecision,
) -> tuple[CommunicationDecision, GoalContext]:
    """Guard one proposed response against the persistent scheduling goal."""
    meaning = grounded.meaning
    text = _normalize(agent_turn)

    # Booking evidence and concrete appointment offers are authoritative
    # scheduling events and always restore scheduling focus.
    if meaning.booking_confirmed or meaning.appointment_offer is not None:
        return (
            base_decision,
            GoalContext(focus=WorkflowFocus.SCHEDULING),
        )

    # Once complete, do not reopen the objective merely because a generic
    # prompt happens to appear in the same or a later turn.
    if progress.objective_complete:
        return base_decision, context

    # A receptionist asking how they can help is not ambiguous. Answer with
    # the immutable mission rather than asking the receptionist to clarify.
    if meaning.topic == "call purpose":
        return (
            CommunicationDecision(
                kind=CommunicationKind.ANSWER,
                facts_to_communicate=_objective_facts(scenario),
                state_objective=True,
            ),
            GoalContext(focus=WorkflowFocus.SCHEDULING),
        )

    explicit_side_workflow = (
        _SIDE_WORKFLOW_RE.search(text) is not None
    )
    explicit_scheduling = (
        _SCHEDULING_RE.search(text) is not None
    )
    explicitly_required = (
        _REQUIRED_RE.search(text) is not None
    )

    # A profile/account/setup workflow must not hijack the scheduling goal.
    # The narrow exception is an explicitly required scheduling prerequisite.
    if (
        explicit_side_workflow
        and not (
            explicitly_required
            and explicit_scheduling
        )
    ):
        return (
            CommunicationDecision(
                kind=CommunicationKind.DECLINE_WORKFLOW,
            ),
            GoalContext(focus=WorkflowFocus.SIDE_WORKFLOW),
        )

    # Context survives sentence boundaries. If the previous turn established
    # an unwanted side workflow, a bare fact request still belongs to that
    # side workflow unless the agent explicitly pivots back to scheduling.
    if context.focus is WorkflowFocus.SIDE_WORKFLOW:
        if meaning.requested_facts:
            if explicit_scheduling:
                return (
                    base_decision,
                    GoalContext(focus=WorkflowFocus.SCHEDULING),
                )

            return (
                CommunicationDecision(
                    kind=CommunicationKind.DECLINE_WORKFLOW,
                ),
                GoalContext(focus=WorkflowFocus.SIDE_WORKFLOW),
            )

        # "Would you like to do that?" inherits the active workflow rather
        # than being treated as a context-free uncertain yes/no question.
        if (
            _ELLIPTICAL_PERMISSION_RE.search(text) is not None
            or (
                meaning.question_kind
                is QuestionKind.WORKFLOW_PERMISSION
                and meaning.workflow_relation
                in {
                    WorkflowRelation.NONE,
                    WorkflowRelation.UNCERTAIN,
                }
            )
        ):
            return (
                CommunicationDecision(
                    kind=CommunicationKind.DECLINE_WORKFLOW,
                ),
                GoalContext(focus=WorkflowFocus.SIDE_WORKFLOW),
            )

    # Explicit scheduling language or an interpreter classification that is
    # known to advance the objective moves focus back to scheduling.
    if (
        explicit_scheduling
        or meaning.workflow_relation
        is WorkflowRelation.ADVANCES_OBJECTIVE
    ):
        return (
            base_decision,
            GoalContext(focus=WorkflowFocus.SCHEDULING),
        )

    return base_decision, context
