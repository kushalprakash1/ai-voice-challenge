
"""Deterministic policy over semantic frames.

No natural-language phrase matching occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .semantic_frame import (
    Commitment,
    Focus,
    Operation,
    SemanticFrame,
    SpeechAct,
)


class SemanticRoute(StrEnum):
    ANSWER_RESCHEDULE_REASON = "answer_reschedule_reason"
    ANSWER_FACT = "answer_fact"

    # Transaction authorization must return to the authoritative
    # scheduling state machine.
    TRANSACTION_GATE = "transaction_gate"

    WAIT = "wait"
    HOLD = "hold"

    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RoutedSemanticTurn:
    route: SemanticRoute
    fact_focus: Focus = Focus.NONE


_FACT_FOCI = {
    Focus.INSURANCE,
    Focus.PROVIDER_PREFERENCE,
    Focus.DOB,
    Focus.NAME,
    Focus.COMPLAINT,
    Focus.PREFERRED_DAY,
    Focus.PREFERRED_TIME,
}

_TRANSACTION_OPERATIONS = {
    Operation.BOOK,
    Operation.RESCHEDULE,
    Operation.CANCEL,
    Operation.KEEP,
}


def route_semantic_frame(
    frame: SemanticFrame,
) -> RoutedSemanticTurn:
    """Map semantic meaning to deterministic VoiceProbe behavior."""

    # Transaction gating has highest priority.
    if (
        frame.operation in _TRANSACTION_OPERATIONS
        and frame.commitment is Commitment.PERMISSION_REQUEST
    ):
        return RoutedSemanticTurn(
            SemanticRoute.TRANSACTION_GATE
        )

    # Incomplete language never triggers durable behavior.
    if frame.speech_act is SpeechAct.FRAGMENT:
        return RoutedSemanticTurn(
            SemanticRoute.HOLD
        )

    # Slot-list introductions should be allowed to continue.
    if frame.focus is Focus.SLOT_OPTIONS_INTRO:
        return RoutedSemanticTurn(
            SemanticRoute.HOLD
        )

    # Acknowledgements require no patient response.
    if frame.speech_act is SpeechAct.ACKNOWLEDGE:
        return RoutedSemanticTurn(
            SemanticRoute.WAIT
        )

    # Existing appointment information is not a new transaction.
    if (
        frame.focus is Focus.APPOINTMENT_STATUS
        and frame.speech_act is SpeechAct.INFORM
    ):
        return RoutedSemanticTurn(
            SemanticRoute.WAIT
        )

    # The semantic topic "visit_reason" describes WHY the patient
    # is seeking care. The authoritative value comes from PatientFacts.complaint,
    # so translate semantic focus -> authoritative fact focus here.
    if (
        frame.focus is Focus.VISIT_REASON
        and frame.speech_act in {
            SpeechAct.ASK,
            SpeechAct.REQUEST,
        }
    ):
        return RoutedSemanticTurn(
            SemanticRoute.ANSWER_FACT,
            fact_focus=Focus.COMPLAINT,
        )

    # Informational WHY-reschedule questions.
    if (
        frame.focus is Focus.RESCHEDULE_REASON
        and frame.speech_act in {
            SpeechAct.ASK,
            SpeechAct.REQUEST,
        }
        and frame.commitment is not Commitment.PERMISSION_REQUEST
    ):
        return RoutedSemanticTurn(
            SemanticRoute.ANSWER_RESCHEDULE_REASON
        )

    # Authoritative patient facts are rendered by Python.
    if (
        frame.focus in _FACT_FOCI
        and frame.speech_act in {
            SpeechAct.ASK,
            SpeechAct.REQUEST,
        }
    ):
        return RoutedSemanticTurn(
            SemanticRoute.ANSWER_FACT,
            fact_focus=frame.focus,
        )

    return RoutedSemanticTurn(
        SemanticRoute.UNKNOWN
    )
