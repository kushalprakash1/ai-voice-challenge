"""Deterministic semantic evaluation for Reasoning Core v2.

This module does not judge whether a response sounds stylistically ideal.

It checks invariants that should hold regardless of phrasing:

- deterministic policy must remain valid
- hard scheduling constraints must never be violated
- the caller must not terminate before booking/end evidence
- the caller must not speak while the remote agent is still working
- a required response should not silently disappear
- confirmed appointments should match the slot the caller selected
- detected patient-truth conflicts should actually be corrected
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from voiceprobe.reasoning.action_plan import (
    ActionPlan,
    PatientActionKind,
)
from voiceprobe.reasoning.constraint_validator import (
    ConstraintValidator,
)
from voiceprobe.reasoning.fact_grounding import (
    FactGrounding,
)
from voiceprobe.reasoning.turn_frame import (
    RequestedAction,
    SlotOption,
    TurnFrame,
)
from voiceprobe.reasoning.world_model import (
    PatientWorldModel,
)


class EvaluationSeverity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"


@dataclass(
    frozen=True,
    slots=True,
)
class EvaluationFinding:
    code: str
    severity: EvaluationSeverity
    detail: str


@dataclass(
    frozen=True,
    slots=True,
)
class ConversationEvaluationState:
    """Cross-turn state needed for semantic regression checks."""

    last_selected_option: SlotOption | None = None


_NON_ALNUM_RE = re.compile(
    r"[^a-z0-9]+",
)


def _normalize(
    value: object,
) -> str:
    return _NON_ALNUM_RE.sub(
        "",
        str(value).casefold(),
    )


def _same_when_present(
    left: str | None,
    right: str | None,
) -> bool:
    """Compare fields only when both sides provide them."""

    if left is None or right is None:
        return True

    return (
        _normalize(left)
        == _normalize(right)
    )


def _patient_text_contains(
    patient_text: str,
    value: object,
) -> bool:
    candidate = _normalize(
        value
    )

    if not candidate:
        return True

    return candidate in _normalize(
        patient_text
    )


def evaluate_turn(
    *,
    world: PatientWorldModel,
    turn: TurnFrame,
    plan: ActionPlan,
    grounding: FactGrounding,
    patient_text: str,
    state: ConversationEvaluationState | None = None,
    validator: ConstraintValidator | None = None,
) -> tuple[
    tuple[EvaluationFinding, ...],
    ConversationEvaluationState,
]:
    """Evaluate one completed v2 reasoning turn."""

    if state is None:
        state = ConversationEvaluationState()

    if validator is None:
        validator = ConstraintValidator()

    findings: list[
        EvaluationFinding
    ] = []

    # --------------------------------------------------------
    # 1. Re-run deterministic policy validation independently.
    # --------------------------------------------------------

    violations = validator.validate(
        world=world,
        turn=turn,
        plan=plan,
    )

    for violation in violations:
        findings.append(
            EvaluationFinding(
                code=(
                    "policy_violation:"
                    f"{violation.code}"
                ),
                severity=EvaluationSeverity.CRITICAL,
                detail=violation.detail,
            )
        )

    # --------------------------------------------------------
    # 2. Turn-taking invariants.
    # --------------------------------------------------------

    if (
        turn.agent_is_still_working
        and plan.action
        is not PatientActionKind.WAIT
    ):
        findings.append(
            EvaluationFinding(
                code="spoke_while_agent_working",
                severity=EvaluationSeverity.CRITICAL,
                detail=(
                    "Remote agent is still working, but caller "
                    f"planned {plan.action.value!r}."
                ),
            )
        )

    if (
        turn.response_required
        and plan.action
        is PatientActionKind.WAIT
        and turn.requested_action
        is not RequestedAction.WAIT
    ):
        findings.append(
            EvaluationFinding(
                code="silent_when_response_required",
                severity=EvaluationSeverity.WARNING,
                detail=(
                    "Semantic layer says a response is required, "
                    "but the planner chose WAIT."
                ),
            )
        )

    # Passive informational turns must not spontaneously terminate.
    if (
        turn.requested_action
        is RequestedAction.NONE
        and not turn.response_required
        and not turn.booking_confirmed
        and not turn.conversation_end_requested
        and plan.action
        is PatientActionKind.END_CONVERSATION
    ):
        findings.append(
            EvaluationFinding(
                code="premature_end_conversation",
                severity=EvaluationSeverity.CRITICAL,
                detail=(
                    "Conversation ended without booking confirmation "
                    "or an explicit remote conversation-end signal."
                ),
            )
        )

    # --------------------------------------------------------
    # 3. Booking state.
    # --------------------------------------------------------

    if (
        turn.booking_confirmed
        and turn.confirmed_appointment is None
    ):
        findings.append(
            EvaluationFinding(
                code="booking_without_confirmed_slot",
                severity=EvaluationSeverity.WARNING,
                detail=(
                    "booking_confirmed=true but no structured "
                    "confirmed appointment was extracted."
                ),
            )
        )

    if (
        turn.booking_confirmed
        and turn.confirmed_appointment is not None
        and turn.requested_action
        is RequestedAction.NONE
        and not turn.response_required
        and plan.action
        is not PatientActionKind.END_CONVERSATION
    ):
        findings.append(
            EvaluationFinding(
                code="confirmed_booking_not_closed",
                severity=EvaluationSeverity.CRITICAL,
                detail=(
                    "Remote side confirmed the booking, but the "
                    "caller did not terminate cleanly."
                ),
            )
        )

    # --------------------------------------------------------
    # 4. Option-choice invariants beyond schema validity.
    # --------------------------------------------------------

    if (
        turn.requested_action
        is RequestedAction.CHOOSE_OPTION
    ):
        compatible = (
            validator.compatible_option_indices(
                world=world,
                turn=turn,
            )
        )

        if (
            len(compatible) == 0
            and plan.action
            is not PatientActionKind.REQUEST_ALTERNATIVE
        ):
            findings.append(
                EvaluationFinding(
                    code="failed_to_reject_incompatible_options",
                    severity=EvaluationSeverity.CRITICAL,
                    detail=(
                        "No offered option satisfied caller hard "
                        "constraints, but REQUEST_ALTERNATIVE "
                        "was not selected."
                    ),
                )
            )

        if len(compatible) == 1:
            expected = compatible[0]

            if (
                plan.action
                is not PatientActionKind.SELECT_OPTION
                or plan.selected_option_index
                != expected
            ):
                findings.append(
                    EvaluationFinding(
                        code="failed_to_select_only_compatible_option",
                        severity=EvaluationSeverity.CRITICAL,
                        detail=(
                            "Exactly one offered option satisfied "
                            "all hard constraints, but it was not "
                            "selected."
                        ),
                    )
                )

    # --------------------------------------------------------
    # 5. Cross-turn selected-slot -> confirmed-slot consistency.
    # --------------------------------------------------------

    next_state = state

    if (
        plan.action
        is PatientActionKind.SELECT_OPTION
        and plan.selected_option_index is not None
        and 0
        <= plan.selected_option_index
        < len(turn.appointment_options)
    ):
        next_state = (
            ConversationEvaluationState(
                last_selected_option=(
                    turn.appointment_options[
                        plan.selected_option_index
                    ]
                )
            )
        )

    confirmed = (
        turn.confirmed_appointment
    )

    selected = (
        state.last_selected_option
    )

    if (
        turn.booking_confirmed
        and confirmed is not None
        and selected is not None
    ):
        same_day = _same_when_present(
            selected.day,
            confirmed.day,
        )

        same_time = _same_when_present(
            selected.time,
            confirmed.time,
        )

        same_provider = _same_when_present(
            selected.provider,
            confirmed.provider,
        )

        if not (
            same_day
            and same_time
            and same_provider
        ):
            findings.append(
                EvaluationFinding(
                    code="confirmed_slot_differs_from_selected",
                    severity=EvaluationSeverity.CRITICAL,
                    detail=(
                        "Remote booking confirmation does not match "
                        "the slot previously selected by the caller."
                    ),
                )
            )

    # --------------------------------------------------------
    # 6. Fact-conflict correction realization.
    # --------------------------------------------------------

    if (
        grounding.conflicts
        and not patient_text.strip()
    ):
        findings.append(
            EvaluationFinding(
                code="fact_conflict_not_answered",
                severity=EvaluationSeverity.CRITICAL,
                detail=(
                    "A conflicting remote assertion was detected "
                    "but the caller produced no correction."
                ),
            )
        )

    for conflict in grounding.conflicts:
        if not _patient_text_contains(
            patient_text,
            conflict.authoritative_value,
        ):
            findings.append(
                EvaluationFinding(
                    code="authoritative_correction_not_realized",
                    severity=EvaluationSeverity.WARNING,
                    detail=(
                        f"Conflict for {conflict.fact.value!r} was "
                        "detected, but the authoritative value was "
                        "not found in the caller response."
                    ),
                )
            )

    return (
        tuple(findings),
        next_state,
    )
