"""Deterministic validation of model-proposed patient actions.

The LLM may reason and propose.

It never receives final authority over immutable caller constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from voiceprobe.conversation.scheduling import (
    time_matches_preference,
)
from voiceprobe.reasoning.action_plan import (
    ActionPlan,
    PatientActionKind,
)
from voiceprobe.reasoning.turn_frame import (
    RequestedAction,
    SlotOption,
    TurnFrame,
)
from voiceprobe.reasoning.world_model import (
    ConstraintSpec,
    ConstraintStrength,
    PatientWorldModel,
)


@dataclass(
    frozen=True,
    slots=True,
)
class ConstraintViolation:
    """One deterministic reason a proposed action is invalid."""

    code: str
    detail: str


def _normalize_text(
    value: str,
) -> str:
    value = " ".join(
        value.casefold().split()
    )

    for prefix in (
        "a ",
        "an ",
        "the ",
    ):
        if value.startswith(prefix):
            value = value[
                len(prefix):
            ]
            break

    return value


def _option_value(
    *,
    option: SlotOption,
    field: str,
) -> str | None:

    if field == "day":
        return option.day

    if field == "time":
        return (
            option.time
            if option.time is not None
            else option.daypart
        )

    if field == "provider":
        return option.provider

    if field == "appointment_type":
        return option.appointment_type

    return None


def _matches_constraint(
    *,
    constraint: ConstraintSpec,
    candidate: str,
) -> bool:

    if constraint.field == "time":
        return time_matches_preference(
            preferred=constraint.value,
            offered=candidate,
        )

    return (
        _normalize_text(
            constraint.value
        )
        ==
        _normalize_text(
            candidate
        )
    )


class ConstraintValidator:
    """Validate proposed plans against turn semantics and patient truth."""

    def compatible_option_indices(
        self,
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
    ) -> tuple[int, ...]:
        """Return every offered appointment option satisfying policy.

        This is entirely generic.

        It does not know Alex, Friday, afternoon, or any literal scenario.
        Every option is tested using the same deterministic constraint
        validation that protects final planner actions.
        """

        if (
            turn.requested_action
            is not RequestedAction.CHOOSE_OPTION
        ):
            return ()

        compatible: list[int] = []

        for index in range(
            len(turn.appointment_options)
        ):
            probe = ActionPlan(
                action=PatientActionKind.SELECT_OPTION,
                selected_option_index=index,
                reason_code="compatibility_probe",
                confidence=1.0,
            )

            violations = self.validate(
                world=world,
                turn=turn,
                plan=probe,
            )

            if not violations:
                compatible.append(
                    index
                )

        return tuple(
            compatible
        )

    def validate(
        self,
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
        plan: ActionPlan,
    ) -> tuple[ConstraintViolation, ...]:

        violations: list[
            ConstraintViolation
        ] = []

        self._validate_action_matches_turn(
            turn=turn,
            plan=plan,
            violations=violations,
        )

        if (
            plan.action
            is PatientActionKind.SELECT_OPTION
        ):
            self._validate_selected_option(
                world=world,
                turn=turn,
                plan=plan,
                violations=violations,
            )

        return tuple(
            violations
        )

    @staticmethod
    def _validate_action_matches_turn(
        *,
        turn: TurnFrame,
        plan: ActionPlan,
        violations: list[ConstraintViolation],
    ) -> None:

        action = plan.action

        if (
            turn.requested_action
            is RequestedAction.WAIT
            and action
            is not PatientActionKind.WAIT
        ):
            violations.append(
                ConstraintViolation(
                    code="must_wait",
                    detail=(
                        "Remote agent is still working; "
                        "caller should remain silent."
                    ),
                )
            )

        if (
            turn.requested_action
            is RequestedAction.ANSWER_FACT
            and action
            not in {
                PatientActionKind.ANSWER_FACT,
                PatientActionKind.CLARIFY,
            }
        ):
            violations.append(
                ConstraintViolation(
                    code="fact_request_requires_answer",
                    detail=(
                        "Remote agent requested caller facts."
                    ),
                )
            )

        if (
            turn.requested_action
            is RequestedAction.GRANT_PERMISSION
            and action
            not in {
                PatientActionKind.GRANT_PERMISSION,
                PatientActionKind.DECLINE_PERMISSION,
                PatientActionKind.CLARIFY,
            }
        ):
            violations.append(
                ConstraintViolation(
                    code="permission_requires_permission_action",
                    detail=(
                        "Turn requests permission, not slot selection."
                    ),
                )
            )

        if (
            turn.requested_action
            is RequestedAction.CHOOSE_OPTION
            and action
            not in {
                PatientActionKind.SELECT_OPTION,
                PatientActionKind.REQUEST_ALTERNATIVE,
                PatientActionKind.CLARIFY,
            }
        ):
            violations.append(
                ConstraintViolation(
                    code="choice_requires_explicit_choice_action",
                    detail=(
                        "Option-selection turn requires selecting a "
                        "specific compatible option, requesting an "
                        "alternative, or clarifying."
                    ),
                )
            )

    @staticmethod
    def _validate_selected_option(
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
        plan: ActionPlan,
        violations: list[ConstraintViolation],
    ) -> None:

        index = plan.selected_option_index

        if index is None:
            return

        if index >= len(
            turn.appointment_options
        ):
            violations.append(
                ConstraintViolation(
                    code="option_index_out_of_range",
                    detail=(
                        f"Option index {index} does not exist."
                    ),
                )
            )
            return

        option = (
            turn.appointment_options[
                index
            ]
        )

        for constraint in world.constraints:

            if (
                constraint.strength
                is not ConstraintStrength.HARD
            ):
                continue

            candidate = _option_value(
                option=option,
                field=constraint.field,
            )

            # A caller may not select an option when a hard constraint
            # cannot even be verified from the available structured state.
            if candidate is None:
                violations.append(
                    ConstraintViolation(
                        code="hard_constraint_unverified",
                        detail=(
                            f"Selected option does not establish "
                            f"{constraint.field!r}, required by "
                            f"{constraint.source!r}."
                        ),
                    )
                )
                continue

            if not _matches_constraint(
                constraint=constraint,
                candidate=candidate,
            ):
                violations.append(
                    ConstraintViolation(
                        code="hard_constraint_conflict",
                        detail=(
                            f"{constraint.field!r} candidate "
                            f"{candidate!r} conflicts with required "
                            f"value {constraint.value!r}."
                        ),
                    )
                )
