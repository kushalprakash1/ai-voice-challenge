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
    PresentedChoice,
    RequestedAction,
    RequestedFact,
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


def _presented_choice_value(
    *,
    choice: PresentedChoice,
    field: str,
) -> str | None:
    # Recover only explicit, source-grounded constraints from the choice label.
    # This never uses patient preferences to fill missing choice data.

    label = " ".join(
        choice.label.casefold().split()
    )

    if field == "day":
        if choice.day is not None:
            return choice.day

        if any(
            phrase in label
            for phrase in (
                "another day",
                "other day",
                "different day",
            )
        ):
            return "another day"

        weekdays = (
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        )

        label_tokens = set(
            label.replace(",", " ")
            .replace(".", " ")
            .replace("?", " ")
            .split()
        )

        for weekday in weekdays:
            if weekday in label_tokens:
                return weekday.capitalize()

        return None

    if field == "time":
        if choice.time is not None:
            return choice.time

        if choice.daypart is not None:
            return choice.daypart

        label_tokens = set(
            label.replace(",", " ")
            .replace(".", " ")
            .replace("?", " ")
            .split()
        )

        for daypart in (
            "morning",
            "afternoon",
            "evening",
        ):
            if daypart in label_tokens:
                return daypart

        return None

    if field == "provider":
        return choice.provider

    if field == "appointment_type":
        return choice.appointment_type

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

    def compatible_presented_choice_indices(
        self,
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
    ) -> tuple[int, ...]:
        if (
            turn.requested_action
            is not RequestedAction.CHOOSE_PRESENTED_CHOICE
        ):
            return ()

        compatible: list[int] = []

        for index in range(
            len(turn.presented_choices)
        ):
            probe = ActionPlan(
                action=PatientActionKind.SELECT_PRESENTED_CHOICE,
                selected_choice_index=index,
                reason_code="presented_choice_compatibility_probe",
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

        self._validate_fact_payload(
            world=world,
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

        if (
            plan.action
            is PatientActionKind.SELECT_PRESENTED_CHOICE
        ):
            self._validate_selected_presented_choice(
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

        if (
            action
            is PatientActionKind.SELECT_OPTION
            and turn.requested_action
            is not RequestedAction.CHOOSE_OPTION
        ):
            violations.append(
                ConstraintViolation(
                    code="selection_without_choice_request",
                    detail=(
                        "Caller may select an appointment option only "
                        "when the remote turn actually requests a choice."
                    ),
                )
            )

        if (
            turn.requested_action
            is RequestedAction.CHOOSE_PRESENTED_CHOICE
            and action
            not in {
                PatientActionKind.SELECT_PRESENTED_CHOICE,
                PatientActionKind.REQUEST_ALTERNATIVE,
                PatientActionKind.CLARIFY,
            }
        ):
            violations.append(
                ConstraintViolation(
                    code="presented_choice_requires_explicit_choice_action",
                    detail=(
                        "General choice requires a specific compatible branch, "
                        "an alternative request, or clarification."
                    ),
                )
            )

        if (
            action
            is PatientActionKind.SELECT_PRESENTED_CHOICE
            and turn.requested_action
            is not RequestedAction.CHOOSE_PRESENTED_CHOICE
        ):
            violations.append(
                ConstraintViolation(
                    code="presented_selection_without_choice_request",
                    detail=(
                        "Caller may select a presented branch only when "
                        "the remote turn requests that general choice."
                    ),
                )
            )

    @staticmethod
    def _validate_fact_payload(
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
        plan: ActionPlan,
        violations: list[ConstraintViolation],
    ) -> None:
        """Prevent unrequested patient-data disclosure.

        FULL_NAME is allowed to satisfy a joint first_name + last_name
        request, but no other semantic substitution is permitted here.
        """

        if not plan.facts_to_answer:
            return

        requested = set(
            turn.requested_facts
        )

        allowed = set(
            requested
        )

        if {
            RequestedFact.FIRST_NAME,
            RequestedFact.LAST_NAME,
        } <= requested:
            allowed.add(
                RequestedFact.FULL_NAME
            )

        unexpected = [
            fact
            for fact in plan.facts_to_answer
            if fact not in allowed
        ]

        fact_keys = {
            RequestedFact.FIRST_NAME: "first_name",
            RequestedFact.LAST_NAME: "last_name",
            RequestedFact.FULL_NAME: "name",
            RequestedFact.DATE_OF_BIRTH: "date_of_birth",
            RequestedFact.INSURANCE: "insurance",
            RequestedFact.COMPLAINT: "complaint",
            RequestedFact.SYMPTOM_DURATION: "duration",
            RequestedFact.PREFERRED_DAY: "preferred_day",
            RequestedFact.PREFERRED_TIME: "preferred_time",
            RequestedFact.PROVIDER_PREFERENCE: "provider_preference",
            RequestedFact.APPOINTMENT_TYPE: "appointment_type",
            RequestedFact.PATIENT_STATUS: "patient_status",
            RequestedFact.VISITED_BEFORE: "visited_before",
            RequestedFact.PHONE_NUMBER: "phone_number",
            RequestedFact.EMAIL: "email",
            RequestedFact.ADDRESS: "address",
        }

        unavailable = [
            fact
            for fact in plan.facts_to_answer
            if (
                fact_keys[fact]
                not in world.facts
                or world.facts[
                    fact_keys[fact]
                ]
                is None
            )
        ]

        if unavailable:
            violations.append(
                ConstraintViolation(
                    code="unavailable_fact_disclosure",
                    detail=(
                        "Plan attempted to disclose caller facts "
                        "that are not present in authoritative truth: "
                        + ", ".join(
                            fact.value
                            for fact in unavailable
                        )
                    ),
                )
            )

        if unexpected:
            violations.append(
                ConstraintViolation(
                    code="unrequested_fact_disclosure",
                    detail=(
                        "Plan attempted to disclose caller facts that "
                        "the remote agent did not request: "
                        + ", ".join(
                            fact.value
                            for fact in unexpected
                        )
                    ),
                )
            )

        if (
            plan.action
            in {
                PatientActionKind.WAIT,
                PatientActionKind.DECLINE_PERMISSION,
                PatientActionKind.END_CONVERSATION,
            }
        ):
            violations.append(
                ConstraintViolation(
                    code="fact_disclosure_with_nonresponsive_action",
                    detail=(
                        "This action must not disclose requested "
                        "caller facts."
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

    @staticmethod
    def _validate_selected_presented_choice(
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
        plan: ActionPlan,
        violations: list[ConstraintViolation],
    ) -> None:
        index = plan.selected_choice_index

        if index is None:
            return

        if index >= len(
            turn.presented_choices
        ):
            violations.append(
                ConstraintViolation(
                    code="presented_choice_index_out_of_range",
                    detail=(
                        f"Presented choice index {index} does not exist."
                    ),
                )
            )
            return

        choice = turn.presented_choices[
            index
        ]

        for constraint in world.constraints:
            if (
                constraint.strength
                is not ConstraintStrength.HARD
            ):
                continue

            candidate = _presented_choice_value(
                choice=choice,
                field=constraint.field,
            )

            # Missing search detail may be carried forward by the caller.
            if candidate is None:
                continue

            if not _matches_constraint(
                constraint=constraint,
                candidate=candidate,
            ):
                violations.append(
                    ConstraintViolation(
                        code="hard_constraint_conflict",
                        detail=(
                            f"Presented choice {index} explicitly sets "
                            f"{constraint.field!r} to {candidate!r}, "
                            f"conflicting with required value "
                            f"{constraint.value!r}."
                        ),
                    )
                )
