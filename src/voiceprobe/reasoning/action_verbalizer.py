"""Safe natural realization of approved Reasoning Core actions.

This component does not reason about what the caller SHOULD do.

Semantic understanding and planning have already happened.

Its only job is to turn an approved ActionPlan plus authoritative world
state into concise caller speech without another language-model request.
"""

from __future__ import annotations

from voiceprobe.reasoning.action_plan import (
    ActionPlan,
    PatientActionKind,
)
from voiceprobe.reasoning.fact_grounding import (
    FactConflict,
)
from voiceprobe.reasoning.turn_frame import (
    RequestedFact,
    TurnFrame,
)
from voiceprobe.reasoning.world_model import (
    ConstraintStrength,
    PatientWorldModel,
)


_FACT_KEYS: dict[
    RequestedFact,
    str,
] = {
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


_ANY_PROVIDER_VALUES = {
    "any",
    "any provider",
    "any available provider",
    "no preference",
    "none",
    "whoever is available",
}


class GenericActionVerbalizer:
    """Convert policy-approved actions into safe caller speech."""

    def verbalize(
        self,
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
        plan: ActionPlan,
        corrections: tuple[FactConflict, ...] = (),
    ) -> str:
        """Realize corrections plus the approved primary action.

        Corrections are additive semantic events.

        A remote agent may assert a wrong caller fact AND ask a valid
        question in the same utterance. We therefore do not replace the
        primary action merely because a correction is necessary.
        """

        action = plan.action

        primary = self._primary_text(
            world=world,
            turn=turn,
            plan=plan,
        )

        correction_text = self._correction_text(
            world=world,
            corrections=corrections,
        )

        # ANSWER_FACT already realizes its payload as the primary text.
        # For multi-intent turns, the same fact payload is supplementary
        # to another primary conversational action.
        supplemental_fact_text = ""

        if (
            plan.action
            is not PatientActionKind.ANSWER_FACT
            and plan.facts_to_answer
        ):
            supplemental_fact_text = self._answer_facts(
                world=world,
                facts=plan.facts_to_answer,
            )

        pieces = [
            piece
            for piece in (
                correction_text,
                primary,
                supplemental_fact_text,
            )
            if piece
        ]

        return " ".join(
            pieces
        )

    def _primary_text(
        self,
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
        plan: ActionPlan,
    ) -> str:

        action = plan.action

        if action is PatientActionKind.WAIT:
            return ""

        if action is PatientActionKind.STATE_OBJECTIVE:
            return self._objective_text(
                world
            )

        if action is PatientActionKind.ANSWER_FACT:
            return self._answer_facts(
                world=world,
                facts=plan.facts_to_answer,
            )

        if action is PatientActionKind.GRANT_PERMISSION:
            return "Yes, please."

        if action is PatientActionKind.DECLINE_PERMISSION:
            return (
                "No, thank you. "
                "I'd like to continue with my request."
            )

        if action is PatientActionKind.SELECT_OPTION:
            return self._selected_option_text(
                turn=turn,
                plan=plan,
            )

        if action is PatientActionKind.SELECT_PRESENTED_CHOICE:
            return self._selected_presented_choice_text(
                world=world,
                turn=turn,
                plan=plan,
            )

        if action is PatientActionKind.REQUEST_ALTERNATIVE:
            return self._alternative_text(
                world
            )

        if action is PatientActionKind.CONFIRM:
            return "Yes, that's correct."

        if action is PatientActionKind.REJECT_CONFIRMATION:
            return "No, that's not correct."

        if action is PatientActionKind.CLARIFY:
            return "Could you clarify that?"

        if action is PatientActionKind.VERIFY_BOOKING:
            return (
                "Just to confirm, is my appointment booked?"
            )

        if action is PatientActionKind.END_CONVERSATION:
            return "Okay, thank you. Bye."

        raise ValueError(
            f"Unsupported patient action: {action}"
        )

    @staticmethod
    def _sentence(
        value: object,
    ) -> str:
        text = " ".join(
            str(value).split()
        )

        if not text:
            raise ValueError(
                "Cannot verbalize a blank value."
            )

        if text.endswith(
            (".", "?", "!")
        ):
            return text

        return f"{text}."

    @classmethod
    def _objective_text(
        cls,
        world: PatientWorldModel,
    ) -> str:
        objective = " ".join(
            world.objective.split()
        ).rstrip(".")

        if not objective:
            raise ValueError(
                "Patient objective cannot be blank."
            )

        lowered = objective.casefold()

        if lowered.startswith(
            (
                "i need ",
                "i want ",
                "i would like ",
            )
        ):
            return cls._sentence(
                objective
            )

        natural = (
            objective[0].lower()
            + objective[1:]
        )

        return cls._sentence(
            f"I need to {natural}"
        )

    @staticmethod
    def _fact_value(
        *,
        world: PatientWorldModel,
        fact: RequestedFact,
    ) -> object:

        key = _FACT_KEYS[
            fact
        ]

        if key not in world.facts:
            raise ValueError(
                f"Caller does not have requested fact {key!r}."
            )

        value = world.facts[
            key
        ]

        if value is None:
            raise ValueError(
                f"Caller fact {key!r} is unavailable."
            )

        return value

    @classmethod
    def _answer_facts(
        cls,
        *,
        world: PatientWorldModel,
        facts: list[RequestedFact],
    ) -> str:

        if not facts:
            raise ValueError(
                "ANSWER_FACT requires facts."
            )

        values = {
            fact: cls._fact_value(
                world=world,
                fact=fact,
            )
            for fact in facts
        }

        fact_set = set(
            facts
        )

        if fact_set == {
            RequestedFact.FIRST_NAME,
            RequestedFact.LAST_NAME,
        }:
            return cls._sentence(
                f"{values[RequestedFact.FIRST_NAME]} "
                f"{values[RequestedFact.LAST_NAME]}"
            )

        if fact_set == {
            RequestedFact.COMPLAINT,
            RequestedFact.SYMPTOM_DURATION,
        }:
            return cls._sentence(
                f"{values[RequestedFact.COMPLAINT]} "
                f"for "
                f"{values[RequestedFact.SYMPTOM_DURATION]}"
            )

        if fact_set == {
            RequestedFact.PROVIDER_PREFERENCE
        }:
            provider = str(
                values[
                    RequestedFact.PROVIDER_PREFERENCE
                ]
            )

            if (
                " ".join(
                    provider.casefold().split()
                )
                in _ANY_PROVIDER_VALUES
            ):
                return (
                    "I don't have a preference. "
                    "Any available provider is fine."
                )

        if fact_set == {
            RequestedFact.PATIENT_STATUS
        }:
            return cls._sentence(
                f"I'm "
                f"{values[RequestedFact.PATIENT_STATUS]}"
            )

        if fact_set == {
            RequestedFact.VISITED_BEFORE
        }:
            visited = values[
                RequestedFact.VISITED_BEFORE
            ]

            if not isinstance(
                visited,
                bool,
            ):
                raise ValueError(
                    "visited_before must be boolean."
                )

            return (
                "Yes, I've visited before."
                if visited
                else "No, I haven't visited before."
            )

        if fact_set == {
            RequestedFact.APPOINTMENT_TYPE
        }:
            return cls._sentence(
                f"I need "
                f"{values[RequestedFact.APPOINTMENT_TYPE]}"
            )

        if len(
            facts
        ) == 1:
            return cls._sentence(
                values[
                    facts[0]
                ]
            )

        return cls._sentence(
            ", ".join(
                str(
                    values[fact]
                )
                for fact
                in facts
            )
        )

    @classmethod
    def _correction_text(
        cls,
        *,
        world: PatientWorldModel,
        corrections: tuple[FactConflict, ...],
    ) -> str:
        """Render authoritative corrections without another model call."""

        if not corrections:
            return ""

        pieces: list[str] = []

        for conflict in corrections:

            fact = conflict.fact

            value = conflict.authoritative_value

            if fact is RequestedFact.DATE_OF_BIRTH:
                pieces.append(
                    f"my date of birth is {value}"
                )
                continue

            if fact is RequestedFact.INSURANCE:
                pieces.append(
                    f"my insurance is {value}"
                )
                continue

            if fact is RequestedFact.FIRST_NAME:
                pieces.append(
                    f"my first name is {value}"
                )
                continue

            if fact is RequestedFact.LAST_NAME:
                pieces.append(
                    f"my last name is {value}"
                )
                continue

            if fact is RequestedFact.FULL_NAME:
                pieces.append(
                    f"my name is {value}"
                )
                continue

            if fact is RequestedFact.APPOINTMENT_TYPE:
                pieces.append(
                    f"I need {value}"
                )
                continue

            if fact is RequestedFact.PROVIDER_PREFERENCE:
                pieces.append(
                    f"my provider preference is {value}"
                )
                continue

            if fact is RequestedFact.PATIENT_STATUS:
                pieces.append(
                    f"I'm {value}"
                )
                continue

            if fact is RequestedFact.PREFERRED_DAY:
                pieces.append(
                    f"I need {value}"
                )
                continue

            if fact is RequestedFact.PREFERRED_TIME:
                pieces.append(
                    f"I need {value}"
                )
                continue

            # Generic fallback still uses authoritative patient truth and
            # never repeats the remote agent's conflicting value.
            readable = (
                fact.value
                .replace("_", " ")
            )

            pieces.append(
                f"my {readable} is {value}"
            )

        if not pieces:
            return ""

        if len(pieces) == 1:
            body = pieces[0]
        else:
            body = ", and ".join(
                pieces
            )

        return cls._sentence(
            f"Actually, {body}"
        )

    @staticmethod
    def _selected_option_text(
        *,
        turn: TurnFrame,
        plan: ActionPlan,
    ) -> str:

        index = (
            plan.selected_option_index
        )

        if index is None:
            raise ValueError(
                "SELECT_OPTION has no option index."
            )

        if index >= len(
            turn.appointment_options
        ):
            raise ValueError(
                "SELECT_OPTION index is outside the offered options."
            )

        option = (
            turn.appointment_options[
                index
            ]
        )

        if (
            option.day is not None
            and option.time is not None
        ):
            return (
                f"{option.day} at "
                f"{option.time} works for me."
            )

        if option.time is not None:
            return (
                f"{option.time} works for me."
            )

        if (
            option.day is not None
            and option.daypart is not None
        ):
            return (
                f"{option.day} "
                f"{option.daypart} works for me."
            )

        if option.daypart is not None:
            return (
                f"{option.daypart.capitalize()} "
                "works for me."
            )

        if option.day is not None:
            return (
                f"{option.day} works for me."
            )

        raise ValueError(
            "Selected appointment option has no usable scheduling detail."
        )

    @staticmethod
    def _selected_presented_choice_text(
        *,
        world: PatientWorldModel,
        turn: TurnFrame,
        plan: ActionPlan,
    ) -> str:
        index = plan.selected_choice_index

        if index is None:
            raise ValueError(
                "SELECT_PRESENTED_CHOICE has no choice index."
            )

        if index >= len(
            turn.presented_choices
        ):
            raise ValueError(
                "SELECT_PRESENTED_CHOICE index is outside presented choices."
            )

        choice = turn.presented_choices[
            index
        ]

        if choice.kind.value == "search_availability":
            constraints = {
                item.field: item.value
                for item in world.constraints
                if (
                    item.strength
                    is ConstraintStrength.HARD
                )
            }

            search_time = (
                choice.time
                or choice.daypart
                or constraints.get("time")
            )

            # Prefer structured fields, but recover a weekday from the
            # source-grounded choice label when Qwen leaves choice.day null.
            # This mirrors the deterministic validator's source recovery and
            # never invents a day from patient preference.
            source_label = " ".join(
                choice.label.casefold().split()
            )

            source_tokens = set(
                source_label.replace(",", " ")
                .replace(".", " ")
                .replace("?", " ")
                .split()
            )

            recovered_day: str | None = None

            for weekday in (
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ):
                if weekday in source_tokens:
                    recovered_day = weekday.capitalize()
                    break

            explicit_day = (
                choice.day
                if choice.day is not None
                else recovered_day
            )

            if (
                explicit_day is not None
                and choice.date_text is not None
            ):
                when = (
                    f"{explicit_day}, "
                    f"{choice.date_text}"
                )
            elif choice.date_text is not None:
                when = choice.date_text
            elif explicit_day is not None:
                when = explicit_day
            else:
                when = constraints.get(
                    "day"
                )

            if (
                when is not None
                and search_time is not None
            ):
                return (
                    f"Please check {when} for "
                    f"{search_time} appointments."
                )

            if when is not None:
                return (
                    f"Please check {when} for appointments."
                )

            if search_time is not None:
                return (
                    f"Please check for {search_time} appointments."
                )

        label = choice.label.rstrip(
            ".?!"
        )

        if label.casefold().startswith(
            (
                "check ",
                "look ",
                "search ",
                "continue ",
                "create ",
                "verify ",
            )
        ):
            return (
                f"Please {label}."
            )

        return (
            f"I'd like to {label}."
        )

    @staticmethod
    def _alternative_text(
        world: PatientWorldModel,
    ) -> str:
        constraints = {
            item.field: item.value
            for item in world.constraints
            if (
                item.strength
                is ConstraintStrength.HARD
            )
        }

        day = constraints.get(
            "day"
        )

        time = constraints.get(
            "time"
        )

        provider = constraints.get(
            "provider"
        )

        pieces: list[str] = []

        if day is not None:
            pieces.append(
                day
            )

        if time is not None:
            pieces.append(
                time
            )

        core = " ".join(
            pieces
        )

        if provider is not None:
            provider_text = (
                f" with {provider}"
            )
        else:
            provider_text = ""

        if core:
            return (
                "Those options don't work for me. "
                f"Do you have anything {core}"
                f"{provider_text}?"
            )

        if provider is not None:
            return (
                "Those options don't work for me. "
                f"Do you have anything with {provider}?"
            )

        return (
            "Those options don't work for me. "
            "Do you have any other options?"
        )
