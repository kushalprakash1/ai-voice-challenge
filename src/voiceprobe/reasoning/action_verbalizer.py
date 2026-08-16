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
