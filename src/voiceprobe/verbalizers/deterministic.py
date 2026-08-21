"""Deterministic speech realization for production patient calls.

PatientBrain has already made the semantic decision before this component
runs. For structured scheduling actions there is no benefit in asking a
language model to rediscover what to say. Deterministic realization keeps
approved facts authoritative, prevents hallucinated scheduling details,
and removes model latency from the response path.
"""

from __future__ import annotations

from typing import Any

from voiceprobe.agents.brain import (
    CommunicationDecision,
    CommunicationKind,
)
from voiceprobe.conversation.state import PatientState, Speaker
from voiceprobe.scenarios.models import PatientScenario


class DeterministicNaturalVerbalizer:
    """Realize structured patient decisions without model inference.

    Legacy constructor arguments are accepted intentionally so this class
    can replace OllamaNaturalVerbalizer at existing production call sites
    without coupling migration safety to unrelated CLI plumbing.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        url: str | None = None,
        timeout_seconds: float | None = None,
        client: Any | None = None,
    ) -> None:
        del model, url, timeout_seconds, client

    def close(self) -> None:
        """No resources are owned by the deterministic verbalizer."""

    def verbalize(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        decision: CommunicationDecision,
    ) -> str:
        """Return safe natural speech for one authoritative brain decision."""
        if state.scenario_id != scenario.scenario_id:
            raise ValueError(
                "PatientState does not belong to the supplied scenario."
            )

        kind = decision.kind

        if decision.state_objective:
            return self._objective_text(
                scenario=scenario,
            )

        if kind is CommunicationKind.ANSWER:
            return self._answer_text(
                scenario=scenario,
                decision=decision,
            )

        if kind is CommunicationKind.CORRECT:
            answer = self._answer_text(
                scenario=scenario,
                decision=decision,
            ).rstrip(".")

            return f"Actually, {answer}."

        if kind is CommunicationKind.ACCEPT_OFFER:
            return "Yes, that works."

        if kind is CommunicationKind.ACCEPT_PARTIAL_OFFER:
            if (
                decision.offered_day is not None
                and decision.offered_time is None
            ):
                return (
                    f"{decision.offered_day} works. "
                    "What time is that?"
                )

            if (
                decision.offered_time is not None
                and decision.offered_day is None
            ):
                return (
                    f"{decision.offered_time} works. "
                    "What day is that?"
                )

            raise ValueError(
                "Partial offer requires exactly one slot component."
            )

        if kind is CommunicationKind.DECLINE_OFFER:
            preference = self._preference_text(
                scenario=scenario,
                decision=decision,
            )

            if preference:
                return f"No, I need {preference}."

            return "No, that doesn't work for me."

        if kind is CommunicationKind.REPEAT:
            previous = self._previous_patient_message(state)

            if previous is None:
                return "Could you repeat that?"

            return previous

        if kind is CommunicationKind.ACKNOWLEDGE_COMPLETE:
            return "Okay, great."

        if kind is CommunicationKind.END_CONVERSATION:
            return "Okay, thank you. Bye."

        if kind is CommunicationKind.ASK_AGENT_TO_REPEAT:
            return "Could you repeat that?"

        if kind is CommunicationKind.VERIFY_BOOKING:
            day = decision.offered_day
            time = decision.offered_time

            if day is not None and time is not None:
                return (
                    "Just to confirm, am I booked for "
                    f"{day} at {time}?"
                )

            if day is not None:
                return (
                    "Just to confirm, is my "
                    f"{day} appointment booked?"
                )

            if time is not None:
                return (
                    "Just to confirm, is my "
                    f"{time} appointment booked?"
                )

            return "Just to confirm, is my appointment booked?"

        if kind is CommunicationKind.AGREE:
            return "Yes, please."

        if kind is CommunicationKind.DECLINE_WORKFLOW:
            return "No, I need an appointment."

        if kind is CommunicationKind.CLARIFY:
            return "Could you clarify that?"

        if kind is CommunicationKind.WAIT:
            raise ValueError(
                "WAIT must be handled by PatientSession without verbalization."
            )

        raise ValueError(
            f"Unsupported communication kind: {decision.kind}"
        )

    @staticmethod
    def _objective_text(
        *,
        scenario: PatientScenario,
    ) -> str:
        """State the scheduling mission using authoritative preferences."""
        day = scenario.facts.preferred_day
        preferred_time = scenario.facts.preferred_time

        if day is not None and preferred_time is not None:
            return (
                "I need to schedule an appointment for "
                f"{day} {preferred_time}."
            )

        if day is not None:
            return (
                "I need to schedule an appointment for "
                f"{day}."
            )

        if preferred_time is not None:
            return (
                "I need to schedule an appointment for "
                f"{preferred_time}."
            )

        return "I need to schedule an appointment."

    @staticmethod
    def _fact_value(
        *,
        scenario: PatientScenario,
        fact_key: str,
    ) -> str:
        value = getattr(
            scenario.facts,
            fact_key,
        )

        if value is None:
            raise ValueError(
                f"PatientBrain approved an unavailable fact: {fact_key}"
            )

        return str(value)

    @classmethod
    def _answer_text(
        cls,
        *,
        scenario: PatientScenario,
        decision: CommunicationDecision,
    ) -> str:
        keys = tuple(decision.facts_to_communicate)

        if not keys:
            raise ValueError(
                "ANSWER/CORRECT decision requires approved patient facts."
            )

        values = {
            key: cls._fact_value(
                scenario=scenario,
                fact_key=key,
            )
            for key in keys
        }

        key_set = set(keys)

        adaptive_persona_keys = {
            "patient_status",
            "visited_before",
            "appointment_type",
        }

        if key_set and key_set <= adaptive_persona_keys:
            sentences: list[str] = []

            if "patient_status" in key_set:
                sentences.append(
                    cls._sentence(
                        f"I'm {values['patient_status']}"
                    )
                )

            if "visited_before" in key_set:
                if scenario.facts.visited_before is True:
                    sentences.append(
                        "Yes, I've visited before."
                    )
                else:
                    sentences.append(
                        "No, I haven't visited before."
                    )

            if "appointment_type" in key_set:
                sentences.append(
                    cls._sentence(
                        f"I need {values['appointment_type']}"
                    )
                )

            return " ".join(sentences)

        if key_set == {"first_name", "last_name"}:
            return cls._sentence(
                f"{values['first_name']} {values['last_name']}"
            )

        if key_set == {"provider_preference"}:
            return (
                "I don't have a preference. "
                "Any available provider is fine."
            )

        if key_set == {"complaint", "duration"}:
            return cls._sentence(
                f"{values['complaint']} for {values['duration']}"
            )

        if key_set == {"preferred_day", "preferred_time"}:
            return cls._sentence(
                f"{values['preferred_day']} "
                f"{values['preferred_time']}"
            )

        if len(keys) == 1:
            return cls._sentence(values[keys[0]])

        ordered_values = [
            values[key]
            for key in keys
        ]

        return cls._sentence(", ".join(ordered_values))

    @classmethod
    def _preference_text(
        cls,
        *,
        scenario: PatientScenario,
        decision: CommunicationDecision,
    ) -> str:
        keys = set(decision.facts_to_communicate)

        pieces: list[str] = []

        if "preferred_day" in keys:
            pieces.append(
                cls._fact_value(
                    scenario=scenario,
                    fact_key="preferred_day",
                )
            )

        if "preferred_time" in keys:
            pieces.append(
                cls._fact_value(
                    scenario=scenario,
                    fact_key="preferred_time",
                )
            )

        return " ".join(pieces)

    @staticmethod
    def _previous_patient_message(
        state: PatientState,
    ) -> str | None:
        for message in reversed(state.messages):
            if message.speaker is Speaker.PATIENT:
                return message.text

        return None

    @staticmethod
    def _sentence(text: str) -> str:
        normalized = " ".join(text.split())

        if not normalized:
            raise ValueError("Patient response cannot be blank.")

        if normalized.endswith((".", "?", "!")):
            return normalized

        return f"{normalized}."
