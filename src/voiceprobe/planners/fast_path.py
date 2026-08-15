"""Low-latency patient decisions for common scheduling questions.

The fast path handles conversational turns whose answers can be derived
directly from authoritative scenario facts. Ambiguous turns return None
so a higher-level planner can decide whether to invoke an LLM.
"""

from __future__ import annotations

from voiceprobe.conversation.state import (
    ActionKind,
    PatientAction,
    PatientState,
)
from voiceprobe.scenarios.models import PatientScenario


class FastPathPatientPlanner:
    """Resolve high-confidence patient turns without model inference."""

    def try_plan(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> PatientAction | None:
        """Return a grounded action when the intent is unambiguous."""
        if state.scenario_id != scenario.scenario_id:
            raise ValueError("PatientState does not belong to the supplied scenario.")

        text = " ".join(agent_turn.lower().split())

        if self._asks_for_name(text):
            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"My name is {scenario.facts.name}.",
                facts_used=("name",),
            )

        if self._asks_for_date_of_birth(text):
            date_of_birth = scenario.facts.date_of_birth

            if date_of_birth is None:
                return None

            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"My date of birth is {date_of_birth}.",
                facts_used=("date_of_birth",),
            )

        if self._asks_for_insurance(text):
            insurance = scenario.facts.insurance

            if insurance is None:
                return None

            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"I have {insurance}.",
                facts_used=("insurance",),
            )

        if self._asks_for_duration(text):
            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"It's been about {scenario.facts.duration}.",
                facts_used=("duration",),
            )

        if self._asks_about_complaint(text):
            return PatientAction(
                kind=ActionKind.ANSWER,
                response=(
                    f"I've had {scenario.facts.complaint} "
                    f"for about {scenario.facts.duration}."
                ),
                facts_used=("complaint", "duration"),
            )

        if self._asks_for_schedule(text):
            return self._schedule_response(scenario)

        return None

    @staticmethod
    def _contains_any(
        text: str,
        phrases: tuple[str, ...],
    ) -> bool:
        return any(phrase in text for phrase in phrases)

    def _asks_for_name(self, text: str) -> bool:
        return self._contains_any(
            text,
            (
                "your name",
                "name please",
                "who am i speaking",
                "who is this",
                "who am i talking",
            ),
        )

    def _asks_for_date_of_birth(self, text: str) -> bool:
        return self._contains_any(
            text,
            (
                "date of birth",
                "birthday",
                "dob",
            ),
        )

    def _asks_for_insurance(self, text: str) -> bool:
        return self._contains_any(
            text,
            (
                "insurance",
                "coverage",
                "insurance provider",
                "insurance company",
            ),
        )

    def _asks_for_duration(self, text: str) -> bool:
        return self._contains_any(
            text,
            (
                "how long",
                "when did it start",
                "when did this start",
                "how many days",
                "how many weeks",
            ),
        )

    def _asks_about_complaint(self, text: str) -> bool:
        return self._contains_any(
            text,
            (
                "what seems to be bothering",
                "what is bothering",
                "what's bothering",
                "what brings you in",
                "reason for the appointment",
                "reason for your appointment",
                "what hurts",
                "what are you experiencing",
            ),
        )

    def _asks_for_schedule(self, text: str) -> bool:
        return self._contains_any(
            text,
            (
                "when would you like",
                "what day",
                "which day",
                "what time",
                "morning or afternoon",
                "when works",
                "what works for you",
                "appointment time",
                "appointment day",
            ),
        )

    @staticmethod
    def _schedule_response(
        scenario: PatientScenario,
    ) -> PatientAction | None:
        preferred_day = scenario.facts.preferred_day
        preferred_time = scenario.facts.preferred_time

        if preferred_day is not None and preferred_time is not None:
            return PatientAction(
                kind=ActionKind.ANSWER,
                response=(f"{preferred_day} {preferred_time} would work best for me."),
                facts_used=(
                    "preferred_day",
                    "preferred_time",
                ),
            )

        if preferred_day is not None:
            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"{preferred_day} would work best for me.",
                facts_used=("preferred_day",),
            )

        if preferred_time is not None:
            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"{preferred_time} would work best for me.",
                facts_used=("preferred_time",),
            )

        return None
