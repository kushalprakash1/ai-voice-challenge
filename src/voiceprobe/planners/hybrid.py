"""Hybrid patient planner for VoiceProbe.

Common scheduling questions use the deterministic fast path. Ambiguous
language falls through to a constrained selector that chooses a small
response plan. Python, not the model, turns that plan into grounded
patient speech.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from voiceprobe.conversation.state import (
    ActionKind,
    PatientAction,
    PatientState,
)
from voiceprobe.planners.fast_path import FastPathPatientPlanner
from voiceprobe.scenarios.models import PatientScenario


class ResponsePlan(StrEnum):
    """Small set of actions available to the fallback selector."""

    ANSWER_NAME = "answer_name"
    ANSWER_COMPLAINT = "answer_complaint"
    ANSWER_DURATION = "answer_duration"
    ANSWER_COMPLAINT_DURATION = "answer_complaint_duration"
    ANSWER_DATE_OF_BIRTH = "answer_date_of_birth"
    ANSWER_INSURANCE = "answer_insurance"
    ANSWER_SCHEDULE = "answer_schedule"
    CORRECT_COMPLAINT_DURATION = "correct_complaint_duration"
    CLARIFY = "clarify"
    PROBE = "probe"


class SelectorDecision(BaseModel):
    """Minimal structured output produced by the fallback model."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    plan: ResponsePlan


class PatientActionSelector(Protocol):
    """Interface for an ambiguous-turn classifier."""

    def select(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> SelectorDecision:
        """Choose one constrained response plan."""
        ...


class HybridPatientPlanner:
    """Use deterministic answers first and model reasoning only as fallback."""

    def __init__(
        self,
        *,
        selector: PatientActionSelector,
    ) -> None:
        self._fast_path = FastPathPatientPlanner()
        self._selector = selector

    def plan(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> PatientAction:
        """Produce one grounded patient action."""
        fast_action = self._fast_path.try_plan(
            scenario=scenario,
            state=state,
            agent_turn=agent_turn,
        )

        if fast_action is not None:
            return fast_action

        decision = self._selector.select(
            scenario=scenario,
            state=state,
            agent_turn=agent_turn,
        )

        return self._render_decision(
            decision=decision,
            scenario=scenario,
            agent_turn=agent_turn,
        )

    @staticmethod
    def _render_decision(
        *,
        decision: SelectorDecision,
        scenario: PatientScenario,
        agent_turn: str,
    ) -> PatientAction:
        facts = scenario.facts
        plan = decision.plan

        if plan is ResponsePlan.ANSWER_NAME:
            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"My name is {facts.name}.",
                facts_used=("name",),
            )

        if plan is ResponsePlan.ANSWER_COMPLAINT:
            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"It's {facts.complaint}.",
                facts_used=("complaint",),
            )

        if plan is ResponsePlan.ANSWER_DURATION:
            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"It's been about {facts.duration}.",
                facts_used=("duration",),
            )

        if plan is ResponsePlan.ANSWER_COMPLAINT_DURATION:
            return PatientAction(
                kind=ActionKind.ANSWER,
                response=(f"I've had {facts.complaint} for about {facts.duration}."),
                facts_used=("complaint", "duration"),
            )

        if plan is ResponsePlan.ANSWER_DATE_OF_BIRTH:
            if facts.date_of_birth is None:
                return PatientAction(
                    kind=ActionKind.CLARIFY,
                    response="Could you clarify what information you need?",
                )

            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"My date of birth is {facts.date_of_birth}.",
                facts_used=("date_of_birth",),
            )

        if plan is ResponsePlan.ANSWER_INSURANCE:
            if facts.insurance is None:
                return PatientAction(
                    kind=ActionKind.CLARIFY,
                    response="Could you clarify what information you need?",
                )

            return PatientAction(
                kind=ActionKind.ANSWER,
                response=f"I have {facts.insurance}.",
                facts_used=("insurance",),
            )

        if plan is ResponsePlan.ANSWER_SCHEDULE:
            if facts.preferred_day is not None and facts.preferred_time is not None:
                return PatientAction(
                    kind=ActionKind.ANSWER,
                    response=(
                        f"{facts.preferred_day} {facts.preferred_time} "
                        "would work best for me."
                    ),
                    facts_used=(
                        "preferred_day",
                        "preferred_time",
                    ),
                )

            return PatientAction(
                kind=ActionKind.CLARIFY,
                response="What appointment times are available?",
            )

        if plan is ResponsePlan.CORRECT_COMPLAINT_DURATION:
            return PatientAction(
                kind=ActionKind.CORRECT,
                response=(
                    f"No, it's {facts.complaint}, and it's been about {facts.duration}."
                ),
                facts_used=("complaint", "duration"),
                corrected_claim=agent_turn,
            )

        if plan is ResponsePlan.PROBE:
            return PatientAction(
                kind=ActionKind.PROBE,
                response="Could you explain that a little more?",
            )

        return PatientAction(
            kind=ActionKind.CLARIFY,
            response="Sorry, could you say that again?",
        )
