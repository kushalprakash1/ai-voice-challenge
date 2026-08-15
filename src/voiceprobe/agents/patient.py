"""Patient-agent orchestration for VoiceProbe.

The patient agent coordinates immutable scenario facts, deterministic
conversation state, and a replaceable response planner. The planner may
reason about what to say, but it cannot directly mutate authoritative
call state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from voiceprobe.conversation.state import (
    PatientAction,
    PatientState,
    apply_patient_action,
    record_agent_turn,
)
from voiceprobe.scenarios.models import PatientScenario


class PatientPlanner(Protocol):
    """Interface implemented by a patient response planner."""

    def plan(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> PatientAction:
        """Choose the patient's next structured action."""
        ...


@dataclass(frozen=True, slots=True)
class PatientStep:
    """Result of processing one completed tested-agent turn."""

    heard_text: str
    action: PatientAction
    state: PatientState


class PatientAgent:
    """Coordinate one simulated patient conversation."""

    def __init__(
        self,
        *,
        scenario: PatientScenario,
        planner: PatientPlanner,
    ) -> None:
        self._scenario = scenario
        self._planner = planner

    @property
    def scenario(self) -> PatientScenario:
        """Return the immutable scenario assigned to this agent."""
        return self._scenario

    def respond(
        self,
        state: PatientState,
        agent_turn: str,
    ) -> PatientStep:
        """Process one tested-agent turn and produce a patient action."""
        if state.scenario_id != self._scenario.scenario_id:
            raise ValueError("PatientState does not belong to this agent's scenario.")

        if state.objective_complete:
            raise RuntimeError(
                "Cannot generate another patient response after "
                "the scenario objective is complete."
            )

        state_with_agent_turn = record_agent_turn(
            state,
            agent_turn,
        )

        action = self._planner.plan(
            scenario=self._scenario,
            state=state_with_agent_turn,
            agent_turn=state_with_agent_turn.messages[-1].text,
        )

        if not isinstance(action, PatientAction):
            raise TypeError("Patient planner must return a PatientAction.")

        updated_state = apply_patient_action(
            state_with_agent_turn,
            self._scenario,
            action,
        )

        return PatientStep(
            heard_text=state_with_agent_turn.messages[-1].text,
            action=action,
            state=updated_state,
        )
