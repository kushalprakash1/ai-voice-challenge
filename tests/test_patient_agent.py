import pytest

from voiceprobe.agents.patient import PatientAgent
from voiceprobe.conversation.state import (
    ActionKind,
    PatientAction,
    PatientState,
    build_initial_state,
)
from voiceprobe.scenarios.models import PatientFacts, PatientScenario


class StubPlanner:
    """Deterministic planner used to test agent orchestration."""

    def __init__(self, action: PatientAction) -> None:
        self.action = action
        self.received_turn: str | None = None
        self.received_state: PatientState | None = None

    def plan(
        self,
        *,
        scenario: PatientScenario,
        state: PatientState,
        agent_turn: str,
    ) -> PatientAction:
        self.received_turn = agent_turn
        self.received_state = state
        return self.action


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="shoulder-friday",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )


def test_agent_processes_turn_and_updates_state() -> None:
    scenario = build_scenario()

    planner = StubPlanner(
        PatientAction(
            kind=ActionKind.ANSWER,
            response=("My right shoulder has been hurting for about five days."),
            facts_used=("complaint", "duration"),
        )
    )

    agent = PatientAgent(
        scenario=scenario,
        planner=planner,
    )

    state = build_initial_state(scenario)

    step = agent.respond(
        state,
        "What seems to be bothering you?",
    )

    assert step.heard_text == "What seems to be bothering you?"
    assert step.action.kind is ActionKind.ANSWER
    assert step.state.agent_turn_count == 1
    assert step.state.patient_turn_count == 1
    assert step.state.answered_facts == frozenset({"complaint", "duration"})


def test_planner_receives_recorded_agent_turn() -> None:
    scenario = build_scenario()

    planner = StubPlanner(
        PatientAction(
            kind=ActionKind.ANSWER,
            response="About five days.",
            facts_used=("duration",),
        )
    )

    agent = PatientAgent(
        scenario=scenario,
        planner=planner,
    )

    state = build_initial_state(scenario)

    agent.respond(
        state,
        "  How   long has it hurt?  ",
    )

    assert planner.received_turn == "How long has it hurt?"
    assert planner.received_state is not None
    assert planner.received_state.agent_turn_count == 1


def test_agent_rejects_state_from_another_scenario() -> None:
    scenario = build_scenario()

    other = PatientScenario(
        scenario_id="ankle-monday",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Jordan Lee",
            complaint="left ankle pain",
            duration="three days",
        ),
    )

    planner = StubPlanner(
        PatientAction(
            kind=ActionKind.ANSWER,
            response="My ankle hurts.",
            facts_used=("complaint",),
        )
    )

    agent = PatientAgent(
        scenario=scenario,
        planner=planner,
    )

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        agent.respond(
            build_initial_state(other),
            "What is bothering you?",
        )


def test_agent_stops_after_objective_completion() -> None:
    scenario = build_scenario()

    planner = StubPlanner(
        PatientAction(
            kind=ActionKind.COMPLETE,
            response="Friday afternoon works for me.",
            facts_used=(
                "preferred_day",
                "preferred_time",
            ),
        )
    )

    agent = PatientAgent(
        scenario=scenario,
        planner=planner,
    )

    first_step = agent.respond(
        build_initial_state(scenario),
        "I can schedule you Friday afternoon.",
    )

    assert first_step.state.objective_complete

    with pytest.raises(
        RuntimeError,
        match="objective is complete",
    ):
        agent.respond(
            first_step.state,
            "Anything else?",
        )
