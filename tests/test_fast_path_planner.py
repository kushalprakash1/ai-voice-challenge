import pytest

from voiceprobe.conversation.state import (
    build_initial_state,
)
from voiceprobe.planners.fast_path import FastPathPatientPlanner
from voiceprobe.scenarios.models import PatientFacts, PatientScenario


@pytest.fixture
def scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="shoulder-friday",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            date_of_birth="January 12, 1998",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )


def test_answers_complaint_question(
    scenario: PatientScenario,
) -> None:
    planner = FastPathPatientPlanner()
    state = build_initial_state(scenario)

    action = planner.try_plan(
        scenario=scenario,
        state=state,
        agent_turn="What seems to be bothering you?",
    )

    assert action is not None
    assert action.response == "I've had right shoulder pain for about five days."
    assert action.facts_used == ("complaint", "duration")


def test_answers_duration_question(
    scenario: PatientScenario,
) -> None:
    planner = FastPathPatientPlanner()

    action = planner.try_plan(
        scenario=scenario,
        state=build_initial_state(scenario),
        agent_turn="How long has this been bothering you?",
    )

    assert action is not None
    assert action.response == "It's been about five days."
    assert action.facts_used == ("duration",)


def test_answers_insurance_question(
    scenario: PatientScenario,
) -> None:
    planner = FastPathPatientPlanner()

    action = planner.try_plan(
        scenario=scenario,
        state=build_initial_state(scenario),
        agent_turn="What insurance do you have?",
    )

    assert action is not None
    assert action.response == "I have Blue Cross."
    assert action.facts_used == ("insurance",)


def test_answers_name_question(
    scenario: PatientScenario,
) -> None:
    planner = FastPathPatientPlanner()

    action = planner.try_plan(
        scenario=scenario,
        state=build_initial_state(scenario),
        agent_turn="Can I get your name please?",
    )

    assert action is not None
    assert action.response == "My name is Alex Morgan."


def test_answers_schedule_question(
    scenario: PatientScenario,
) -> None:
    planner = FastPathPatientPlanner()

    action = planner.try_plan(
        scenario=scenario,
        state=build_initial_state(scenario),
        agent_turn="When would you like to come in?",
    )

    assert action is not None
    assert action.response == "Friday afternoon would work best for me."
    assert action.facts_used == (
        "preferred_day",
        "preferred_time",
    )


def test_unknown_turn_falls_through(
    scenario: PatientScenario,
) -> None:
    planner = FastPathPatientPlanner()

    action = planner.try_plan(
        scenario=scenario,
        state=build_initial_state(scenario),
        agent_turn=(
            "I think I heard something different earlier. Could you explain that again?"
        ),
    )

    assert action is None


def test_rejects_state_from_different_scenario(
    scenario: PatientScenario,
) -> None:
    other = PatientScenario(
        scenario_id="other",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Jordan Lee",
            complaint="ankle pain",
            duration="three days",
        ),
    )

    planner = FastPathPatientPlanner()

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        planner.try_plan(
            scenario=scenario,
            state=build_initial_state(other),
            agent_turn="What is your name?",
        )
