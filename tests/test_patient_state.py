import pytest

from voiceprobe.conversation.state import (
    ActionKind,
    PatientAction,
    Speaker,
    apply_patient_action,
    build_initial_state,
    record_agent_turn,
)
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
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )


def test_initial_state_is_empty(
    scenario: PatientScenario,
) -> None:
    state = build_initial_state(scenario)

    assert state.scenario_id == "shoulder-friday"
    assert state.messages == ()
    assert state.answered_facts == frozenset()
    assert not state.objective_complete


def test_records_agent_turn(
    scenario: PatientScenario,
) -> None:
    state = build_initial_state(scenario)

    state = record_agent_turn(
        state,
        "  What   seems to be the problem? ",
    )

    assert len(state.messages) == 1
    assert state.messages[0].speaker is Speaker.AGENT
    assert state.messages[0].text == "What seems to be the problem?"
    assert state.agent_turn_count == 1


def test_applies_patient_answer(
    scenario: PatientScenario,
) -> None:
    state = build_initial_state(scenario)

    action = PatientAction(
        kind=ActionKind.ANSWER,
        response=("My right shoulder has been hurting for about five days."),
        facts_used=("complaint", "duration"),
    )

    state = apply_patient_action(
        state,
        scenario,
        action,
    )

    assert state.patient_turn_count == 1
    assert state.messages[-1].speaker is Speaker.PATIENT
    assert state.answered_facts == frozenset({"complaint", "duration"})


def test_rejects_use_of_unavailable_fact() -> None:
    scenario = PatientScenario(
        scenario_id="no-insurance",
        objective="Ask about appointment availability.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
        ),
    )

    state = build_initial_state(scenario)

    action = PatientAction(
        kind=ActionKind.ANSWER,
        response="I have Blue Cross.",
        facts_used=("insurance",),
    )

    with pytest.raises(
        ValueError,
        match="unavailable fact",
    ):
        apply_patient_action(
            state,
            scenario,
            action,
        )


def test_records_correction(
    scenario: PatientScenario,
) -> None:
    state = build_initial_state(scenario)

    action = PatientAction(
        kind=ActionKind.CORRECT,
        response=("No, it is my right shoulder, not my knee."),
        facts_used=("complaint",),
        corrected_claim="knee pain",
    )

    state = apply_patient_action(
        state,
        scenario,
        action,
    )

    assert len(state.corrections) == 1
    assert state.corrections[0].claim == "knee pain"
    assert "right shoulder" in state.corrections[0].response


def test_correction_requires_claim() -> None:
    with pytest.raises(
        ValueError,
        match="requires corrected_claim",
    ):
        PatientAction(
            kind=ActionKind.CORRECT,
            response="No, that is incorrect.",
        )


def test_non_correction_rejects_corrected_claim() -> None:
    with pytest.raises(
        ValueError,
        match="only valid for correction",
    ):
        PatientAction(
            kind=ActionKind.ANSWER,
            response="Five days.",
            corrected_claim="two weeks",
        )


def test_complete_action_marks_objective_complete(
    scenario: PatientScenario,
) -> None:
    state = build_initial_state(scenario)

    action = PatientAction(
        kind=ActionKind.COMPLETE,
        response="Great, Friday afternoon works for me.",
        facts_used=(
            "preferred_day",
            "preferred_time",
        ),
    )

    state = apply_patient_action(
        state,
        scenario,
        action,
    )

    assert state.objective_complete


def test_state_rejects_wrong_scenario(
    scenario: PatientScenario,
) -> None:
    state = build_initial_state(scenario)

    other = PatientScenario(
        scenario_id="different",
        objective="Schedule another appointment.",
        facts=PatientFacts(
            name="Jordan Lee",
            complaint="ankle pain",
            duration="three days",
        ),
    )

    action = PatientAction(
        kind=ActionKind.ANSWER,
        response="My shoulder hurts.",
        facts_used=("complaint",),
    )

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        apply_patient_action(
            state,
            other,
            action,
        )
