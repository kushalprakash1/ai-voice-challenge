from voiceprobe.conversation.grounding import (
    ground_turn_meaning,
)
from voiceprobe.conversation.meaning import (
    FactAssertion,
    TurnMeaning,
)
from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
)


def build_scenario() -> PatientScenario:
    return PatientScenario(
        scenario_id="grounding-equivalence",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            insurance="Blue Cross",
        ),
    )


def test_insurance_suffix_does_not_create_false_conflict() -> None:
    grounded = ground_turn_meaning(
        scenario=build_scenario(),
        meaning=TurnMeaning(
            stated_facts=(
                FactAssertion(
                    fact="insurance",
                    value="Blue Cross insurance",
                ),
            ),
        ),
    )

    assert grounded.conflicts == ()


def test_different_insurer_still_creates_conflict() -> None:
    grounded = ground_turn_meaning(
        scenario=build_scenario(),
        meaning=TurnMeaning(
            stated_facts=(
                FactAssertion(
                    fact="insurance",
                    value="Aetna",
                ),
            ),
        ),
    )

    assert len(grounded.conflicts) == 1
    assert grounded.conflicts[0].fact == "insurance"


def test_afternoon_preference_matches_230_pm_statement() -> None:
    scenario = PatientScenario(
        scenario_id="time-equivalence",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )

    grounded = ground_turn_meaning(
        scenario=scenario,
        meaning=TurnMeaning(
            stated_facts=(
                FactAssertion(
                    fact="preferred_time",
                    value="2:30 PM",
                ),
            ),
        ),
    )

    assert grounded.conflicts == ()


def test_afternoon_preference_conflicts_with_morning_time() -> None:
    scenario = PatientScenario(
        scenario_id="time-conflict",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
    )

    grounded = ground_turn_meaning(
        scenario=scenario,
        meaning=TurnMeaning(
            stated_facts=(
                FactAssertion(
                    fact="preferred_time",
                    value="9:00 AM",
                ),
            ),
        ),
    )

    assert len(grounded.conflicts) == 1
    assert grounded.conflicts[0].fact == "preferred_time"
