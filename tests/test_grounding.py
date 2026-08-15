from voiceprobe.conversation.grounding import ground_turn_meaning
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
        scenario_id="shoulder-friday",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
        ),
    )


def test_detects_incorrect_complaint_and_duration() -> None:
    meaning = TurnMeaning(
        stated_facts=(
            FactAssertion(
                fact="complaint",
                value="left knee pain",
            ),
            FactAssertion(
                fact="duration",
                value="two weeks",
            ),
        )
    )

    grounded = ground_turn_meaning(
        scenario=build_scenario(),
        meaning=meaning,
    )

    assert {conflict.fact for conflict in grounded.conflicts} == {
        "complaint",
        "duration",
    }


def test_correct_claims_do_not_create_conflicts() -> None:
    meaning = TurnMeaning(
        stated_facts=(
            FactAssertion(
                fact="complaint",
                value="right shoulder pain",
            ),
            FactAssertion(
                fact="duration",
                value="five days",
            ),
        )
    )

    grounded = ground_turn_meaning(
        scenario=build_scenario(),
        meaning=meaning,
    )

    assert grounded.conflicts == ()


def test_question_without_claim_has_no_conflict() -> None:
    meaning = TurnMeaning(
        requested_facts=("insurance",),
    )

    grounded = ground_turn_meaning(
        scenario=build_scenario(),
        meaning=meaning,
    )

    assert grounded.conflicts == ()
