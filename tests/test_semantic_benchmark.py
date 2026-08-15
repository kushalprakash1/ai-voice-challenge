from voiceprobe.conversation.grounding import ground_turn_meaning
from voiceprobe.conversation.meaning import (
    AppointmentOffer,
    FactAssertion,
    TurnMeaning,
)
from voiceprobe.semantic_benchmark import (
    SemanticCase,
    build_scenario,
    compare_case,
    percentile_95,
)


def test_matching_semantic_case_has_no_failures() -> None:
    case = SemanticCase(
        case_id="test",
        utterance="So your left knee has hurt for two weeks?",
        expected_requested_facts=("complaint", "duration"),
        expected_stated_fact_keys=("complaint", "duration"),
        expected_conflicts=("complaint", "duration"),
    )

    meaning = TurnMeaning(
        requested_facts=("complaint", "duration"),
        stated_facts=(
            FactAssertion(
                fact="complaint",
                value="left knee",
            ),
            FactAssertion(
                fact="duration",
                value="two weeks",
            ),
        ),
    )

    grounded = ground_turn_meaning(
        scenario=build_scenario(),
        meaning=meaning,
    )

    assert (
        compare_case(
            case=case,
            meaning=meaning,
            grounded=grounded,
        )
        == ()
    )


def test_detects_offer_mismatch() -> None:
    case = SemanticCase(
        case_id="offer",
        utterance="Friday at 2:30 is available.",
        expected_appointment_offer={
            "day": "Friday",
            "time": "2:30 PM",
        },
    )

    meaning = TurnMeaning(
        appointment_offer=AppointmentOffer(
            day="Friday",
            time="3:00 PM",
        )
    )

    grounded = ground_turn_meaning(
        scenario=build_scenario(),
        meaning=meaning,
    )

    failures = compare_case(
        case=case,
        meaning=meaning,
        grounded=grounded,
    )

    assert any("appointment_offer" in failure for failure in failures)


def test_percentile_95_uses_nearest_rank() -> None:
    values = [float(number) for number in range(1, 21)]

    assert percentile_95(values) == 19.0
