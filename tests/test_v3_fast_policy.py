from voiceprobe.v3.corpus import load_regression_cases
from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy


def test_all_annotated_live_call_regressions() -> None:
    policy = RoutineSchedulingPolicy()

    for case in load_regression_cases():
        decision = policy.decide(case["agent_turn"])

        assert decision.kind.value == case["expected_kind"], (
            case["call_uuid"],
            case["ordinal"],
            case["agent_turn"],
            decision,
        )

        for expected_piece in case["expected_text_contains"]:
            assert expected_piece.casefold() in decision.text.casefold(), (
                case["call_uuid"],
                case["ordinal"],
                expected_piece,
                decision.text,
            )


def test_generic_objective_is_not_used_for_reason_for_visit() -> None:
    decision = RoutineSchedulingPolicy().decide(
        "What is the reason for your visit?"
    )

    assert "shoulder" in decision.text.casefold()
    assert "friday afternoon" not in decision.text.casefold()


def test_provider_choice_uses_stored_provider_preference() -> None:
    decision = RoutineSchedulingPolicy().decide(
        (
            "We have openings on Friday afternoon with two providers. "
            "Would you prefer Dr. A or Dr. B, or is the first available okay?"
        )
    )

    assert decision.text == "First available is fine."
