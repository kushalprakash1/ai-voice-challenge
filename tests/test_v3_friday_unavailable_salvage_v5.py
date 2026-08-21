from __future__ import annotations

from voiceprobe.v3.fast_policy import RoutineSchedulingPolicy
from voiceprobe.v3.models import DecisionKind


FAILED_FRIDAY_UNAVAILABLE = (
    "There are no Friday afternoon openings between August twenty first "
    "and September twenty first. Would you like to try a different day "
    "or time, or should I check for other options?"
)


def test_exact_failed_friday_unavailable_prompt_is_owned_deterministically() -> None:
    policy = RoutineSchedulingPolicy()

    assert policy.should_relax_day_constraint_for_afternoon(
        FAILED_FRIDAY_UNAVAILABLE
    )

    decision = policy.decide(FAILED_FRIDAY_UNAVAILABLE)

    assert decision.kind == DecisionKind.SEARCH_ALTERNATE_DAY_AFTERNOON
    assert decision.reason == "friday_afternoon_unavailable_choose_alternate_day"
    assert decision.text == "Please check another weekday afternoon."


def test_informational_friday_unavailable_statement_does_not_relax_by_itself() -> None:
    policy = RoutineSchedulingPolicy()

    text = "There are no Friday afternoon openings this week."

    assert not policy.should_relax_day_constraint_for_afternoon(text)

    decision = policy.decide(text)

    assert decision.kind == DecisionKind.WAIT
    assert decision.reason == "informational_availability_statement"


def test_existing_day_vs_provider_branch_is_preserved() -> None:
    policy = RoutineSchedulingPolicy()

    text = (
        "Would you like me to check a different day in the afternoon "
        "or a different provider?"
    )

    assert policy.should_relax_day_constraint_for_afternoon(text)

    decision = policy.decide(text)

    assert decision.kind == DecisionKind.CHOOSE_SEARCH_BRANCH
    assert decision.reason == "choose_earlier_week_afternoon_search"


def test_unrelated_different_day_question_does_not_relax_friday_constraint() -> None:
    policy = RoutineSchedulingPolicy()

    text = "Would a different day work better for your follow-up?"

    assert not policy.should_relax_day_constraint_for_afternoon(text)
