from __future__ import annotations

import pytest

from voiceprobe.policy import CallPolicy
from voiceprobe.safety import ALLOWED_TEST_NUMBER
from voiceprobe.scenarios.catalog import (
    list_scenarios,
    scenario_ids,
)
from voiceprobe.suite import (
    ASSESSMENT_SUITE_ID,
    SUITE_CONCURRENCY,
    InvalidSuitePlanError,
    build_suite_plan,
)

ORIGINATING_NUMBER = "+12025550101"


def build_policy(
    *,
    dry_run: bool = True,
    max_suite_calls: int = 16,
    max_call_duration_seconds: int = 180,
) -> CallPolicy:
    return CallPolicy(
        originating_number=ORIGINATING_NUMBER,
        dry_run=dry_run,
        max_suite_calls=max_suite_calls,
        max_call_duration_seconds=(max_call_duration_seconds),
    )


def test_default_suite_contains_entire_catalog() -> None:
    plan = build_suite_plan(build_policy())

    assert tuple(call.scenario_id for call in plan.calls) == scenario_ids()


def test_default_suite_contains_sixteen_calls() -> None:
    plan = build_suite_plan(build_policy())

    assert plan.call_count == 16


def test_suite_reuses_hardcoded_destination() -> None:
    plan = build_suite_plan(build_policy())

    assert plan.destination == ALLOWED_TEST_NUMBER


def test_suite_preserves_originating_number() -> None:
    plan = build_suite_plan(build_policy())

    assert plan.originating_number == ORIGINATING_NUMBER


def test_suite_concurrency_is_one() -> None:
    plan = build_suite_plan(build_policy())

    assert plan.concurrency == SUITE_CONCURRENCY
    assert plan.concurrency == 1


def test_suite_worst_case_duration_is_deterministic() -> None:
    plan = build_suite_plan(build_policy())

    assert plan.worst_case_duration_seconds == 16 * 180
    assert plan.worst_case_duration_minutes == 48.0


def test_suite_plan_is_never_live_executable() -> None:
    plan = build_suite_plan(
        build_policy(
            dry_run=False,
        )
    )

    assert plan.dry_run is False
    assert plan.live_execution_enabled is False


def test_empty_suite_is_rejected() -> None:
    with pytest.raises(
        InvalidSuitePlanError,
        match="at least one scenario",
    ):
        build_suite_plan(
            build_policy(),
            scenarios=(),
        )


def test_suite_larger_than_policy_limit_is_rejected() -> None:
    scenarios = list_scenarios()[:2]

    with pytest.raises(
        InvalidSuitePlanError,
        match="policy allows at most 1",
    ):
        build_suite_plan(
            build_policy(
                max_suite_calls=1,
            ),
            scenarios=scenarios,
        )


def test_duplicate_scenario_ids_are_rejected() -> None:
    scenario = list_scenarios()[0]

    with pytest.raises(
        InvalidSuitePlanError,
        match="duplicate scenario IDs",
    ):
        build_suite_plan(
            build_policy(),
            scenarios=(
                scenario,
                scenario,
            ),
        )


def test_suite_id_is_stable() -> None:
    plan = build_suite_plan(build_policy())

    assert plan.suite_id == ASSESSMENT_SUITE_ID
