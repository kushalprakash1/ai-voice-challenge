import pytest

from voiceprobe.policy import CallPolicy, InvalidCallPolicyError
from voiceprobe.safety import ALLOWED_TEST_NUMBER


def test_policy_uses_only_assessment_destination() -> None:
    policy = CallPolicy(originating_number="+12025550101")

    assert policy.destination == ALLOWED_TEST_NUMBER


def test_dry_run_is_enabled_by_default() -> None:
    policy = CallPolicy(originating_number="+12025550101")

    assert policy.dry_run is True


@pytest.mark.parametrize(
    "originating_number",
    [
        "",
        "2025550101",
        "(202) 555-0101",
        "+1 202 555 0101",
        "+١٤١٥٥٥٥١٢١٢",
        "not-a-number",
        None,
    ],
)
def test_rejects_invalid_originating_numbers(
    originating_number: str | None,
) -> None:
    with pytest.raises(InvalidCallPolicyError):
        CallPolicy(originating_number=originating_number)  # type: ignore[arg-type]


def test_rejects_assessment_number_as_originating_number() -> None:
    with pytest.raises(InvalidCallPolicyError):
        CallPolicy(originating_number=ALLOWED_TEST_NUMBER)


@pytest.mark.parametrize(
    "duration",
    [
        0,
        601,
        1.5,
        True,
        None,
    ],
)
def test_rejects_invalid_call_durations(duration: object) -> None:
    with pytest.raises(InvalidCallPolicyError):
        CallPolicy(
            originating_number="+12025550101",
            max_call_duration_seconds=duration,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "call_count",
    [
        0,
        17,
        100,
        1.5,
        True,
        None,
    ],
)
def test_rejects_invalid_suite_sizes(call_count: object) -> None:
    with pytest.raises(InvalidCallPolicyError):
        CallPolicy(
            originating_number="+12025550101",
            max_suite_calls=call_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "dry_run",
    [
        0,
        1,
        "true",
        "false",
        None,
    ],
)
def test_rejects_non_boolean_dry_run(dry_run: object) -> None:
    with pytest.raises(InvalidCallPolicyError):
        CallPolicy(
            originating_number="+12025550101",
            dry_run=dry_run,  # type: ignore[arg-type]
        )
