import pytest

from voiceprobe.policy import CallPolicy
from voiceprobe.run_one import prepare_one_call


ORIGINATING_NUMBER = "+12025550101"
SCENARIO_ID = "autonomous-phone-diagnostic"


class FakeSettings:
    """Minimal Settings stand-in exposing the production call_policy API."""

    def call_policy(self) -> CallPolicy:
        # Configuration is intentionally dry-run by default. prepare_one_call()
        # must only make it live-capable after an explicit live request.
        return CallPolicy(
            originating_number=ORIGINATING_NUMBER,
            dry_run=True,
        )


def test_prepare_one_call_remains_dry_without_live_request() -> None:
    manifest = prepare_one_call(
        settings=FakeSettings(),  # type: ignore[arg-type]
        scenario_id=SCENARIO_ID,
        live_requested=False,
    )

    assert manifest.dry_run is True
    assert manifest.scenario_ids == (SCENARIO_ID,)
    assert manifest.call_count == 1


def test_prepare_one_call_becomes_live_capable_with_explicit_live_request() -> None:
    manifest = prepare_one_call(
        settings=FakeSettings(),  # type: ignore[arg-type]
        scenario_id=SCENARIO_ID,
        live_requested=True,
    )

    assert manifest.dry_run is False
    assert manifest.scenario_ids == (SCENARIO_ID,)
    assert manifest.call_count == 1


def test_prepare_one_call_rejects_non_boolean_live_request() -> None:
    with pytest.raises(
        TypeError,
        match="live_requested must be a boolean",
    ):
        prepare_one_call(
            settings=FakeSettings(),  # type: ignore[arg-type]
            scenario_id=SCENARIO_ID,
            live_requested=1,  # type: ignore[arg-type]
        )
