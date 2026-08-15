from voiceprobe.config import Settings
from voiceprobe.run_one import prepare_one_call
from voiceprobe.safety import ALLOWED_TEST_NUMBER


def test_prepare_one_call_contains_exactly_requested_scenario() -> None:
    settings = Settings(
        originating_number="+18402001303",
        dry_run=True,
    )

    manifest = prepare_one_call(
        settings=settings,
        scenario_id="autonomous-phone-diagnostic",
    )

    assert manifest.call_count == 1
    assert manifest.scenario_ids == ("autonomous-phone-diagnostic",)
    assert manifest.destination == ALLOWED_TEST_NUMBER
    assert manifest.dry_run


def test_prepare_one_call_never_expands_to_catalog() -> None:
    settings = Settings(
        originating_number="+18402001303",
        dry_run=True,
    )

    manifest = prepare_one_call(
        settings=settings,
        scenario_id="repetition-clarification",
    )

    assert len(manifest.scenario_ids) == 1
    assert manifest.scenario_ids[0] == "repetition-clarification"
