from __future__ import annotations

import pytest

from voiceprobe.autonomous_phone import (
    _build_parser,
    build_scenario,
)
from voiceprobe.scenarios.catalog import (
    DEFAULT_SCENARIO_ID,
    SCENARIOS,
    get_scenario,
    list_scenarios,
    scenario_ids,
)


def test_catalog_contains_sixteen_scenarios() -> None:
    assert len(SCENARIOS) == 16


def test_scenario_ids_are_unique() -> None:
    ids = scenario_ids()

    assert len(ids) == len(set(ids))


def test_catalog_order_is_deterministic() -> None:
    assert list_scenarios() == SCENARIOS
    assert scenario_ids() == tuple(scenario.scenario_id for scenario in SCENARIOS)


def test_every_scenario_has_scheduling_truth() -> None:
    for scenario in SCENARIOS:
        if scenario.scenario_id == "farthest-date-scheduling":
            # Its truth is a LATEST selection policy, not a day/daypart
            # compatibility constraint.
            assert scenario.facts.preferred_day is None
            assert scenario.facts.preferred_time is None
            continue
        assert scenario.facts.preferred_day is not None
        assert scenario.facts.preferred_time is not None


def test_every_scenario_has_explicit_test_targets() -> None:
    for scenario in SCENARIOS:
        assert scenario.test_targets
        assert len(scenario.test_targets) == len(set(scenario.test_targets))


def test_default_scenario_preserves_original_patient() -> None:
    scenario = build_scenario()

    assert scenario.scenario_id == DEFAULT_SCENARIO_ID
    assert scenario.facts.name == "Alex Morgan"
    assert scenario.facts.complaint == "right shoulder pain"
    assert scenario.facts.duration == "five days"
    assert scenario.facts.insurance == "Blue Cross"
    assert scenario.facts.preferred_day == "Friday"
    assert scenario.facts.preferred_time == "afternoon"


def test_get_scenario_resolves_by_id() -> None:
    scenario = get_scenario("wrong-time-offer")

    assert scenario.facts.name == "Daniel Kim"
    assert scenario.facts.preferred_day == "Monday"
    assert scenario.facts.preferred_time == "evening"


def test_catalog_accepts_medication_refill_correction() -> None:
    scenario = get_scenario("medication-refill-correction")
    assert scenario.objective.startswith("Request a synthetic lisinopril refill")


def test_catalog_accepts_self_pay_location_switch() -> None:
    scenario = get_scenario("self-pay-location-switch")
    assert "self-pay" in scenario.objective


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown scenario_id",
    ):
        get_scenario("does-not-exist")


def test_phone_parser_accepts_scenario_selection() -> None:
    args = _build_parser().parse_args(
        [
            "--scenario",
            "booking-confirmation-robustness",
        ]
    )

    assert args.scenario == "booking-confirmation-robustness"


def test_phone_parser_defaults_to_original_scenario() -> None:
    args = _build_parser().parse_args([])

    assert args.scenario == DEFAULT_SCENARIO_ID
