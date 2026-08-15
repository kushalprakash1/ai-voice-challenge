import argparse

import pytest

from voiceprobe.interactive_session import (
    _positive_float,
    build_terminal_scenario,
)


def test_build_terminal_scenario_preserves_patient_truth() -> None:
    scenario = build_terminal_scenario(
        name="Alex Morgan",
        complaint="right shoulder pain",
        duration="five days",
        date_of_birth="April 12, 1998",
        insurance="Blue Cross",
        preferred_day="Friday",
        preferred_time="afternoon",
    )

    assert scenario.facts.name == "Alex Morgan"
    assert scenario.facts.complaint == "right shoulder pain"
    assert scenario.facts.duration == "five days"
    assert scenario.facts.insurance == "Blue Cross"
    assert scenario.facts.preferred_day == "Friday"
    assert scenario.facts.preferred_time == "afternoon"


def test_positive_float_rejects_nonpositive_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_float("0")
