import pytest
from pydantic import ValidationError

from voiceprobe.scenarios.models import PatientFacts, PatientScenario


def test_builds_patient_scenario() -> None:
    scenario = PatientScenario(
        scenario_id="shoulder-friday",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
        test_targets=(
            "appointment scheduling",
            "complaint correction",
        ),
    )

    assert scenario.facts.complaint == "right shoulder pain"
    assert scenario.facts.duration == "five days"
    assert scenario.facts.preferred_day == "Friday"


def test_normalizes_fact_whitespace() -> None:
    facts = PatientFacts(
        name="  Alex   Morgan ",
        complaint=" right   shoulder pain ",
        duration=" five days ",
    )

    assert facts.name == "Alex Morgan"
    assert facts.complaint == "right shoulder pain"
    assert facts.duration == "five days"


def test_rejects_blank_required_fact() -> None:
    with pytest.raises(ValidationError):
        PatientFacts(
            name="Alex Morgan",
            complaint="   ",
            duration="five days",
        )


def test_rejects_invalid_scenario_id() -> None:
    with pytest.raises(ValidationError):
        PatientScenario(
            scenario_id="Shoulder Friday!",
            objective="Schedule an appointment.",
            facts=PatientFacts(
                name="Alex Morgan",
                complaint="right shoulder pain",
                duration="five days",
            ),
        )


def test_scenario_is_immutable() -> None:
    scenario = PatientScenario(
        scenario_id="shoulder-friday",
        objective="Schedule an appointment.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="right shoulder pain",
            duration="five days",
        ),
    )

    with pytest.raises(ValidationError):
        scenario.facts.complaint = "knee pain"
