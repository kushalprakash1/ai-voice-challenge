"""Deterministic patient-scenario catalog for VoiceProbe.

Scenarios contain immutable ground truth. They describe what the simulated
patient knows and what behavior the call is intended to exercise. Language
models may verbalize these facts, but they may not mutate them.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from voiceprobe.scenarios.models import (
    PatientFacts,
    PatientScenario,
    ProbeKind,
)

DEFAULT_SCENARIO_ID: Final = "autonomous-phone-diagnostic"


SCENARIOS: Final[tuple[PatientScenario, ...]] = (
    PatientScenario(
        scenario_id="autonomous-phone-diagnostic",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Alex Morgan",
            first_name="Alex",
            last_name="Morgan",
            patient_status="a new patient",
            visited_before=False,
            appointment_type="a new patient consultation",
            provider_preference="any available provider",
            complaint="right shoulder pain",
            duration="five days",
            date_of_birth="April 12, 1998",
            insurance="Blue Cross",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
        test_targets=(
            "baseline",
            "insurance",
            "scheduling_preference",
            "booking_confirmation",
        ),
    ),
    PatientScenario(
        scenario_id="identity-insurance-check",
        objective="Schedule an appointment for Monday morning.",
        facts=PatientFacts(
            name="Maya Patel",
            complaint="recurring headaches",
            duration="two weeks",
            date_of_birth="October 3, 2001",
            insurance="Aetna",
            preferred_day="Monday",
            preferred_time="morning",
        ),
        test_targets=(
            "identity",
            "insurance",
            "date_of_birth",
            "booking_confirmation",
        ),
    ),
    PatientScenario(
        scenario_id="morning-slot-preference",
        objective="Schedule an appointment for Tuesday morning.",
        facts=PatientFacts(
            name="Jordan Lee",
            complaint="lower back pain",
            duration="three weeks",
            date_of_birth="June 18, 1995",
            insurance="Kaiser Permanente",
            preferred_day="Tuesday",
            preferred_time="morning",
        ),
        test_targets=(
            "scheduling_preference",
            "morning_offer",
            "offer_acceptance",
        ),
    ),
    PatientScenario(
        scenario_id="evening-slot-preference",
        objective="Schedule an appointment for Thursday evening.",
        facts=PatientFacts(
            name="Sofia Ramirez",
            complaint="left ankle swelling",
            duration="four days",
            date_of_birth="January 27, 1997",
            insurance="Cigna",
            preferred_day="Thursday",
            preferred_time="evening",
        ),
        test_targets=(
            "scheduling_preference",
            "evening_offer",
            "offer_acceptance",
        ),
    ),
    PatientScenario(
        scenario_id="symptom-duration-check",
        objective="Schedule an appointment for Wednesday afternoon.",
        facts=PatientFacts(
            name="Ethan Brooks",
            complaint="persistent cough",
            duration="three weeks",
            date_of_birth="September 14, 1992",
            insurance="UnitedHealthcare",
            preferred_day="Wednesday",
            preferred_time="afternoon",
        ),
        test_targets=(
            "complaint",
            "symptom_duration",
            "scheduling_preference",
        ),
    ),
    PatientScenario(
        scenario_id="dob-verification",
        objective="Schedule an appointment for Friday morning.",
        facts=PatientFacts(
            name="Priya Shah",
            complaint="right wrist pain",
            duration="two days",
            date_of_birth="February 29, 2000",
            insurance="Anthem",
            preferred_day="Friday",
            preferred_time="morning",
        ),
        test_targets=(
            "date_of_birth",
            "identity",
            "insurance",
        ),
    ),
    PatientScenario(
        scenario_id="name-correction",
        objective="Schedule an appointment for Tuesday afternoon.",
        facts=PatientFacts(
            name="Nina O'Connor",
            complaint="neck stiffness",
            duration="six days",
            date_of_birth="May 8, 1989",
            insurance="Blue Shield",
            preferred_day="Tuesday",
            preferred_time="afternoon",
        ),
        test_targets=(
            "identity",
            "name_correction",
            "fact_grounding",
        ),
    ),
    PatientScenario(
        scenario_id="complaint-correction",
        objective="Schedule an appointment for Wednesday morning.",
        facts=PatientFacts(
            name="Marcus Chen",
            complaint="left knee pain",
            duration="nine days",
            date_of_birth="December 11, 1994",
            insurance="Blue Cross",
            preferred_day="Wednesday",
            preferred_time="morning",
        ),
        test_targets=(
            "complaint",
            "fact_correction",
            "fact_grounding",
        ),
    ),
    PatientScenario(
        scenario_id="wrong-day-offer",
        objective="Schedule an appointment for Wednesday morning.",
        facts=PatientFacts(
            name="Olivia Turner",
            complaint="right elbow pain",
            duration="one week",
            date_of_birth="March 21, 1999",
            insurance="Aetna",
            preferred_day="Wednesday",
            preferred_time="morning",
        ),
        test_targets=(
            "wrong_day_offer",
            "offer_rejection",
            "preference_consistency",
        ),
    ),
    PatientScenario(
        scenario_id="wrong-time-offer",
        objective="Schedule an appointment for Monday evening.",
        facts=PatientFacts(
            name="Daniel Kim",
            complaint="upper back soreness",
            duration="four days",
            date_of_birth="July 6, 1996",
            insurance="Cigna",
            preferred_day="Monday",
            preferred_time="evening",
        ),
        test_targets=(
            "wrong_time_offer",
            "offer_rejection",
            "preference_consistency",
        ),
    ),
    PatientScenario(
        scenario_id="repetition-clarification",
        objective="Schedule an appointment for Thursday afternoon.",
        facts=PatientFacts(
            name="Layla Johnson",
            complaint="left shoulder stiffness",
            duration="ten days",
            date_of_birth="November 19, 1993",
            insurance="Kaiser Permanente",
            preferred_day="Thursday",
            preferred_time="afternoon",
        ),
        test_targets=(
            "repetition",
            "clarification",
            "conversation_recovery",
        ),
        probes=(ProbeKind.REQUEST_AGENT_REPEAT_ONCE,),
    ),
    PatientScenario(
        scenario_id="booking-confirmation-robustness",
        objective="Schedule an appointment for Friday afternoon.",
        facts=PatientFacts(
            name="Ava Williams",
            complaint="right knee soreness",
            duration="five days",
            date_of_birth="August 25, 1998",
            insurance="UnitedHealthcare",
            preferred_day="Friday",
            preferred_time="afternoon",
        ),
        test_targets=(
            "booking_confirmation",
            "asr_booking_confirmation",
            "objective_completion",
        ),
        probes=(ProbeKind.VERIFY_BOOKING_BEFORE_END,),
    ),
    PatientScenario(
        scenario_id="medication-refill-correction",
        objective="Request a synthetic lisinopril refill and correct the dose.",
        facts=PatientFacts(
            name="Alex Morgan",
            complaint="medication refill request",
            duration="not applicable",
            preferred_day="not applicable",
            preferred_time="not applicable",
        ),
        test_targets=(
            "medication_refill",
            "dose_correction",
            "correction_retention",
            "fact_grounding",
        ),
    ),
    PatientScenario(
        scenario_id="farthest-date-scheduling",
        objective="Find and, if possible, book the farthest future appointment date currently available.",
        facts=PatientFacts(
            name="Chitragupta Subramnian Singh", first_name="Chitragupta", last_name="Subramnian Singh",
            patient_status="a new patient", visited_before=False,
            appointment_type="a new patient consultation", complaint="right shoulder pain",
            duration="not specified",
            # Call #6 has a selection policy, not a calendar constraint.
            # The farthest-date overlay owns LATEST/FURTHEST selection.
            preferred_day=None,
            preferred_time=None,
        ),
        test_targets=("latest_vs_earliest_intent", "booking_horizon", "farthest_date_selection", "horizon_consistency"),
    ),
    PatientScenario(
        scenario_id="office-hours-location-insurance",
        objective=(
            "Establish self-pay status and adaptively switch between "
            "target-offered office locations."
        ),
        facts=PatientFacts(
            name="Chitragupta Subramnian Singh",
            first_name="Chitragupta",
            last_name="Subramnian Singh",
            complaint="office information request",
            duration="not applicable",
            preferred_day="not applicable",
            preferred_time="not applicable",
        ),
        test_targets=(
            "self_pay_retention",
            "dynamic_location_selection",
            "location_switch_retention",
            "office_hours_consistency",
            "fact_grounding",
        ),
    ),
    PatientScenario(
        scenario_id="doctor-specialist-directory",
        objective="Register the synthetic caller, verify the stored spelling, and investigate one target-offered specialist.",
        facts=PatientFacts(
            name="Gyeong-hyeon Gwak",
            first_name="Gyeong-hyeon",
            last_name="Gwak",
            complaint="doctor directory information request",
            duration="not applicable",
            preferred_day="not applicable",
            preferred_time="not applicable",
        ),
        test_targets=(
            "profile_name_spelling", "grounded_specialist_selection",
            "specialty_identity", "explicit_gender_only", "doctor_location",
            "doctor_hours_consistency", "context_retention",
        ),
    ),
)


_SCENARIO_BY_ID: Final = MappingProxyType(
    {scenario.scenario_id: scenario for scenario in SCENARIOS}
)

if len(_SCENARIO_BY_ID) != len(SCENARIOS):
    raise RuntimeError("Scenario catalog contains duplicate scenario IDs.")


def list_scenarios() -> tuple[PatientScenario, ...]:
    """Return the catalog in deterministic execution order."""
    return SCENARIOS


def scenario_ids() -> tuple[str, ...]:
    """Return valid scenario IDs in deterministic order."""
    return tuple(scenario.scenario_id for scenario in SCENARIOS)


def get_scenario(
    scenario_id: str,
) -> PatientScenario:
    """Resolve one scenario by its stable identifier."""
    if scenario_id == "self-pay-location-switch":
        current = _SCENARIO_BY_ID["office-hours-location-insurance"]
        return PatientScenario(
            scenario_id=scenario_id,
            objective=current.objective,
            facts=current.facts,
            test_targets=current.test_targets,
            probes=current.probes,
        )
    try:
        return _SCENARIO_BY_ID[scenario_id]
    except KeyError as error:
        valid = ", ".join(scenario_ids())

        raise ValueError(
            f"Unknown scenario_id {scenario_id!r}. Valid scenarios: {valid}"
        ) from error
