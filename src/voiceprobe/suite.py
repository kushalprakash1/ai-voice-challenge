"""Pure planning for deterministic VoiceProbe assessment suites.

This module never places calls. It validates which scenarios would run and
applies the existing outbound safety policy before execution is ever allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

from voiceprobe.dialer import build_call_plan
from voiceprobe.policy import CallPolicy
from voiceprobe.scenarios.catalog import list_scenarios
from voiceprobe.scenarios.models import PatientScenario

ASSESSMENT_SUITE_ID = "assessment-v1"
SUITE_CONCURRENCY = 1


class InvalidSuitePlanError(ValueError):
    """Raised when a proposed assessment suite violates planning rules."""


@dataclass(frozen=True, slots=True)
class PlannedScenarioCall:
    """One scenario's position in a validated suite."""

    position: int
    scenario_id: str
    objective: str
    test_targets: tuple[str, ...]
    probes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AssessmentSuitePlan:
    """Validated description of a suite that has not been executed."""

    suite_id: str
    originating_number: str
    destination: str
    calls: tuple[PlannedScenarioCall, ...]
    concurrency: int
    max_call_duration_seconds: int
    max_suite_calls: int
    worst_case_duration_seconds: int
    dry_run: bool
    live_execution_enabled: bool = False

    @property
    def call_count(self) -> int:
        """Return the number of planned calls."""
        return len(self.calls)

    @property
    def worst_case_duration_minutes(self) -> float:
        """Return the maximum theoretical suite duration in minutes."""
        return self.worst_case_duration_seconds / 60.0


def build_suite_plan(
    policy: CallPolicy,
    *,
    scenarios: tuple[PatientScenario, ...] | None = None,
    suite_id: str = ASSESSMENT_SUITE_ID,
) -> AssessmentSuitePlan:
    """Build a validated suite without contacting a telephony provider."""
    selected = list_scenarios() if scenarios is None else scenarios

    if not selected:
        raise InvalidSuitePlanError(
            "Assessment suite must contain at least one scenario."
        )

    if len(selected) > policy.max_suite_calls:
        raise InvalidSuitePlanError(
            "Assessment suite contains "
            f"{len(selected)} calls but policy allows at most "
            f"{policy.max_suite_calls}."
        )

    scenario_ids = tuple(scenario.scenario_id for scenario in selected)

    if len(scenario_ids) != len(set(scenario_ids)):
        raise InvalidSuitePlanError(
            "Assessment suite cannot contain duplicate scenario IDs."
        )

    # Reuse the single-call planner so suite planning passes through the
    # exact same hard destination validation as individual calls.
    call_plan = build_call_plan(policy)

    planned_calls = tuple(
        PlannedScenarioCall(
            position=index,
            scenario_id=scenario.scenario_id,
            objective=scenario.objective,
            test_targets=scenario.test_targets,
            probes=tuple(probe.value for probe in scenario.probes),
        )
        for index, scenario in enumerate(
            selected,
            start=1,
        )
    )

    worst_case_duration_seconds = len(planned_calls) * policy.max_call_duration_seconds

    return AssessmentSuitePlan(
        suite_id=suite_id,
        originating_number=call_plan.originating_number,
        destination=call_plan.destination,
        calls=planned_calls,
        concurrency=SUITE_CONCURRENCY,
        max_call_duration_seconds=(call_plan.max_duration_seconds),
        max_suite_calls=policy.max_suite_calls,
        worst_case_duration_seconds=(worst_case_duration_seconds),
        dry_run=call_plan.dry_run,
        # Planning is intentionally non-executable even when a user's
        # environment has dry_run disabled. A later runner will require
        # a separate explicit execution boundary.
        live_execution_enabled=False,
    )
